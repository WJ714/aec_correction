import os
import re
import setuptools

PKG_NAME = "aec_correction"

HERE = os.path.abspath(os.path.dirname(__file__))

PATTERN = r'^{target}\s*=\s*([\'"])(.+)\1$'

AUTHOR = re.compile(PATTERN.format(target='__author__'), re.M)
VERSION = re.compile(PATTERN.format(target='__version__'), re.M)
LICENSE = re.compile(PATTERN.format(target='__license__'), re.M)

def parse_init():
    with open(os.path.join(HERE, PKG_NAME, '__init__.py')) as f:
        file_data = f.read()
    return [regex.search(file_data).group(2) for regex in
            (AUTHOR, VERSION, LICENSE)]

with open("README.md", "r") as fh:
    long_description = fh.read()

author, version, license = parse_init()

print(setuptools.find_packages())

setuptools.setup(
    name             = PKG_NAME,
    author           = author,
    license          = license,
    version          = version,
    long_description = long_description,
    long_description_content_type = "text/markdown",
    packages         = [PKG_NAME],
    url              = "https://github.com/WJ714/aec_correction",
    python_requires  = '>=3.8',
)
