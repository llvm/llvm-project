from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("google-benchmark")
except PackageNotFoundError:
    # package is not installed
    pass
