"""The public API of this library is defined or imported here."""
import dataclasses
import typing


@dataclasses.dataclass
class BenchmarkRunConfig:
    """Any benchmark runnable by this library must return an instance of this
    class. The `compiler` attribute is optional, for example for python
    benchmarks.
    """

    runner: typing.Callable
    compiler: typing.Optional[typing.Callable] = None
