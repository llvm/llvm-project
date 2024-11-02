<a name="perf-counters" />

# User-Requested Performance Counters

When running benchmarks, the user may choose to request collection of
performance counters. This may be useful in investigation scenarios - narrowing
down the cause of a regression; or verifying that the underlying cause of a
performance improvement matches expectations.

This feature is available if:

* The benchmark is run on an architecture featuring a Performance Monitoring
  Unit (PMU),
* The benchmark is compiled with support for collecting counters. Currently,
  this requires [libpfm](http://perfmon2.sourceforge.net/), which is built as a
  dependency via Bazel.

The feature does not require modifying benchmark code. Counter collection is
handled at the boundaries where timer collection is also handled. 

To opt-in:
* If using a Bazel build, add `--define pfm=1` to your build flags
* If using CMake:
  * Install `libpfm4-dev`, e.g. `apt-get install libpfm4-dev`.
  * Enable the CMake flag `BENCHMARK_ENABLE_LIBPFM` in `CMakeLists.txt`.

To use, pass a comma-separated list of counter names through the
`--benchmark_perf_counters` flag. The names are decoded through libpfm - meaning,
they are platform specific, but some (e.g. `CYCLES` or `INSTRUCTIONS`) are
mapped by libpfm to platform-specifics - see libpfm
[documentation](http://perfmon2.sourceforge.net/docs.html) for more details.

The counter values are reported back through the [User Counters](../README.md#custom-counters)
mechanism, meaning, they are available in all the formats (e.g. JSON) supported
by User Counters.
