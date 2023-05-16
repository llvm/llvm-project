# llvm-exegesis

`llvm-exegesis` is a benchmarking tool that accepts or assembles a snippet and
can measure characteristics of that snippet by executing it while keeping track
of performance counters.

### Currently Supported Platforms

`llvm-exegesis` is quite platform-dependent and currently only supports a couple
platform configurations.

##### Currently Supported Operating Systems

Currently, `llvm-exegesis` only supports Linux. This is mainly due to a
dependency on the Linux perf subsystem for reading performance counters.

##### Currently Supported Architectures

Currently, `llvm-exegesis` supports the following architectures:

* x86
  * 64-bit only due to this being the only implemented calling convention
    in `llvm-exegesis` currently.
* ARM
  * AArch64 only
* MIPS
* PowerPC (PowerPC64LE only)

Note that not all functionality is guaranteed to work on all architectures.
