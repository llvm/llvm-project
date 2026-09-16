# Reducing Variance

<a name="disabling-cpu-frequency-scaling" />

## Disabling CPU Frequency Scaling

If you see this error:

```
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
```

you might want to disable the CPU frequency scaling while running the
benchmark, as well as consider other ways to stabilize the performance of
your system while benchmarking.

Exactly how to do this depends on the Linux distribution,
desktop environment, and installed programs.  Specific details are a moving
target, so we will not attempt to exhaustively document them here.

One simple option is to use the `cpupower` program to change the
performance governor to "performance".  This tool is maintained along with
the Linux kernel and provided by your distribution.

It must be run as root, like this:

```bash
sudo cpupower frequency-set --governor performance
```

After this you can verify that all CPUs are using the performance governor
by running this command:

```bash
cpupower frequency-info -o proc
```

The benchmarks you subsequently run will have less variance.

<a name="reducing-variance" />

## Disabling ASLR

If you see this error:

```
***WARNING*** ASLR is enabled, the results may have unreproducible noise in them.
```

you might want to disable the ASLR security hardening feature while running the
benchmark.

The simplest way is to add
```
benchmark::MaybeReenterWithoutASLR(argc, argv);
```
as the first line of your `main()` function. It will try to disable ASLR
for the current processor, and, if successful, re-execute the binary.
Note that `personality(2)` may be forbidden by e.g. seccomp (which happens
by default if you are running in a Docker container).

Note that if you link to `benchmark_main` already does that for you.

To globally disable ASLR on Linux, run
```
echo 0 > /proc/sys/kernel/randomize_va_space
```

To run a single benchmark with ASLR disabled on Linux, do:
```
setarch `uname -m` -R ./a_benchmark
```

Note that for the information on how to disable ASLR on other operating systems,
please refer to their documentation.

## Reducing Variance in Benchmarks

The Linux CPU frequency governor [discussed
above](user_guide#disabling-cpu-frequency-scaling) is not the only source
of noise in benchmarks.  Some, but not all, of the sources of variance
include:

1. On multi-core machines not all CPUs/CPU cores/CPU threads run the same
   speed, so running a benchmark one time and then again may give a
   different result depending on which CPU it ran on.
2. CPU scaling features that run on the CPU, like Intel's Turbo Boost and
   AMD Turbo Core and Precision Boost, can temporarily change the CPU
   frequency even when the using the "performance" governor on Linux.
3. Context switching between CPUs, or scheduling competition on the CPU the
   benchmark is running on.
4. Intel Hyperthreading or AMD SMT causing the same issue as above.
5. Cache effects caused by code running on other CPUs.
6. Non-uniform memory architectures (NUMA).

These can cause variance in benchmarks results within a single run
(`--benchmark_repetitions=N`) or across multiple runs of the benchmark
program.

Reducing sources of variance is OS and architecture dependent, which is one
reason some companies maintain machines dedicated to performance testing.

Some of the easier and effective ways of reducing variance on a typical
Linux workstation are:

1. Use the performance governor as [discussed
above](user_guide#disabling-cpu-frequency-scaling).
2. Disable processor boosting by:
   ```sh
   echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost
   ```
   See the Linux kernel's
   [boost.txt](https://www.kernel.org/doc/Documentation/cpu-freq/boost.txt)
   for more information.
3. Set the benchmark program's task affinity to a fixed cpu.  For example:
   ```sh
   taskset -c 0 ./mybenchmark
   ```
4. Increase the program's scheduling priority to minimize context switches using `nice` or `chrt`:
   ```sh
   sudo nice -n -20 ./mybenchmark
   sudo chrt -f 80 ./mybenchmark
   ```
5. Disabling Hyperthreading/SMT.  This can be done in the Bios or using the
   `/sys` file system (see the LLVM project's [Benchmarking
   tips](https://llvm.org/docs/Benchmarking.html)).
6. Close other programs that do non-trivial things based on timers, such as
   your web browser, desktop environment, etc.
7. Reduce the working set of your benchmark to fit within the L1 cache, but
   do be aware that this may lead you to optimize for an unrealistic
   situation.

Further resources on this topic:

1. The LLVM project's [Benchmarking
   tips](https://llvm.org/docs/Benchmarking.html).
1. The Arch Wiki [Cpu frequency
scaling](https://wiki.archlinux.org/title/CPU_frequency_scaling) page.
