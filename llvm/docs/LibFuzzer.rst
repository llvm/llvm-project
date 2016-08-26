=======================================================
libFuzzer – a library for coverage-guided fuzz testing.
=======================================================
.. contents::
   :local:
   :depth: 1

Introduction
============

LibFuzzer is a library for in-process, coverage-guided, evolutionary fuzzing
of other libraries.

LibFuzzer is similar in concept to American Fuzzy Lop (AFL_), but it performs
all of its fuzzing inside a single process.  This in-process fuzzing can be more
restrictive and fragile, but is potentially much faster as there is no overhead
for process start-up.

The fuzzer is linked with the library under test, and feeds fuzzed inputs to the
library via a specific fuzzing entrypoint (aka "target function"); the fuzzer
then tracks which areas of the code are reached, and generates mutations on the
corpus of input data in order to maximize the code coverage.  The code coverage
information for libFuzzer is provided by LLVM's SanitizerCoverage_
instrumentation.

Contact: libfuzzer(#)googlegroups.com

Versions
========

LibFuzzer is under active development so a current (or at least very recent)
version of Clang is the only supported variant.

(If `building Clang from trunk`_ is too time-consuming or difficult, then
the Clang binaries that the Chromium developers build are likely to be
fairly recent:

.. code-block:: console

  mkdir TMP_CLANG
  cd TMP_CLANG
  git clone https://chromium.googlesource.com/chromium/src/tools/clang
  cd ..
  TMP_CLANG/clang/scripts/update.py

This installs the Clang binary as
``./third_party/llvm-build/Release+Asserts/bin/clang``)

The libFuzzer code resides in the LLVM repository, and requires a recent Clang
compiler to build (and is used to `fuzz various parts of LLVM itself`_).
However the fuzzer itself does not (and should not) depend on any part of LLVM
infrastructure and can be used for other projects without requiring the rest
of LLVM.



Getting Started
===============

.. contents::
   :local:
   :depth: 1

Building
--------

The first step for using libFuzzer on a library is to implement a fuzzing
target function that accepts a sequence of bytes, like this:

.. code-block:: c++

  // fuzz_target.cc
  extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    DoSomethingInterestingWithMyAPI(Data, Size);
    return 0;  // Non-zero return values are reserved for future use.
  }

Next, build the libFuzzer library as a static archive, without any sanitizer
options. Note that the libFuzzer library contains the ``main()`` function:

.. code-block:: console

  svn co http://llvm.org/svn/llvm-project/llvm/trunk/lib/Fuzzer
  # Alternative: get libFuzzer from a dedicated git mirror:
  # git clone https://chromium.googlesource.com/chromium/llvm-project/llvm/lib/Fuzzer
  clang++ -c -g -O2 -std=c++11 Fuzzer/*.cpp -IFuzzer
  ar ruv libFuzzer.a Fuzzer*.o

Then build the fuzzing target function and the library under test using
the SanitizerCoverage_ option, which instruments the code so that the fuzzer
can retrieve code coverage information (to guide the fuzzing).  Linking with
the libFuzzer code then gives an fuzzer executable.

You should also enable one or more of the *sanitizers*, which help to expose
latent bugs by making incorrect behavior generate errors at runtime:

 - AddressSanitizer_ (ASAN) detects memory access errors. Use `-fsanitize=address`.
 - UndefinedBehaviorSanitizer_ (UBSAN) detects the use of various features of C/C++ that are explicitly
   listed as resulting in undefined behavior.  Use `-fsanitize=undefined -fno-sanitize-recover=undefined`
   or any individual UBSAN check, e.g.  `-fsanitize=signed-integer-overflow -fno-sanitize-recover=undefined`.
   You may combine ASAN and UBSAN in one build.
 - MemorySanitizer_ (MSAN) detects uninitialized reads: code whose behavior relies on memory
   contents that have not been initialized to a specific value. Use `-fsanitize=memory`.
   MSAN can not be combined with other sanirizers and should be used as a seprate build.

Finally, link with ``libFuzzer.a``::

  clang -fsanitize-coverage=edge -fsanitize=address your_lib.cc fuzz_target.cc libFuzzer.a -o my_fuzzer

Corpus
------

Coverage-guided fuzzers like libFuzzer rely on a corpus of sample inputs for the
code under test.  This corpus should ideally be seeded with a varied collection
of valid and invalid inputs for the code under test; for example, for a graphics
library the initial corpus might hold a variety of different small PNG/JPG/GIF
files.  The fuzzer generates random mutations based around the sample inputs in
the current corpus.  If a mutation triggers execution of a previously-uncovered
path in the code under test, then that mutation is saved to the corpus for
future variations.

LibFuzzer will work without any initial seeds, but will be less
efficient if the library under test accepts complex,
structured inputs.

The corpus can also act as a sanity/regression check, to confirm that the
fuzzing entrypoint still works and that all of the sample inputs run through
the code under test without problems.

If you have a large corpus (either generated by fuzzing or acquired by other means)
you may want to minimize it while still preserving the full coverage. One way to do that
is to use the `-merge=1` flag:

.. code-block:: console

  mkdir NEW_CORPUS_DIR  # Store minimized corpus here.
  ./my_fuzzer -merge=1 NEW_CORPUS_DIR FULL_CORPUS_DIR

You may use the same flag to add more interesting items to an existing corpus.
Only the inputs that trigger new coverage will be added to the first corpus.

.. code-block:: console

  ./my_fuzzer -merge=1 CURRENT_CORPUS_DIR NEW_POTENTIALLY_INTERESTING_INPUTS_DIR


Running
-------

To run the fuzzer, first create a Corpus_ directory that holds the
initial "seed" sample inputs:

.. code-block:: console

  mkdir CORPUS_DIR
  cp /some/input/samples/* CORPUS_DIR

Then run the fuzzer on the corpus directory:

.. code-block:: console

  ./my_fuzzer CORPUS_DIR  # -max_len=1000 -jobs=20 ...

As the fuzzer discovers new interesting test cases (i.e. test cases that
trigger coverage of new paths through the code under test), those test cases
will be added to the corpus directory.

By default, the fuzzing process will continue indefinitely – at least until
a bug is found.  Any crashes or sanitizer failures will be reported as usual,
stopping the fuzzing process, and the particular input that triggered the bug
will be written to disk (typically as ``crash-<sha1>``, ``leak-<sha1>``,
or ``timeout-<sha1>``).


Parallel Fuzzing
----------------

Each libFuzzer process is single-threaded, unless the library under test starts
its own threads.  However, it is possible to run multiple libFuzzer processes in
parallel with a shared corpus directory; this has the advantage that any new
inputs found by one fuzzer process will be available to the other fuzzer
processes (unless you disable this with the ``-reload=0`` option).

This is primarily controlled by the ``-jobs=N`` option, which indicates that
that `N` fuzzing jobs should be run to completion (i.e. until a bug is found or
time/iteration limits are reached).  These jobs will be run across a set of
worker processes, by default using half of the available CPU cores; the count of
worker processes can be overridden by the ``-workers=N`` option.  For example,
running with ``-jobs=30`` on a 12-core machine would run 6 workers by default,
with each worker averaging 5 bugs by completion of the entire process.


Options
=======

To run the fuzzer, pass zero or more corpus directories as command line
arguments.  The fuzzer will read test inputs from each of these corpus
directories, and any new test inputs that are generated will be written
back to the first corpus directory:

.. code-block:: console

  ./fuzzer [-flag1=val1 [-flag2=val2 ...] ] [dir1 [dir2 ...] ]

If a list of files (rather than directories) are passed to the fuzzer program,
then it will re-run those files as test inputs but will not perform any fuzzing.
In this mode the fuzzer binary can be used as a regression test (e.g. on a
continuous integration system) to check the target function and saved inputs
still work.

The most important command line options are:

``-help``
  Print help message.
``-seed``
  Random seed. If 0 (the default), the seed is generated.
``-runs``
  Number of individual test runs, -1 (the default) to run indefinitely.
``-max_len``
  Maximum length of a test input. If 0 (the default), libFuzzer tries to guess
  a good value based on the corpus (and reports it).
``-timeout``
  Timeout in seconds, default 1200. If an input takes longer than this timeout,
  the process is treated as a failure case.
``-rss_limit_mb``
  Memory usage limit in Mb, default 2048. Use 0 to disable the limit.
  If an input requires more than this amount of RSS memory to execute,
  the process is treated as a failure case.
  The limit is checked in a separate thread every second.
  If running w/o ASAN/MSAN, you may use 'ulimit -v' instead.
``-timeout_exitcode``
  Exit code (default 77) to emit when terminating due to timeout, when
  ``-abort_on_timeout`` is not set.
``-max_total_time``
  If positive, indicates the maximum total time in seconds to run the fuzzer.
  If 0 (the default), run indefinitely.
``-merge``
  If set to 1, any corpus inputs from the 2nd, 3rd etc. corpus directories
  that trigger new code coverage will be merged into the first corpus
  directory.  Defaults to 0. This flag can be used to minimize a corpus.
``-reload``
  If set to 1 (the default), the corpus directory is re-read periodically to
  check for new inputs; this allows detection of new inputs that were discovered
  by other fuzzing processes.
``-jobs``
  Number of fuzzing jobs to run to completion. Default value is 0, which runs a
  single fuzzing process until completion.  If the value is >= 1, then this
  number of jobs performing fuzzing are run, in a collection of parallel
  separate worker processes; each such worker process has its
  ``stdout``/``stderr`` redirected to ``fuzz-<JOB>.log``.
``-workers``
  Number of simultaneous worker processes to run the fuzzing jobs to completion
  in. If 0 (the default), ``min(jobs, NumberOfCpuCores()/2)`` is used.
``-dict``
  Provide a dictionary of input keywords; see Dictionaries_.
``-use_counters``
  Use `coverage counters`_ to generate approximate counts of how often code
  blocks are hit; defaults to 1.
``-use_value_profile``
  Use `value profile`_ to guide corpus expansion; defaults to 0.
``-use_traces``
  Use instruction traces (experimental, defaults to 0); see `Data-flow-guided fuzzing`_.
``-only_ascii``
  If 1, generate only ASCII (``isprint``+``isspace``) inputs. Defaults to 0.
``-artifact_prefix``
  Provide a prefix to use when saving fuzzing artifacts (crash, timeout, or
  slow inputs) as ``$(artifact_prefix)file``.  Defaults to empty.
``-exact_artifact_path``
  Ignored if empty (the default).  If non-empty, write the single artifact on
  failure (crash, timeout) as ``$(exact_artifact_path)``. This overrides
  ``-artifact_prefix`` and will not use checksum in the file name. Do not use
  the same path for several parallel processes.
``-print_pcs``
  If 1, print out newly covered PCs. Defaults to 0.
``-print_final_stats``
  If 1, print statistics at exit.  Defaults to 0.
``-detect_leaks``
  If 1 (default) and if LeakSanitizer is enabled
  try to detect memory leaks during fuzzing (i.e. not only at shut down).
``-close_fd_mask``
  Indicate output streams to close at startup. Be careful, this will
  remove diagnostic output from target code (e.g. messages on assert failure).

   - 0 (default): close neither ``stdout`` nor ``stderr``
   - 1 : close ``stdout``
   - 2 : close ``stderr``
   - 3 : close both ``stdout`` and ``stderr``.

For the full list of flags run the fuzzer binary with ``-help=1``.

Output
======

During operation the fuzzer prints information to ``stderr``, for example::

  INFO: Seed: 3338750330
  Loaded 1024/1211 files from corpus/
  INFO: -max_len is not provided, using 64
  #0	READ   units: 1211 exec/s: 0
  #1211	INITED cov: 2575 bits: 8855 indir: 5 units: 830 exec/s: 1211
  #1422	NEW    cov: 2580 bits: 8860 indir: 5 units: 831 exec/s: 1422 L: 21 MS: 1 ShuffleBytes-
  #1688	NEW    cov: 2581 bits: 8865 indir: 5 units: 832 exec/s: 1688 L: 19 MS: 2 EraseByte-CrossOver-
  #1734	NEW    cov: 2583 bits: 8879 indir: 5 units: 833 exec/s: 1734 L: 27 MS: 3 ChangeBit-EraseByte-ShuffleBytes-
  ...

The early parts of the output include information about the fuzzer options and
configuration, including the current random seed (in the ``Seed:`` line; this
can be overridden with the ``-seed=N`` flag).

Further output lines have the form of an event code and statistics.  The
possible event codes are:

``READ``
  The fuzzer has read in all of the provided input samples from the corpus
  directories.
``INITED``
  The fuzzer has completed initialization, which includes running each of
  the initial input samples through the code under test.
``NEW``
  The fuzzer has created a test input that covers new areas of the code
  under test.  This input will be saved to the primary corpus directory.
``pulse``
  The fuzzer has generated 2\ :sup:`n` inputs (generated periodically to reassure
  the user that the fuzzer is still working).
``DONE``
  The fuzzer has completed operation because it has reached the specified
  iteration limit (``-runs``) or time limit (``-max_total_time``).
``MIN<n>``
  The fuzzer is minimizing the combination of input corpus directories into
  a single unified corpus (due to the ``-merge`` command line option).
``RELOAD``
  The fuzzer is performing a periodic reload of inputs from the corpus
  directory; this allows it to discover any inputs discovered by other
  fuzzer processes (see `Parallel Fuzzing`_).

Each output line also reports the following statistics (when non-zero):

``cov:``
  Total number of code blocks or edges covered by the executing the current
  corpus.
``vp:``
  Size of the `value profile`_.
``bits:``
  Rough measure of the number of code blocks or edges covered, and how often;
  only valid if the fuzzer is run with ``-use_counters=1``.
``indir:``
  Number of distinct function `caller-callee pairs`_ executed with the
  current corpus; only valid if the code under test was built with
  ``-fsanitize-coverage=indirect-calls``.
``units:``
  Number of entries in the current input corpus.
``exec/s:``
  Number of fuzzer iterations per second.

For ``NEW`` events, the output line also includes information about the mutation
operation that produced the new input:

``L:``
  Size of the new input in bytes.
``MS: <n> <operations>``
  Count and list of the mutation operations used to generate the input.


Examples
========
.. contents::
   :local:
   :depth: 1

Toy example
-----------

A simple function that does something interesting if it receives the input
"HI!"::

  cat << EOF > test_fuzzer.cc
  #include <stdint.h>
  #include <stddef.h>
  extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > 0 && data[0] == 'H')
      if (size > 1 && data[1] == 'I')
         if (size > 2 && data[2] == '!')
         __builtin_trap();
    return 0;
  }
  EOF
  # Build test_fuzzer.cc with asan and link against libFuzzer.a
  clang++ -fsanitize=address -fsanitize-coverage=edge test_fuzzer.cc libFuzzer.a
  # Run the fuzzer with no corpus.
  ./a.out

You should get an error pretty quickly::

  #0  READ   units: 1 exec/s: 0
  #1  INITED cov: 3 units: 1 exec/s: 0
  #2  NEW    cov: 5 units: 2 exec/s: 0 L: 64 MS: 0
  #19237  NEW    cov: 9 units: 3 exec/s: 0 L: 64 MS: 0
  #20595  NEW    cov: 10 units: 4 exec/s: 0 L: 1 MS: 4 ChangeASCIIInt-ShuffleBytes-ChangeByte-CrossOver-
  #34574  NEW    cov: 13 units: 5 exec/s: 0 L: 2 MS: 3 ShuffleBytes-CrossOver-ChangeBit-
  #34807  NEW    cov: 15 units: 6 exec/s: 0 L: 3 MS: 1 CrossOver-
  ==31511== ERROR: libFuzzer: deadly signal
  ...
  artifact_prefix='./'; Test unit written to ./crash-b13e8756b13a00cf168300179061fb4b91fefbed


PCRE2
-----

Here we show how to use libFuzzer on something real, yet simple: pcre2_::

  COV_FLAGS=" -fsanitize-coverage=edge,indirect-calls,8bit-counters"
  # Get PCRE2
  wget ftp://ftp.csx.cam.ac.uk/pub/software/programming/pcre/pcre2-10.20.tar.gz
  tar xf pcre2-10.20.tar.gz
  # Build PCRE2 with AddressSanitizer and coverage; requires autotools.
  (cd pcre2-10.20; ./autogen.sh; CC="clang -fsanitize=address $COV_FLAGS" ./configure --prefix=`pwd`/../inst && make -j && make install)
  # Build the fuzzing target function that does something interesting with PCRE2.
  cat << EOF > pcre_fuzzer.cc
  #include <string.h>
  #include <stdint.h>
  #include "pcre2posix.h"
  extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 1) return 0;
    char *str = new char[size+1];
    memcpy(str, data, size);
    str[size] = 0;
    regex_t preg;
    if (0 == regcomp(&preg, str, 0)) {
      regexec(&preg, str, 0, 0, 0);
      regfree(&preg);
    }
    delete [] str;
    return 0;
  }
  EOF
  clang++ -g -fsanitize=address $COV_FLAGS -c -std=c++11  -I inst/include/ pcre_fuzzer.cc
  # Link.
  clang++ -g -fsanitize=address -Wl,--whole-archive inst/lib/*.a -Wl,-no-whole-archive libFuzzer.a pcre_fuzzer.o -o pcre_fuzzer

This will give you a binary of the fuzzer, called ``pcre_fuzzer``.
Now, create a directory that will hold the test corpus:

.. code-block:: console

  mkdir -p CORPUS

For simple input languages like regular expressions this is all you need.
For more complicated/structured inputs, the fuzzer works much more efficiently
if you can populate the corpus directory with a variety of valid and invalid
inputs for the code under test.
Now run the fuzzer with the corpus directory as the only parameter:

.. code-block:: console

  ./pcre_fuzzer ./CORPUS

Initially, you will see Output_ like this::

  INFO: Seed: 2938818941
  INFO: -max_len is not provided, using 64
  INFO: A corpus is not provided, starting from an empty corpus
  #0	READ   units: 1 exec/s: 0
  #1	INITED cov: 3 bits: 3 units: 1 exec/s: 0
  #2	NEW    cov: 176 bits: 176 indir: 3 units: 2 exec/s: 0 L: 64 MS: 0
  #8	NEW    cov: 176 bits: 179 indir: 3 units: 3 exec/s: 0 L: 63 MS: 2 ChangeByte-EraseByte-
  ...
  #14004	NEW    cov: 1500 bits: 4536 indir: 5 units: 406 exec/s: 0 L: 54 MS: 3 ChangeBit-ChangeBit-CrossOver-

Now, interrupt the fuzzer and run it again the same way. You will see::

  INFO: Seed: 3398349082
  INFO: -max_len is not provided, using 64
  #0	READ   units: 405 exec/s: 0
  #405	INITED cov: 1499 bits: 4535 indir: 5 units: 286 exec/s: 0
  #587	NEW    cov: 1499 bits: 4540 indir: 5 units: 287 exec/s: 0 L: 52 MS: 2 InsertByte-EraseByte-
  #667	NEW    cov: 1501 bits: 4542 indir: 5 units: 288 exec/s: 0 L: 39 MS: 2 ChangeBit-InsertByte-
  #672	NEW    cov: 1501 bits: 4543 indir: 5 units: 289 exec/s: 0 L: 15 MS: 2 ChangeASCIIInt-ChangeBit-
  #739	NEW    cov: 1501 bits: 4544 indir: 5 units: 290 exec/s: 0 L: 64 MS: 4 ShuffleBytes-ChangeASCIIInt-InsertByte-ChangeBit-
  ...

On the second execution the fuzzer has a non-empty input corpus (405 items).  As
the first step, the fuzzer minimized this corpus (the ``INITED`` line) to
produce 286 interesting items, omitting inputs that do not hit any additional
code.

(Aside: although the fuzzer only saves new inputs that hit additional code, this
does not mean that the corpus as a whole is kept minimized.  For example, if
an input hitting A-B-C then an input that hits A-B-C-D are generated,
they will both be saved, even though the latter subsumes the former.)


You may run ``N`` independent fuzzer jobs in parallel on ``M`` CPUs:

.. code-block:: console

  N=100; M=4; ./pcre_fuzzer ./CORPUS -jobs=$N -workers=$M

By default (``-reload=1``) the fuzzer processes will periodically scan the corpus directory
and reload any new tests. This way the test inputs found by one process will be picked up
by all others.

If ``-workers=$M`` is not supplied, ``min($N,NumberOfCpuCore/2)`` will be used.

Heartbleed
----------
Remember Heartbleed_?
As it was recently `shown <https://blog.hboeck.de/archives/868-How-Heartbleed-couldve-been-found.html>`_,
fuzzing with AddressSanitizer_ can find Heartbleed. Indeed, here are the step-by-step instructions
to find Heartbleed with libFuzzer::

  wget https://www.openssl.org/source/openssl-1.0.1f.tar.gz
  tar xf openssl-1.0.1f.tar.gz
  COV_FLAGS="-fsanitize-coverage=edge,indirect-calls" # -fsanitize-coverage=8bit-counters
  (cd openssl-1.0.1f/ && ./config &&
    make -j 32 CC="clang -g -fsanitize=address $COV_FLAGS")
  # Get and build libFuzzer
  svn co http://llvm.org/svn/llvm-project/llvm/trunk/lib/Fuzzer
  clang -c -g -O2 -std=c++11 Fuzzer/*.cpp -IFuzzer
  # Get examples of key/pem files.
  git clone   https://github.com/hannob/selftls
  cp selftls/server* . -v
  cat << EOF > handshake-fuzz.cc
  #include <openssl/ssl.h>
  #include <openssl/err.h>
  #include <assert.h>
  #include <stdint.h>
  #include <stddef.h>

  SSL_CTX *sctx;
  int Init() {
    SSL_library_init();
    SSL_load_error_strings();
    ERR_load_BIO_strings();
    OpenSSL_add_all_algorithms();
    assert (sctx = SSL_CTX_new(TLSv1_method()));
    assert (SSL_CTX_use_certificate_file(sctx, "server.pem", SSL_FILETYPE_PEM));
    assert (SSL_CTX_use_PrivateKey_file(sctx, "server.key", SSL_FILETYPE_PEM));
    return 0;
  }
  extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    static int unused = Init();
    SSL *server = SSL_new(sctx);
    BIO *sinbio = BIO_new(BIO_s_mem());
    BIO *soutbio = BIO_new(BIO_s_mem());
    SSL_set_bio(server, sinbio, soutbio);
    SSL_set_accept_state(server);
    BIO_write(sinbio, Data, Size);
    SSL_do_handshake(server);
    SSL_free(server);
    return 0;
  }
  EOF
  # Build the fuzzer.
  clang++ -g handshake-fuzz.cc  -fsanitize=address \
    openssl-1.0.1f/libssl.a openssl-1.0.1f/libcrypto.a Fuzzer*.o
  # Run 20 independent fuzzer jobs.
  ./a.out  -jobs=20 -workers=20

Voila::

  #1048576        pulse  cov 3424 bits 0 units 9 exec/s 24385
  =================================================================
  ==17488==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x629000004748 at pc 0x00000048c979 bp 0x7fffe3e864f0 sp 0x7fffe3e85ca8
  READ of size 60731 at 0x629000004748 thread T0
      #0 0x48c978 in __asan_memcpy
      #1 0x4db504 in tls1_process_heartbeat openssl-1.0.1f/ssl/t1_lib.c:2586:3
      #2 0x580be3 in ssl3_read_bytes openssl-1.0.1f/ssl/s3_pkt.c:1092:4

Note: a `similar fuzzer <https://boringssl.googlesource.com/boringssl/+/HEAD/FUZZING.md>`_
is now a part of the BoringSSL_ source tree.

Advanced features
=================
.. contents::
   :local:
   :depth: 1

Dictionaries
------------
LibFuzzer supports user-supplied dictionaries with input language keywords
or other interesting byte sequences (e.g. multi-byte magic values).
Use ``-dict=DICTIONARY_FILE``. For some input languages using a dictionary
may significantly improve the search speed.
The dictionary syntax is similar to that used by AFL_ for its ``-x`` option::

  # Lines starting with '#' and empty lines are ignored.

  # Adds "blah" (w/o quotes) to the dictionary.
  kw1="blah"
  # Use \\ for backslash and \" for quotes.
  kw2="\"ac\\dc\""
  # Use \xAB for hex values
  kw3="\xF7\xF8"
  # the name of the keyword followed by '=' may be omitted:
  "foo\x0Abar"

Value Profile
---------------

*EXPERIMENTAL*.
With an additional compiler flag ``-fsanitize-coverage=trace-cmp``
(see SanitizerCoverageTraceDataFlow_)
and extra run-time flag ``-use_value_profile=1`` the fuzzer will
collect value profiles for the parameters of compare instructions
and treat some new values as new coverage.

The current imlpementation does roughly the following:

* The compiler instruments all CMP instructions with a callback that receives both CMP arguments.
* The callback computes `(caller_pc&4095) | (popcnt(Arg1 ^ Arg2) << 12)` and uses this value to set a bit in a bitset.
* Every new observed bit in the bitset is treated as new coverage.


This feature has a potential to discover many interesting inputs,
but there are two downsides.
First, the extra instrumentation may bring up to 2x additional slowdown.
Second, the corpus may grow by several times.


Data-flow-guided fuzzing
------------------------

*EXPERIMENTAL*.
With an additional compiler flag ``-fsanitize-coverage=trace-cmp`` (see SanitizerCoverageTraceDataFlow_)
and extra run-time flag ``-use_traces=1`` the fuzzer will try to apply *data-flow-guided fuzzing*.
That is, the fuzzer will record the inputs to comparison instructions, switch statements,
and several libc functions (``memcmp``, ``strcmp``, ``strncmp``, etc).
It will later use those recorded inputs during mutations.

This mode can be combined with DataFlowSanitizer_ to achieve better sensitivity.

Fuzzer-friendly build mode
---------------------------
Sometimes the code under test is not fuzzing-friendly. Examples:

  - The target code uses a PRNG seeded e.g. by system time and
    thus two consequent invocations may potentially execute different code paths
    even if the end result will be the same. This will cause a fuzzer to treat
    two similar inputs as significantly different and it will blow up the test corpus.
    E.g. libxml uses ``rand()`` inside its hash table.
  - The target code uses checksums to protect from invalid inputs.
    E.g. png checks CRC for every chunk.

In many cases it makes sense to build a special fuzzing-friendly build
with certain fuzzing-unfriendly features disabled. We propose to use a common build macro
for all such cases for consistency: ``FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION``.

.. code-block:: c++

  void MyInitPRNG() {
  #ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
    // In fuzzing mode the behavior of the code should be deterministic.
    srand(0);
  #else
    srand(time(0));
  #endif
  }



AFL compatibility
-----------------
LibFuzzer can be used together with AFL_ on the same test corpus.
Both fuzzers expect the test corpus to reside in a directory, one file per input.
You can run both fuzzers on the same corpus, one after another:

.. code-block:: console

  ./afl-fuzz -i testcase_dir -o findings_dir /path/to/program @@
  ./llvm-fuzz testcase_dir findings_dir  # Will write new tests to testcase_dir

Periodically restart both fuzzers so that they can use each other's findings.
Currently, there is no simple way to run both fuzzing engines in parallel while sharing the same corpus dir.

You may also use AFL on your target function ``LLVMFuzzerTestOneInput``:
see an example `here <https://github.com/llvm-mirror/llvm/blob/master/lib/Fuzzer/afl/afl_driver.cpp>`__.

How good is my fuzzer?
----------------------

Once you implement your target function ``LLVMFuzzerTestOneInput`` and fuzz it to death,
you will want to know whether the function or the corpus can be improved further.
One easy to use metric is, of course, code coverage.
You can get the coverage for your corpus like this:

.. code-block:: console

  ASAN_OPTIONS=coverage=1:html_cov_report=1 ./fuzzer CORPUS_DIR -runs=0

This will run all tests in the CORPUS_DIR but will not perform any fuzzing.
At the end of the process it will dump a single html file with coverage information.
See SanitizerCoverage_ for details.

You may also use other ways to visualize coverage,
e.g. using `Clang coverage <http://clang.llvm.org/docs/SourceBasedCodeCoverage.html>`_,
but those will require
you to rebuild the code with different compiler flags.

User-supplied mutators
----------------------

LibFuzzer allows to use custom (user-supplied) mutators,
see FuzzerInterface.h_

Startup initialization
----------------------
If the library being tested needs to be initialized, there are several options.

The simplest way is to have a statically initialized global object inside
`LLVMFuzzerTestOneInput` (or in global scope if that works for you):

.. code-block:: c++

  extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    static bool Initialized = DoInitialization();
    ...

Alternatively, you may define an optional init function and it will receive
the program arguments that you can read and modify. Do this **only** if you
realy need to access ``argv``/``argc``.

.. code-block:: c++

   extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
    ReadAndMaybeModify(argc, argv);
    return 0;
   }


Leaks
-----

Binaries built with AddressSanitizer_ or LeakSanitizer_ will try to detect
memory leaks at the process shutdown.
For in-process fuzzing this is inconvenient
since the fuzzer needs to report a leak with a reproducer as soon as the leaky
mutation is found. However, running full leak detection after every mutation
is expensive.

By default (``-detect_leaks=1``) libFuzzer will count the number of
``malloc`` and ``free`` calls when executing every mutation.
If the numbers don't match (which by itself doesn't mean there is a leak)
libFuzzer will invoke the more expensive LeakSanitizer_
pass and if the actual leak is found, it will be reported with the reproducer
and the process will exit.

If your target has massive leaks and the leak detection is disabled
you will eventually run out of RAM (see the ``-rss_limit_mb`` flag).


Developing libFuzzer
====================

Building libFuzzer as a part of LLVM project and running its test requires
fresh clang as the host compiler and special CMake configuration:

.. code-block:: console

    cmake -GNinja  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_USE_SANITIZER=Address -DLLVM_USE_SANITIZE_COVERAGE=YES -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON /path/to/llvm
    ninja check-fuzzer


Fuzzing components of LLVM
==========================
.. contents::
   :local:
   :depth: 1

To build any of the LLVM fuzz targets use the build instructions above.

clang-format-fuzzer
-------------------
The inputs are random pieces of C++-like text.

.. code-block:: console

    ninja clang-format-fuzzer
    mkdir CORPUS_DIR
    ./bin/clang-format-fuzzer CORPUS_DIR

Optionally build other kinds of binaries (ASan+Debug, MSan, UBSan, etc).

Tracking bug: https://llvm.org/bugs/show_bug.cgi?id=23052

clang-fuzzer
------------

The behavior is very similar to ``clang-format-fuzzer``.

Tracking bug: https://llvm.org/bugs/show_bug.cgi?id=23057

llvm-as-fuzzer
--------------

Tracking bug: https://llvm.org/bugs/show_bug.cgi?id=24639

llvm-mc-fuzzer
--------------

This tool fuzzes the MC layer. Currently it is only able to fuzz the
disassembler but it is hoped that assembly, and round-trip verification will be
added in future.

When run in dissassembly mode, the inputs are opcodes to be disassembled. The
fuzzer will consume as many instructions as possible and will stop when it
finds an invalid instruction or runs out of data.

Please note that the command line interface differs slightly from that of other
fuzzers. The fuzzer arguments should follow ``--fuzzer-args`` and should have
a single dash, while other arguments control the operation mode and target in a
similar manner to ``llvm-mc`` and should have two dashes. For example:

.. code-block:: console

  llvm-mc-fuzzer --triple=aarch64-linux-gnu --disassemble --fuzzer-args -max_len=4 -jobs=10

Buildbot
--------

A buildbot continuously runs the above fuzzers for LLVM components, with results
shown at http://lab.llvm.org:8011/builders/sanitizer-x86_64-linux-fuzzer .

FAQ
=========================

Q. Why doesn't libFuzzer use any of the LLVM support?
-----------------------------------------------------

There are two reasons.

First, we want this library to be used outside of the LLVM without users having to
build the rest of LLVM. This may sound unconvincing for many LLVM folks,
but in practice the need for building the whole LLVM frightens many potential
users -- and we want more users to use this code.

Second, there is a subtle technical reason not to rely on the rest of LLVM, or
any other large body of code (maybe not even STL). When coverage instrumentation
is enabled, it will also instrument the LLVM support code which will blow up the
coverage set of the process (since the fuzzer is in-process). In other words, by
using more external dependencies we will slow down the fuzzer while the main
reason for it to exist is extreme speed.

Q. What about Windows then? The fuzzer contains code that does not build on Windows.
------------------------------------------------------------------------------------

Volunteers are welcome.

Q. When this Fuzzer is not a good solution for a problem?
---------------------------------------------------------

* If the test inputs are validated by the target library and the validator
  asserts/crashes on invalid inputs, in-process fuzzing is not applicable.
* Bugs in the target library may accumulate without being detected. E.g. a memory
  corruption that goes undetected at first and then leads to a crash while
  testing another input. This is why it is highly recommended to run this
  in-process fuzzer with all sanitizers to detect most bugs on the spot.
* It is harder to protect the in-process fuzzer from excessive memory
  consumption and infinite loops in the target library (still possible).
* The target library should not have significant global state that is not
  reset between the runs.
* Many interesting target libraries are not designed in a way that supports
  the in-process fuzzer interface (e.g. require a file path instead of a
  byte array).
* If a single test run takes a considerable fraction of a second (or
  more) the speed benefit from the in-process fuzzer is negligible.
* If the target library runs persistent threads (that outlive
  execution of one test) the fuzzing results will be unreliable.

Q. So, what exactly this Fuzzer is good for?
--------------------------------------------

This Fuzzer might be a good choice for testing libraries that have relatively
small inputs, each input takes < 10ms to run, and the library code is not expected
to crash on invalid inputs.
Examples: regular expression matchers, text or binary format parsers, compression,
network, crypto.

Trophies
========
* GLIBC: https://sourceware.org/glibc/wiki/FuzzingLibc

* MUSL LIBC: `[1] <http://git.musl-libc.org/cgit/musl/commit/?id=39dfd58417ef642307d90306e1c7e50aaec5a35c>`__ `[2] <http://www.openwall.com/lists/oss-security/2015/03/30/3>`__

* `pugixml <https://github.com/zeux/pugixml/issues/39>`_

* PCRE: Search for "LLVM fuzzer" in http://vcs.pcre.org/pcre2/code/trunk/ChangeLog?view=markup;
  also in `bugzilla <https://bugs.exim.org/buglist.cgi?bug_status=__all__&content=libfuzzer&no_redirect=1&order=Importance&product=PCRE&query_format=specific>`_

* `ICU <http://bugs.icu-project.org/trac/ticket/11838>`_

* `Freetype <https://savannah.nongnu.org/search/?words=LibFuzzer&type_of_search=bugs&Search=Search&exact=1#options>`_

* `Harfbuzz <https://github.com/behdad/harfbuzz/issues/139>`_

* `SQLite <http://www3.sqlite.org/cgi/src/info/088009efdd56160b>`_

* `Python <http://bugs.python.org/issue25388>`_

* OpenSSL/BoringSSL: `[1] <https://boringssl.googlesource.com/boringssl/+/cb852981cd61733a7a1ae4fd8755b7ff950e857d>`_ `[2] <https://openssl.org/news/secadv/20160301.txt>`_ `[3] <https://boringssl.googlesource.com/boringssl/+/2b07fa4b22198ac02e0cee8f37f3337c3dba91bc>`_ `[4] <https://boringssl.googlesource.com/boringssl/+/6b6e0b20893e2be0e68af605a60ffa2cbb0ffa64>`_  `[5] <https://github.com/openssl/openssl/pull/931/commits/dd5ac557f052cc2b7f718ac44a8cb7ac6f77dca8>`_ `[6] <https://github.com/openssl/openssl/pull/931/commits/19b5b9194071d1d84e38ac9a952e715afbc85a81>`_

* `Libxml2
  <https://bugzilla.gnome.org/buglist.cgi?bug_status=__all__&content=libFuzzer&list_id=68957&order=Importance&product=libxml2&query_format=specific>`_ and `[HT206167] <https://support.apple.com/en-gb/HT206167>`_ (CVE-2015-5312, CVE-2015-7500, CVE-2015-7942)

* `Linux Kernel's BPF verifier <https://github.com/iovisor/bpf-fuzzer>`_

* Capstone: `[1] <https://github.com/aquynh/capstone/issues/600>`__ `[2] <https://github.com/aquynh/capstone/commit/6b88d1d51eadf7175a8f8a11b690684443b11359>`__

* file:`[1] <http://bugs.gw.com/view.php?id=550>`__  `[2] <http://bugs.gw.com/view.php?id=551>`__  `[3] <http://bugs.gw.com/view.php?id=553>`__  `[4] <http://bugs.gw.com/view.php?id=554>`__

* Radare2: `[1] <https://github.com/revskills?tab=contributions&from=2016-04-09>`__

* gRPC: `[1] <https://github.com/grpc/grpc/pull/6071/commits/df04c1f7f6aec6e95722ec0b023a6b29b6ea871c>`__ `[2] <https://github.com/grpc/grpc/pull/6071/commits/22a3dfd95468daa0db7245a4e8e6679a52847579>`__ `[3] <https://github.com/grpc/grpc/pull/6071/commits/9cac2a12d9e181d130841092e9d40fa3309d7aa7>`__ `[4] <https://github.com/grpc/grpc/pull/6012/commits/82a91c91d01ce9b999c8821ed13515883468e203>`__ `[5] <https://github.com/grpc/grpc/pull/6202/commits/2e3e0039b30edaf89fb93bfb2c1d0909098519fa>`__ `[6] <https://github.com/grpc/grpc/pull/6106/files>`__

* WOFF2: `[1] <https://github.com/google/woff2/commit/a15a8ab>`__

* LLVM: `Clang <https://llvm.org/bugs/show_bug.cgi?id=23057>`_, `Clang-format <https://llvm.org/bugs/show_bug.cgi?id=23052>`_, `libc++ <https://llvm.org/bugs/show_bug.cgi?id=24411>`_, `llvm-as <https://llvm.org/bugs/show_bug.cgi?id=24639>`_, `Demangler <https://bugs.chromium.org/p/chromium/issues/detail?id=606626>`_, Disassembler: http://reviews.llvm.org/rL247405, http://reviews.llvm.org/rL247414, http://reviews.llvm.org/rL247416, http://reviews.llvm.org/rL247417, http://reviews.llvm.org/rL247420, http://reviews.llvm.org/rL247422.

.. _pcre2: http://www.pcre.org/
.. _AFL: http://lcamtuf.coredump.cx/afl/
.. _SanitizerCoverage: http://clang.llvm.org/docs/SanitizerCoverage.html
.. _SanitizerCoverageTraceDataFlow: http://clang.llvm.org/docs/SanitizerCoverage.html#tracing-data-flow
.. _DataFlowSanitizer: http://clang.llvm.org/docs/DataFlowSanitizer.html
.. _AddressSanitizer: http://clang.llvm.org/docs/AddressSanitizer.html
.. _LeakSanitizer: http://clang.llvm.org/docs/LeakSanitizer.html
.. _Heartbleed: http://en.wikipedia.org/wiki/Heartbleed
.. _FuzzerInterface.h: https://github.com/llvm-mirror/llvm/blob/master/lib/Fuzzer/FuzzerInterface.h
.. _3.7.0: http://llvm.org/releases/3.7.0/docs/LibFuzzer.html
.. _building Clang from trunk: http://clang.llvm.org/get_started.html
.. _MemorySanitizer: http://clang.llvm.org/docs/MemorySanitizer.html
.. _UndefinedBehaviorSanitizer: http://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html
.. _`coverage counters`: http://clang.llvm.org/docs/SanitizerCoverage.html#coverage-counters
.. _`value profile`: #value-profile
.. _`caller-callee pairs`: http://clang.llvm.org/docs/SanitizerCoverage.html#caller-callee-coverage
.. _BoringSSL: https://boringssl.googlesource.com/boringssl/
.. _`fuzz various parts of LLVM itself`: `Fuzzing components of LLVM`_
