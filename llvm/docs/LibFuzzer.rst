========================================================
LibFuzzer -- a library for coverage-guided fuzz testing.
========================================================
.. contents::
   :local:
   :depth: 1

Introduction
============

libFuzzer -- library for in-process evolutionary fuzzing of other libraries.

The typical workflow looks like the following.
First, implement a fuzzing target function, like this::

  // fuzz_target.cc
  extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    DoSomethingInterestingWithMyAPI(Data, Size);
    return 0;
  }

Next, build the Fuzzer library as a static archive. Note that libFuzzer contains the `main()` function::

  svn co http://llvm.org/svn/llvm-project/llvm/trunk/lib/Fuzzer
  clang++ -c -g -O2 -std=c++11 Fuzzer/*.cpp -IFuzzer
  ar ruv libFuzzer.a Fuzzer*.o

Then build the target function and the library you are going to test.
You should use SanitizerCoverage_ and one of ASan, MSan, or UBSan.
Link it with `libFuzzer.a`::

  clang -fsanitize-coverage=edge -fsanitize=address your_lib.cc fuzz_target.cc libFuzzer.a -o my_fuzzer

Create a directory with the initial "seed" samlpes.
For some input types libFuzzer will work just fine w/o any seeds,
but for complex inputs this step is very important::

  mkdir CORPUS_DIR
  cp /some/input/samples/* CORPUS_DIR

Finally, run the fuzzer on the `CORPUS_DIR`::

  ./my_fuzzer CORPUS_DIR  # -max_len=1000 -jobs=20 -more_lags=...


As new interesting test cases are discovered they will be added to the corpus.
If a bug is discovered by the sanitizer (ASan, etc) it will be reported as usual and the reproducer
will be written to disk.
Each Fuzzer process is single-threaded (unless the library starts its own
threads). You can run the libFuzzer on the same corpus in multiple processes
in parallel (use the flags `-jobs=N` and `-workers=N`).

libFuzzer is similar in concept to AFL_,
but uses in-process Fuzzing, which is more fragile and restrictive, but
potentially much faster as it has no overhead for process start-up.
It uses LLVM's SanitizerCoverage_ instrumentation to get in-process
coverage-feedback

The code resides in the LLVM repository, requires the fresh Clang compiler to build
and is used to fuzz various parts of LLVM,
but the Fuzzer itself does not (and should not) depend on any
part of LLVM and can be used for other projects w/o requiring the rest of LLVM.

Usage
=====
To run fuzzing pass 0 or more directories. New samples will be written into `dir1`, other directories will be read once during startup.::

./fuzzer [-flag1=val1 [-flag2=val2 ...] ] [dir1 [dir2 ...] ]

To run individual tests without fuzzing pass 1 or more files::

./fuzzer [-flag1=val1 [-flag2=val2 ...] ] file1 [file2 ...]

The most important flags are::

  seed                               	0	Random seed. If 0, seed is generated.
  runs                               	-1	Number of individual test runs (-1 for infinite runs).
  max_len                               0       Maximum length of the test input. If 0, libFuzzer tries to guess a good value based on the corpus and reports it.
  timeout                            	1200	Timeout in seconds (if positive). If one unit runs more than this number of seconds the process will abort.
  timeout_exitcode                     77       Unless abort_on_timeout is set, use this exitcode on timeout.
  max_total_time                        0       If positive, indicates the maximal total time in seconds to run the fuzzer.
  help                               	0	Print help.
  merge                                 0       If 1, the 2-nd, 3-rd, etc corpora will be merged into the 1-st corpus. Only interesting units will be taken.
  jobs                               	0	Number of jobs to run. If jobs >= 1 we spawn this number of jobs in separate worker processes with stdout/stderr redirected to fuzz-JOB.log.
  workers                            	0	Number of simultaneous worker processes to run the jobs. If zero, "min(jobs,NumberOfCpuCores()/2)" is used.
  use_traces                            0       Experimental: use instruction traces
  only_ascii                            0       If 1, generate only ASCII (isprint+isspace) inputs.
  artifact_prefix                       ""      Write fuzzing artifacts (crash, timeout, or slow inputs) as $(artifact_prefix)file
  exact_artifact_path                   ""      Write the single artifact on failure (crash, timeout) as $(exact_artifact_path). This overrides -artifact_prefix and will not use checksum in the file name. Do not use the same path for several parallel processes.
  print_final_stats                     0       If 1, print statistics at exit.
  close_fd_mask                         0       If 1, close stdout at startup; if 2, close stderr; if 3, close both.

For the full list of flags run the fuzzer binary with ``-help=1``.

Usage examples
==============
.. contents::
   :local:
   :depth: 1

Toy example
-----------

A simple function that does something interesting if it receives the input "HI!"::

  cat << EOF >> test_fuzzer.cc
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
  svn co svn://vcs.exim.org/pcre2/code/trunk pcre
  # Build PCRE2 with AddressSanitizer and coverage.
  (cd pcre; ./autogen.sh; CC="clang -fsanitize=address $COV_FLAGS" ./configure --prefix=`pwd`/../inst && make -j && make install)
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
Now, create a directory that will hold the test corpus::

  mkdir -p CORPUS

For simple input languages like regular expressions this is all you need.
For more complicated inputs populate the directory with some input samples.
Now run the fuzzer with the corpus dir as the only parameter::

  ./pcre_fuzzer ./CORPUS

You will see output like this::

  Seed: 1876794929
  #0      READ   cov 0 bits 0 units 1 exec/s 0
  #1      pulse  cov 3 bits 0 units 1 exec/s 0
  #1      INITED cov 3 bits 0 units 1 exec/s 0
  #2      pulse  cov 208 bits 0 units 1 exec/s 0
  #2      NEW    cov 208 bits 0 units 2 exec/s 0 L: 64
  #3      NEW    cov 217 bits 0 units 3 exec/s 0 L: 63
  #4      pulse  cov 217 bits 0 units 3 exec/s 0

* The ``Seed:`` line shows you the current random seed (you can change it with ``-seed=N`` flag).
* The ``READ``  line shows you how many input files were read (since you passed an empty dir there were inputs, but one dummy input was synthesised).
* The ``INITED`` line shows you that how many inputs will be fuzzed.
* The ``NEW`` lines appear with the fuzzer finds a new interesting input, which is saved to the CORPUS dir. If multiple corpus dirs are given, the first one is used.
* The ``pulse`` lines appear periodically to show the current status.

Now, interrupt the fuzzer and run it again the same way. You will see::

  Seed: 1879995378
  #0      READ   cov 0 bits 0 units 564 exec/s 0
  #1      pulse  cov 502 bits 0 units 564 exec/s 0
  ...
  #512    pulse  cov 2933 bits 0 units 564 exec/s 512
  #564    INITED cov 2991 bits 0 units 344 exec/s 564
  #1024   pulse  cov 2991 bits 0 units 344 exec/s 1024
  #1455   NEW    cov 2995 bits 0 units 345 exec/s 1455 L: 49

This time you were running the fuzzer with a non-empty input corpus (564 items).
As the first step, the fuzzer minimized the set to produce 344 interesting items (the ``INITED`` line)

You may run ``N`` independent fuzzer jobs in parallel on ``M`` CPUs::

  N=100; M=4; ./pcre_fuzzer ./CORPUS -jobs=$N -workers=$M

By default (``-reload=1``) the fuzzer processes will periodically scan the CORPUS directory
and reload any new tests. This way the test inputs found by one process will be picked up
by all others.

If ``-workers=$M`` is not supplied, ``min($N,NumberOfCpuCore/2)`` will be used.

Heartbleed
----------
Remember Heartbleed_?
As it was recently `shown <https://blog.hboeck.de/archives/868-How-Heartbleed-couldve-been-found.html>`_,
fuzzing with AddressSanitizer can find Heartbleed. Indeed, here are the step-by-step instructions
to find Heartbleed with LibFuzzer::

  wget https://www.openssl.org/source/openssl-1.0.1f.tar.gz
  tar xf openssl-1.0.1f.tar.gz
  COV_FLAGS="-fsanitize-coverage=edge,indirect-calls" # -fsanitize-coverage=8bit-counters
  (cd openssl-1.0.1f/ && ./config &&
    make -j 32 CC="clang -g -fsanitize=address $COV_FLAGS")
  # Get and build LibFuzzer
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
is now a part of the boringssl source tree.

Advanced features
=================
.. contents::
   :local:
   :depth: 1

Dictionaries
------------
*EXPERIMENTAL*.
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

Data-flow-guided fuzzing
------------------------

*EXPERIMENTAL*.
With an additional compiler flag ``-fsanitize-coverage=trace-cmp`` (see SanitizerCoverageTraceDataFlow_)
and extra run-time flag ``-use_traces=1`` the fuzzer will try to apply *data-flow-guided fuzzing*.
That is, the fuzzer will record the inputs to comparison instructions, switch statements,
and several libc functions (``memcmp``, ``strcmp``, ``strncmp``, etc).
It will later use those recorded inputs during mutations.

This mode can be combined with DataFlowSanitizer_ to achieve better sensitivity.

AFL compatibility
-----------------
LibFuzzer can be used in parallel with AFL_ on the same test corpus.
Both fuzzers expect the test corpus to reside in a directory, one file per input.
You can run both fuzzers on the same corpus in parallel::

  ./afl-fuzz -i testcase_dir -o findings_dir /path/to/program -r @@
  ./llvm-fuzz testcase_dir findings_dir  # Will write new tests to testcase_dir

Periodically restart both fuzzers so that they can use each other's findings.

How good is my fuzzer?
----------------------

Once you implement your target function ``LLVMFuzzerTestOneInput`` and fuzz it to death,
you will want to know whether the function or the corpus can be improved further.
One easy to use metric is, of course, code coverage.
You can get the coverage for your corpus like this::

  ASAN_OPTIONS=coverage=1 ./fuzzer CORPUS_DIR -runs=0

This will run all the tests in the CORPUS_DIR but will not generate any new tests
and dump covered PCs to disk before exiting.
Then you can subtract the set of covered PCs from the set of all instrumented PCs in the binary,
see SanitizerCoverage_ for details.

User-supplied mutators
----------------------

LibFuzzer allows to use custom (user-supplied) mutators,
see FuzzerInterface.h_

Startup initialization
----------------------
If the library being tested needs to be initialized, there are several options.

The simplest way is to have a statically initialized global object::

   static bool Initialized = DoInitialization();

Alternatively, you may define an optional init function and it will receive
the program arguments that you can read and modify::

   extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
    ReadAndMaybeModify(argc, argv);
    return 0;
   }

Try to avoid initialization inside the target function itself as
it will skew the coverage data. Don't do this::

    extern "C" int LLVMFuzzerTestOneInput(...) {
      static bool initialized = false;
      if (!initialized) { 
         ...
      }
    }

Fuzzing components of LLVM
==========================
.. contents::
   :local:
   :depth: 1

clang-format-fuzzer
-------------------
The inputs are random pieces of C++-like text.

Build (make sure to use fresh clang as the host compiler)::

    cmake -GNinja  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_USE_SANITIZER=Address -DLLVM_USE_SANITIZE_COVERAGE=YES -DCMAKE_BUILD_TYPE=Release /path/to/llvm
    ninja clang-format-fuzzer
    mkdir CORPUS_DIR
    ./bin/clang-format-fuzzer CORPUS_DIR

Optionally build other kinds of binaries (asan+Debug, msan, ubsan, etc).

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
similar manner to ``llvm-mc`` and should have two dashes. For example::

  llvm-mc-fuzzer --triple=aarch64-linux-gnu --disassemble --fuzzer-args -max_len=4 -jobs=10

Buildbot
--------

We have a buildbot that runs the above fuzzers for LLVM components
24/7/365 at http://lab.llvm.org:8011/builders/sanitizer-x86_64-linux-fuzzer .

FAQ
=========================

Q. Why libFuzzer does not use any of the LLVM support?
------------------------------------------------------

There are two reasons.

First, we want this library to be used outside of the LLVM w/o users having to
build the rest of LLVM. This may sound unconvincing for many LLVM folks,
but in practice the need for building the whole LLVM frightens many potential
users -- and we want more users to use this code.

Second, there is a subtle technical reason not to rely on the rest of LLVM, or
any other large body of code (maybe not even STL). When coverage instrumentation
is enabled, it will also instrument the LLVM support code which will blow up the
coverage set of the process (since the fuzzer is in-process). In other words, by
using more external dependencies we will slow down the fuzzer while the main
reason for it to exist is extreme speed.

Q. What about Windows then? The Fuzzer contains code that does not build on Windows.
------------------------------------------------------------------------------------

Volunteers are welcome.

Q. When this Fuzzer is not a good solution for a problem?
---------------------------------------------------------

* If the test inputs are validated by the target library and the validator
  asserts/crashes on invalid inputs, in-process fuzzing is not applicable.
* Bugs in the target library may accumulate w/o being detected. E.g. a memory
  corruption that goes undetected at first and then leads to a crash while
  testing another input. This is why it is highly recommended to run this
  in-process fuzzer with all sanitizers to detect most bugs on the spot.
* It is harder to protect the in-process fuzzer from excessive memory
  consumption and infinite loops in the target library (still possible).
* The target library should not have significant global state that is not
  reset between the runs.
* Many interesting target libs are not designed in a way that supports
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

* MUSL LIBC:

  * http://git.musl-libc.org/cgit/musl/commit/?id=39dfd58417ef642307d90306e1c7e50aaec5a35c
  * http://www.openwall.com/lists/oss-security/2015/03/30/3

* `pugixml <https://github.com/zeux/pugixml/issues/39>`_

* PCRE: Search for "LLVM fuzzer" in http://vcs.pcre.org/pcre2/code/trunk/ChangeLog?view=markup;
  also in `bugzilla <https://bugs.exim.org/buglist.cgi?bug_status=__all__&content=libfuzzer&no_redirect=1&order=Importance&product=PCRE&query_format=specific>`_

* `ICU <http://bugs.icu-project.org/trac/ticket/11838>`_

* `Freetype <https://savannah.nongnu.org/search/?words=LibFuzzer&type_of_search=bugs&Search=Search&exact=1#options>`_

* `Harfbuzz <https://github.com/behdad/harfbuzz/issues/139>`_

* `SQLite <http://www3.sqlite.org/cgi/src/info/088009efdd56160b>`_

* `Python <http://bugs.python.org/issue25388>`_

* OpenSSL/BoringSSL: `[1] <https://boringssl.googlesource.com/boringssl/+/cb852981cd61733a7a1ae4fd8755b7ff950e857d>`_ `[2] <https://openssl.org/news/secadv/20160301.txt>`_ `[3] <https://boringssl.googlesource.com/boringssl/+/2b07fa4b22198ac02e0cee8f37f3337c3dba91bc>`_ `[4] <https://boringssl.googlesource.com/boringssl/+/6b6e0b20893e2be0e68af605a60ffa2cbb0ffa64>`_

* `Libxml2
  <https://bugzilla.gnome.org/buglist.cgi?bug_status=__all__&content=libFuzzer&list_id=68957&order=Importance&product=libxml2&query_format=specific>`_

* `Linux Kernel's BPF verifier <https://github.com/iovisor/bpf-fuzzer>`_

* LLVM: `Clang <https://llvm.org/bugs/show_bug.cgi?id=23057>`_, `Clang-format <https://llvm.org/bugs/show_bug.cgi?id=23052>`_, `libc++ <https://llvm.org/bugs/show_bug.cgi?id=24411>`_, `llvm-as <https://llvm.org/bugs/show_bug.cgi?id=24639>`_, Disassembler: http://reviews.llvm.org/rL247405, http://reviews.llvm.org/rL247414, http://reviews.llvm.org/rL247416, http://reviews.llvm.org/rL247417, http://reviews.llvm.org/rL247420, http://reviews.llvm.org/rL247422.

.. _pcre2: http://www.pcre.org/

.. _AFL: http://lcamtuf.coredump.cx/afl/

.. _SanitizerCoverage: http://clang.llvm.org/docs/SanitizerCoverage.html
.. _SanitizerCoverageTraceDataFlow: http://clang.llvm.org/docs/SanitizerCoverage.html#tracing-data-flow
.. _DataFlowSanitizer: http://clang.llvm.org/docs/DataFlowSanitizer.html

.. _Heartbleed: http://en.wikipedia.org/wiki/Heartbleed

.. _FuzzerInterface.h: https://github.com/llvm-mirror/llvm/blob/master/lib/Fuzzer/FuzzerInterface.h
