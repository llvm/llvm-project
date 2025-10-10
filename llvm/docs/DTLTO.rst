===================
DTLTO
===================
.. contents::
   :local:
   :depth: 2

.. toctree::
   :maxdepth: 1

Distributed ThinLTO (DTLTO)
===========================

Distributed ThinLTO (DTLTO) enables the distribution of backend ThinLTO
compilations via external distribution systems, such as Incredibuild, during the
link step.

DTLTO extends the existing ThinLTO distribution support which uses separate
*thin-link*, *backend compilation*, and *link* steps. This method is documented
here:

    https://blog.llvm.org/2016/06/thinlto-scalable-and-incremental-lto.html

Using the *separate thin-link* approach requires a build system capable of
handling the dynamic dependencies specified in the individual summary index
files, such as Bazel. DTLTO removes this requirement, allowing it to be used
with any build process that supports in-process ThinLTO.

The following commands show the steps used for the *separate thin-link*
approach for a basic example:

.. code-block:: console

    1. clang -flto=thin -O2 t1.c t2.c -c
    2. clang -flto=thin -O2 t1.o t2.o -fuse-ld=lld -Wl,--thinlto-index-only
    3. clang -O2 -o t1.native.o t1.o -c -fthinlto-index=t1.o.thinlto.bc
    4. clang -O2 -o t2.native.o t2.o -c -fthinlto-index=t2.o.thinlto.bc
    5. clang t1.native.o t2.native.o -o a.out -fuse-ld=lld

With DTLTO, steps 2-5 are performed internally as part of the link step. The
equivalent DTLTO commands for the above are:

.. code-block:: console

    clang -flto=thin -O2 t1.c t2.c -c
    clang -flto=thin -O2 t1.o t2.o -fuse-ld=lld -fthinlto-distributor=<distributor_process>

For DTLTO, LLD prepares the following for each ThinLTO backend compilation job:

- An individual index file and a list of input and output files (corresponds to
  step 2 above).
- A Clang command line to perform the ThinLTO backend compilations.

This information is supplied, via a JSON file, to ``distributor_process``, which
executes the backend compilations using a distribution system (corresponds to
steps 3 and 4 above). Upon completion, LLD integrates the compiled native object
files into the link process and completes the link (corresponds to step 5
above).

This design keeps the details of distribution systems out of the LLVM source
code.

An example distributor that performs all work on the local system is included in
the LLVM source tree. To run an example with that distributor, a command line
such as the following can be used:

.. code-block:: console

   clang -flto=thin -fuse-ld=lld -O2 t1.o t2.o -fthinlto-distributor=$(which python3) \
     -Xthinlto-distributor=$LLVMSRC/llvm/utils/dtlto/local.py

Distributors
------------

Distributors are programs responsible for:

1. Consuming the JSON backend compilations job description file.
2. Translating job descriptions into requests for the distribution system.
3. Blocking execution until all backend compilations are complete.

Distributors must return a non-zero exit code on failure. They can be
implemented as platform native executables or in a scripting language, such as
Python.

Clang and LLD provide options to specify a distributor program for managing
backend compilations. Distributor options and backend compilation options can
also be specified. Such options are transparently forwarded.

The backend compilations are currently performed by invoking Clang. For further
details, refer to:

* Clang documentation: https://clang.llvm.org/docs/ThinLTO.html
* LLD documentation: https://lld.llvm.org/DTLTO.html

When invoked with a distributor, LLD generates a JSON file describing the
backend compilation jobs and executes the distributor, passing it this file.

JSON Schema
-----------

The JSON format is explained by reference to the following example, which
describes the backend compilation of the modules ``t1.o`` and ``t2.o``:

.. code-block:: json

    {
        "common": {
            "linker_output": "dtlto.elf",
            "args": ["/usr/bin/clang", "-O2", "-c", "-fprofile-sample-use=my.prof"],
            "inputs": ["my.prof"]
        },
        "jobs": [
            {
                "args": ["t1.o", "-fthinlto-index=t1.o.thinlto.bc", "-o", "t1.native.o", "-fproc-stat-report=t1.stats.txt"],
                "inputs": ["t1.o", "t1.o.thinlto.bc"],
                "outputs": ["t1.native.o", "t1.stats.txt"]
            },
            {
                "args": ["t2.o", "-fthinlto-index=t2.o.thinlto.bc", "-o", "t2.native.o", "-fproc-stat-report=t2.stats.txt"],
                "inputs": ["t2.o", "t2.o.thinlto.bc"],
                "outputs": ["t2.native.o", "t2.stats.txt"]
            }
        ]
    }

Each entry in the ``jobs`` array represents a single backend compilation job.
Each job object records its own command-line arguments and input/output files.
Shared arguments and inputs are defined once in the ``common`` object.

Reserved Entries:

- The first entry in the ``common.args`` array specifies the compiler
  executable to invoke.
- The first entry in each job's ``inputs`` array is the bitcode file for the
  module being compiled.
- The second entry in each job's ``inputs`` array is the corresponding
  individual summary index file.
- The first entry in each job's ``outputs`` array is the primary output object
  file.

For the ``outputs`` array, only the first entry is reserved for the primary
output file; there is no guaranteed order for the remaining entries. The primary
output file is specified in a reserved entry because some distribution systems
rely on this path - for example, to provide a meaningful user label for
compilation jobs. Initially, the DTLTO implementation will not produce more than
one output file. However, in the future, if LTO options are added that imply
additional output files, those files will also be included in this array.

Command-line arguments and input/output files are stored separately to allow
the remote compiler to be changed without updating the distributors, as the
distributors do not need to understand the details of the compiler command
line.

To generate the backend compilation commands, the common and job-specific
arguments are concatenated.

When consuming the example JSON above, a distributor is expected to issue the
following backend compilation commands with maximum parallelism:

.. code-block:: console

    /usr/bin/clang -O2 -c -fprofile-sample-use=my.prof t1.o -fthinlto-index=t1.o.thinlto.bc -o t1.native.o \
      -fproc-stat-report=t1.stats.txt

    /usr/bin/clang -O2 -c -fprofile-sample-use=my.prof t2.o  -fthinlto-index=t2.o.thinlto.bc -o t2.native.o \
      -fproc-stat-report=t2.stats.txt

TODOs
-----

The following features are planned for DTLTO but not yet implemented:

- Support for the ThinLTO in-process cache.
- Support for platforms other than ELF and COFF.
- Support for archives with bitcode members.
- Support for more LTO configurations; only a very limited set of LTO
  configurations is supported currently, e.g., support for basic block sections
  is not currently available.

Constraints
-----------

- Matching versions of Clang and LLD should be used.
- The distributor used must support the JSON schema generated by the version of
  LLD in use.

