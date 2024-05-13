.. llvm-libgcc:

===========
llvm-libgcc
===========

.. contents::
  :local:

**Note that these instructions assume a Linux and bash-friendly environment.
YMMV if you’re on a non Linux-based platform.**

.. _introduction:

Motivation
============

Enabling libunwind as a replacement for libgcc on Linux has proven to be
challenging since libgcc_s.so is a required dependency in the [Linux standard
base][5]. Some software is transitively dependent on libgcc because glibc makes
hardcoded calls to functions in libgcc_s. For example, the function
``__GI___backtrace`` eventually makes its way to a [hardcoded dlopen to libgcc_s'
_Unwind_Backtrace][1]. Since libgcc_{eh.a,s.so} and libunwind have the same ABI,
but different implementations, the two libraries end up [cross-talking, which
ultimately results in a segfault][2].

To solve this problem, libunwind needs libgcc "front" that is, link the
necessary functions from compiler-rt and libunwind into an archive and shared
object that advertise themselves as ``libgcc.a``, ``libgcc_eh.a``, and
``libgcc_s.so``, so that glibc’s baked calls are diverted to the correct objects
in memory. Fortunately for us, compiler-rt and libunwind use the same ABI as the
libgcc family, so the problem is solvable at the llvm-project configuration
level: no program source needs to be edited. Thus, the end result is for a
distro manager to configure their LLVM build with a flag that indicates they
want to archive compiler-rt/unwind as libgcc. We achieve this by compiling
libunwind with all the symbols necessary for compiler-rt to emulate the libgcc
family, and then generate symlinks named for our "libgcc" that point to their
corresponding libunwind counterparts.

.. _alternatives

Alternatives
============

We alternatively considered patching glibc so that the source doesn't directly
refer to libgcc, but rather _defaults_ to libgcc, so that a system preferring
compiler-rt/libunwind can point to these libraries at the config stage instead.
Even if we modified the Linux standard base, this alternative won't work because
binaries that are built using libgcc will still end up having cross-talk between
the differing implementations.

.. _target audience:

Target audience
===============

llvm-libgcc is not for the casual LLVM user. It is intended to be used by distro
managers who want to replace libgcc with compiler-rt and libunwind, but cannot
fully abandon the libgcc family (e.g. because they are dependent on glibc). Such
managers must have worked out their compatibility requirements ahead of using
llvm-libgcc.

.. _cmake options:

CMake options
=============

.. option:: `LLVM_LIBGCC_EXPLICIT_OPT_IN`

  **Required**

  Since llvm-libgcc is such a fundamental, low-level component, we have made it
  difficult to accidentally build, by requiring you to set an opt-in flag.

.. _Building llvm-libgcc

Building llvm-libgcc
--------------------

The first build tree is a mostly conventional build tree and gets you a Clang
build with these compiler-rt symbols exposed.

.. code-block:: bash
  # Assumes $(PWD) is /path/to/llvm-project
  $ cmake -GNinja -S llvm -B build-primary                    \
      -DCMAKE_BUILD_TYPE=Release                              \
      -DCMAKE_CROSSCOMPILING=On                               \
      -DCMAKE_INSTALL_PREFIX=/tmp/aarch64-unknown-linux-gnu   \
      -DLLVM_ENABLE_PROJECTS='clang'                          \
      -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;llvm-libgcc"   \
      -DLLVM_TARGETS_TO_BUILD=AArch64                         \
      -DLLVM_DEFAULT_TARGET_TRIPLE=aarch64-unknown-linux-gnu  \
      -DLLVM_LIBGCC_EXPLICIT_OPT_IN=Yes
  $ ninja -C build-primary install

It's very important to notice that neither ``compiler-rt``, nor ``libunwind``,
are listed in ``LLVM_ENABLE_RUNTIMES``. llvm-libgcc makes these subprojects, and
adding them to this list will cause you problems due to there being duplicate
targets. As such, configuring the runtimes build will reject explicitly mentioning
either project with ``llvm-libgcc``.

To avoid issues when building with ``-DLLVM_ENABLE_RUNTIMES=all``, ``llvm-libgcc``
is not included, and all runtimes targets must be manually listed.

## Verifying your results

This gets you a copy of libunwind with the libgcc symbols. You can verify this
using ``readelf``.

.. code-block:: bash

  $ llvm-readelf -W --dyn-syms "${LLVM_LIBGCC_SYSROOT}/lib/libunwind.so" | grep FUNC | grep GCC_3.0


Roughly sixty symbols should appear, all suffixed with ``@@GCC_3.0``. You can
replace ``GCC_3.0`` with any of the supported version names in the version
script you’re exporting to verify that the symbols are exported.


.. _supported platforms:

Supported platforms
===================

llvm-libgcc currently supports the following target triples:

* ``aarch64-*-*-*``
* ``armv7a-*-*-gnueabihf``
* ``i386-*-*-*``
* ``x86_64-*-*-*``

If you would like to support another triple (e.g. ``powerpc64-*-*-*``), you'll
need to generate a new version script, and then edit ``lib/gcc_s.ver``.

.. _Generating a new version script

Generating a new version script
-------------------------------

To generate a new version script, we need to generate the list of symbols that
exist in the set (``clang-builtins.a`` ∪ ``libunwind.a``) ∩ ``libgcc_s.so.1``.
The prerequisites for generating a version script are a binaries for the three
aforementioned libraries targeting your architecture (without having built
llvm-libgcc).

Once these libraries are in place, to generate a new version script, run the
following command.

.. code-block:: bash

  /path/to/llvm-project
  $ export ARCH=powerpc64
  $ llvm/tools/llvm-libgcc/generate_version_script.py       \
      --compiler_rt=/path/to/libclang_rt.builtins-${ARCH}.a \
      --libunwind=/path/to/libunwind.a                      \
      --libgcc_s=/path/to/libgcc_s.so.1                     \
      --output=${ARCH}

This will generate a new version script a la
``/path/to/llvm-project/llvm/tools/llvm-libgcc/gcc_s-${ARCH}.ver``, which we use
in the next section.

.. _Editing ``lib/gcc_s.ver``

Editing ``lib/gcc_s.ver``
-------------------------

Our freshly generated version script is unique to the specific architecture that
it was generated for, but a lot of the symbols are shared among many platforms.
As such, we don't check in unique version scripts, but rather have a single
version script that's run through the C preprocessor to prune symbols we won't
be using in ``lib/gcc_s.ver``.

Working out which symbols are common is largely a manual process at the moment,
because some symbols may be shared across different architectures, but not in
the same versions of libgcc. As such, a symbol appearing in ``lib/gcc_s.ver``
doesn't guarantee that the symbol is available for our new architecture: we need
to verify that the versions are the same, and if they're not, add the symbol to
the new version section, with the appropriate include guards.

There are a few macros that aim to improve readability.

* ``ARM_GNUEABIHF``, which targets exactly ``arm-*-*-gnueabihf``.
* ``GLOBAL_X86``, which should be used to target both x86 and x86_64, regardless
  of the triple.
* ``GLOBAL_32BIT``, which is be used to target 32-bit platforms.
* ``GLOBAL_64BIT``, which is be used to target 64-bit platforms.
