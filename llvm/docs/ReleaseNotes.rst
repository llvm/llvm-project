========================
LLVM 4.0.0 Release Notes
========================

.. contents::
    :local:

Introduction
============

This document contains the release notes for the LLVM Compiler Infrastructure,
release 4.0.0.  Here we describe the status of LLVM, including major improvements
from the previous release, improvements in various subprojects of LLVM, and
some of the current users of the code.  All LLVM releases may be downloaded
from the `LLVM releases web site <http://llvm.org/releases/>`_.

For more information about LLVM, including information about the latest
release, please check out the `main LLVM web site <http://llvm.org/>`_.  If you
have questions or comments, the `LLVM Developer's Mailing List
<http://lists.llvm.org/mailman/listinfo/llvm-dev>`_ is a good place to send
them.

New Versioning Scheme
=====================
Starting with this release, LLVM is using a
`new versioning scheme <http://blog.llvm.org/2016/12/llvms-new-versioning-scheme.html>`_,
increasing the major version number with each major release. Stable updates to
this release will be versioned 4.0.x, and the next major release, six months
from now, will be version 5.0.0.

Non-comprehensive list of changes in this release
=================================================
* The minimum compiler version required for building LLVM has been raised to
  4.8 for GCC and 2017 for Visual Studio.

* The C API functions ``LLVMAddFunctionAttr``, ``LLVMGetFunctionAttr``,
  ``LLVMRemoveFunctionAttr``, ``LLVMAddAttribute``, ``LLVMRemoveAttribute``,
  ``LLVMGetAttribute``, ``LLVMAddInstrAttribute`` and
  ``LLVMRemoveInstrAttribute`` have been removed.

* The C API enum ``LLVMAttribute`` has been deleted.

* The definition and uses of ``LLVM_ATRIBUTE_UNUSED_RESULT`` in the LLVM source
  were replaced with ``LLVM_NODISCARD``, which matches the C++17 ``[[nodiscard]]``
  semantics rather than gcc's ``__attribute__((warn_unused_result))``.

* The Timer related APIs now expect a Name and Description. When upgrading code
  the previously used names should become descriptions and a short name in the
  style of a programming language identifier should be added.

* LLVM now handles ``invariant.group`` across different basic blocks, which makes
  it possible to devirtualize virtual calls inside loops.

* The aggressive dead code elimination phase ("adce") now removes
  branches which do not effect program behavior. Loops are retained by
  default since they may be infinite but these can also be removed
  with LLVM option ``-adce-remove-loops`` when the loop body otherwise has
  no live operations.

* The GVNHoist pass is now enabled by default. The new pass based on Global
  Value Numbering detects similar computations in branch code and replaces
  multiple instances of the same computation with a unique expression.  The
  transform benefits code size and generates better schedules.  GVNHoist is
  more aggressive at ``-Os`` and ``-Oz``, hoisting more expressions at the
  expense of execution time degradations.

 * The llvm-cov tool can now export coverage data as json. Its html output mode
   has also improved.

Improvements to ThinLTO (-flto=thin)
------------------------------------
Integration with profile data (PGO). When available, profile data
enables more accurate function importing decisions, as well as
cross-module indirect call promotion.

Significant build-time and binary-size improvements when compiling with
debug info (-g).

LLVM Coroutines
---------------

Experimental support for :doc:`Coroutines` was added, which can be enabled
with ``-enable-coroutines`` in ``opt`` the command tool or using the
``addCoroutinePassesToExtensionPoints`` API when building the optimization
pipeline.

For more information on LLVM Coroutines and the LLVM implementation, see
`2016 LLVM Developers’ Meeting talk on LLVM Coroutines
<http://llvm.org/devmtg/2016-11/#talk4>`_.

Regcall and Vectorcall Calling Conventions
--------------------------------------------------

Support was added for ``_regcall`` calling convention.
Existing ``__vectorcall`` calling convention support was extended to include
correct handling of HVAs.

The ``__vectorcall`` calling convention was introduced by Microsoft to
enhance register usage when passing parameters.
For more information please read `__vectorcall documentation
<https://msdn.microsoft.com/en-us/library/dn375768.aspx>`_.

The ``__regcall`` calling convention was introduced by Intel to
optimize parameter transfer on function call.
This calling convention ensures that as many values as possible are
passed or returned in registers.
For more information please read `__regcall documentation
<https://software.intel.com/en-us/node/693069>`_.

Code Generation Testing
-----------------------

Passes that work on the machine instruction representation can be tested with
the .mir serialization format. ``llc`` supports the ``-run-pass``,
``-stop-after``, ``-stop-before``, ``-start-after``, ``-start-before`` to
run a single pass of the code generation pipeline, or to stop or start the code
generation pipeline at a given point.

Additional information can be found in the :doc:`MIRLangRef`. The format is
used by the tests ending in ``.mir`` in the ``test/CodeGen`` directory.

This feature is available since 2015. It is used more often lately and was not
mentioned in the release notes yet.

Intrusive list API overhaul
---------------------------

The intrusive list infrastructure was substantially rewritten over the last
couple of releases, primarily to excise undefined behaviour.  The biggest
changes landed in this release.

* ``simple_ilist<T>`` is a lower-level intrusive list that never takes
  ownership of its nodes.  New intrusive-list clients should consider using it
  instead of ``ilist<T>``.

  * ``ilist_tag<class>`` allows a single data type to be inserted into two
    parallel intrusive lists.  A type can inherit twice from ``ilist_node``,
    first using ``ilist_node<T,ilist_tag<A>>`` (enabling insertion into
    ``simple_ilist<T,ilist_tag<A>>``) and second using
    ``ilist_node<T,ilist_tag<B>>`` (enabling insertion into
    ``simple_ilist<T,ilist_tag<B>>``), where ``A`` and ``B`` are arbitrary
    types.

  * ``ilist_sentinel_tracking<bool>`` controls whether an iterator knows
    whether it's pointing at the sentinel (``end()``).  By default, sentinel
    tracking is on when ABI-breaking checks are enabled, and off otherwise;
    this is used for an assertion when dereferencing ``end()`` (this assertion
    triggered often in practice, and many backend bugs were fixed).  Explicitly
    turning on sentinel tracking also enables ``iterator::isEnd()``.  This is
    used by ``MachineInstrBundleIterator`` to iterate over bundles.

* ``ilist<T>`` is built on top of ``simple_ilist<T>``, and supports the same
  configuration options.  As before (and unlike ``simple_ilist<T>``),
  ``ilist<T>`` takes ownership of its nodes.  However, it no longer supports
  *allocating* nodes, and is now equivalent to ``iplist<T>``.  ``iplist<T>``
  will likely be removed in the future.

  * ``ilist<T>`` now always uses ``ilist_traits<T>``.  Instead of passing a
    custom traits class in via a template parameter, clients that want to
    customize the traits should specialize ``ilist_traits<T>``.  Clients that
    want to avoid ownership can specialize ``ilist_alloc_traits<T>`` to inherit
    from ``ilist_noalloc_traits<T>`` (or to do something funky); clients that
    need callbacks can specialize ``ilist_callback_traits<T>`` directly.

* The underlying data structure is now a simple recursive linked list.  The
  sentinel node contains only a "next" (``begin()``) and "prev" (``rbegin()``)
  pointer and is stored in the same allocation as ``simple_ilist<T>``.
  Previously, it was malloc-allocated on-demand by default, although the
  now-defunct ``ilist_sentinel_traits<T>`` was sometimes specialized to avoid
  this.

* The ``reverse_iterator`` class no longer uses ``std::reverse_iterator``.
  Instead, it now has a handle to the same node that it dereferences to.
  Reverse iterators now have the same iterator invalidation semantics as
  forward iterators.

  * ``iterator`` and ``reverse_iterator`` have explicit conversion constructors
    that match ``std::reverse_iterator``'s off-by-one semantics, so that
    reversing the end points of an iterator range results in the same range
    (albeit in reverse).  I.e., ``reverse_iterator(begin())`` equals
    ``rend()``.

  * ``iterator::getReverse()`` and ``reverse_iterator::getReverse()`` return an
    iterator that dereferences to the *same* node.  I.e.,
    ``begin().getReverse()`` equals ``--rend()``.

  * ``ilist_node<T>::getIterator()`` and
    ``ilist_node<T>::getReverseIterator()`` return the forward and reverse
    iterators that dereference to the current node.  I.e.,
    ``begin()->getIterator()`` equals ``begin()`` and
    ``rbegin()->getReverseIterator()`` equals ``rbegin()``.

* ``iterator`` now stores an ``ilist_node_base*`` instead of a ``T*``.  The
  implicit conversions between ``ilist<T>::iterator`` and ``T*`` have been
  removed.  Clients may use ``N->getIterator()`` (if not ``nullptr``) or
  ``&*I`` (if not ``end()``); alternatively, clients may refactor to use
  references for known-good nodes.

Changes to the ARM Targets
--------------------------

**During this release the AArch64 target has:**

* Gained support for ILP32 relocations.
* Gained support for XRay.
* Made even more progress on GlobalISel. There is still some work left before
  it is production-ready though.
* Refined the support for Qualcomm's Falkor and Samsung's Exynos CPUs.
* Learned a few new tricks for lowering multiplications by constants, folding
  spilled/refilled copies etc.

**During this release the ARM target has:**

* Gained support for ROPI (read-only position independence) and RWPI
  (read-write position independence), which can be used to remove the need for
  a dynamic linker.
* Gained support for execute-only code, which is placed in pages without read
  permissions.
* Gained a machine scheduler for Cortex-R52.
* Gained support for XRay.
* Gained Thumb1 implementations for several compiler-rt builtins. It also
  has some support for building the builtins for HF targets.
* Started using the generic bitreverse intrinsic instead of rbit.
* Gained very basic support for GlobalISel.

A lot of work has also been done in LLD for ARM, which now supports more
relocations and TLS.

Note: From the next release (5.0), the "vulcan" target will be renamed to
"thunderx2t99", including command line options, assembly directives, etc. This
release (4.0) will be the last one to accept "vulcan" as its name.

Changes to the AVR Target
-----------------------------

This marks the first release where the AVR backend has been completely merged
from a fork into LLVM trunk. The backend is still marked experimental, but
is generally quite usable. All downstream development has halted on
`GitHub <https://github.com/avr-llvm/llvm>`_, and changes now go directly into
LLVM trunk.

* Instruction selector and pseudo instruction expansion pass landed
* `read_register` and `write_register` intrinsics are now supported
* Support stack stores greater than 63-bytes from the bottom of the stack
* A number of assertion errors have been fixed
* Support stores to `undef` locations
* Very basic support for the target has been added to clang
* Small optimizations to some 16-bit boolean expressions

Most of the work behind the scenes has been on correctness of generated
assembly, and also fixing some assertions we would hit on some well-formed
inputs.

Changes to the MIPS Target
-----------------------------

**During this release the MIPS target has:**

* IAS is now enabled by default for Debian mips64el.
* Added support for the two operand form for many instructions.
* Added the following macros: unaligned load/store, seq, double word load/store for O32.
* Improved the parsing of complex memory offset expressions.
* Enabled the integrated assembler by default for Debian mips64el.
* Added a generic scheduler based on the interAptiv CPU.
* Added support for thread local relocations.
* Added recip, rsqrt, evp, dvp, synci instructions in IAS.
* Optimized the generation of constants from some cases.

**The following issues have been fixed:**

* Thread local debug information is correctly recorded.
* MSA intrinsics are now range checked.
* Fixed an issue with MSA and the no-odd-spreg abi.
* Fixed some corner cases in handling forbidden slots for MIPSR6.
* Fixed an issue with jumps not being converted to relative branches for assembly.
* Fixed the handling of local symbols and jal instruction.
* N32/N64 no longer have their relocation tables sorted as per their ABIs.
* Fixed a crash when half-precision floating point conversion MSA intrinsics are used.
* Fixed several crashes involving FastISel.
* Corrected the corrected definitions for aui/daui/dahi/dati for MIPSR6.

Changes to the X86 Target
-------------------------

**During this release the X86 target has:**

* Added support AMD Ryzen (znver1) CPUs.
* Gained support for using VEX encoding on AVX-512 CPUs to reduce code size when possible.
* Improved AVX-512 codegen.

Changes to the OCaml bindings
-----------------------------

* The attribute API was completely overhauled, following the changes
  to the C API.


External Open Source Projects Using LLVM 4.0.0
==============================================

LDC - the LLVM-based D compiler
-------------------------------

`D <http://dlang.org>`_ is a language with C-like syntax and static typing. It
pragmatically combines efficiency, control, and modeling power, with safety and
programmer productivity. D supports powerful concepts like Compile-Time Function
Execution (CTFE) and Template Meta-Programming, provides an innovative approach
to concurrency and offers many classical paradigms.

`LDC <http://wiki.dlang.org/LDC>`_ uses the frontend from the reference compiler
combined with LLVM as backend to produce efficient native code. LDC targets
x86/x86_64 systems like Linux, OS X, FreeBSD and Windows and also Linux on ARM
and PowerPC (32/64 bit). Ports to other architectures like AArch64 and MIPS64
are underway.

Portable Computing Language (pocl)
----------------------------------

In addition to producing an easily portable open source OpenCL
implementation, another major goal of `pocl <http://pocl.sourceforge.net/>`_
is improving performance portability of OpenCL programs with
compiler optimizations, reducing the need for target-dependent manual
optimizations. An important part of pocl is a set of LLVM passes used to
statically parallelize multiple work-items with the kernel compiler, even in
the presence of work-group barriers. This enables static parallelization of
the fine-grained static concurrency in the work groups in multiple ways.

TTA-based Co-design Environment (TCE)
-------------------------------------

`TCE <http://tce.cs.tut.fi/>`_ is a toolset for designing customized
processors based on the Transport Triggered Architecture (TTA).
The toolset provides a complete co-design flow from C/C++
programs down to synthesizable VHDL/Verilog and parallel program binaries.
Processor customization points include register files, function units,
supported operations, and the interconnection network.

TCE uses Clang and LLVM for C/C++/OpenCL C language support, target independent
optimizations and also for parts of code generation. It generates new
LLVM-based code generators "on the fly" for the designed TTA processors and
loads them in to the compiler backend as runtime libraries to avoid
per-target recompilation of larger parts of the compiler chain.


Additional Information
======================

A wide variety of additional information is available on the `LLVM web page
<http://llvm.org/>`_, in particular in the `documentation
<http://llvm.org/docs/>`_ section.  The web page also contains versions of the
API documentation which is up-to-date with the Subversion version of the source
code.  You can access versions of these documents specific to this release by
going into the ``llvm/docs/`` directory in the LLVM tree.

If you have any questions or comments about LLVM, please feel free to contact
us via the `mailing lists <http://llvm.org/docs/#maillist>`_.
