======================
LLVM 3.6 Release Notes
======================

.. contents::
    :local:

.. warning::
   These are in-progress notes for the upcoming LLVM 3.6 release.  You may
   prefer the `LLVM 3.5 Release Notes <http://llvm.org/releases/3.5.0/docs
   /ReleaseNotes.html>`_.


Introduction
============

This document contains the release notes for the LLVM Compiler Infrastructure,
release 3.6.  Here we describe the status of LLVM, including major improvements
from the previous release, improvements in various subprojects of LLVM, and
some of the current users of the code.  All LLVM releases may be downloaded
from the `LLVM releases web site <http://llvm.org/releases/>`_.

For more information about LLVM, including information about the latest
release, please check out the `main LLVM web site <http://llvm.org/>`_.  If you
have questions or comments, the `LLVM Developer's Mailing List
<http://lists.cs.uiuc.edu/mailman/listinfo/llvmdev>`_ is a good place to send
them.

Note that if you are reading this file from a Subversion checkout or the main
LLVM web page, this document applies to the *next* release, not the current
one.  To see the release notes for a specific release, please see the `releases
page <http://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================

.. NOTE
   For small 1-3 sentence descriptions, just add an entry at the end of
   this list. If your description won't fit comfortably in one bullet
   point (e.g. maybe you would like to give an example of the
   functionality, or simply have a lot to talk about), see the `NOTE` below
   for adding a new subsection.

* Support for AuroraUX has been removed.

* Added support for a `native object file-based bitcode wrapper format
  <BitCodeFormat.html#native-object-file>`_.

* ... next change ...

.. NOTE
   If you would like to document a larger change, then you can add a
   subsection about it right here. You can copy the following boilerplate
   and un-indent it (the indentation causes it to be inside this comment).

   Special New Feature
   -------------------

   Makes programs 10x faster by doing Special New Thing.

Prefix data rework
------------------

The semantics of the ``prefix`` attribute have been changed. Users
that want the previous ``prefix`` semantics should instead use
``prologue``.  To motivate this change, let's examine the primary
usecases that these attributes aim to serve,

  1. Code sanitization metadata (e.g. Clang's undefined behavior
     sanitizer)

  2. Function hot-patching: Enable the user to insert ``nop`` operations
     at the beginning of the function which can later be safely replaced
     with a call to some instrumentation facility.

  3. Language runtime metadata: Allow a compiler to insert data for
     use by the runtime during execution. GHC is one example of a
     compiler that needs this functionality for its
     tables-next-to-code functionality.

Previously ``prefix`` served cases (1) and (2) quite well by allowing the user
to introduce arbitrary data at the entrypoint but before the function
body. Case (3), however, was poorly handled by this approach as it
required that prefix data was valid executable code.

In this release the concept of prefix data has been redefined to be
data which occurs immediately before the function entrypoint (i.e. the
symbol address). Since prefix data now occurs before the function
entrypoint, there is no need for the data to be valid code.

The previous notion of prefix data now goes under the name "prologue
data" to emphasize its duality with the function epilogue.

The intention here is to handle cases (1) and (2) with prologue data and
case (3) with prefix data. See the language reference for further details
on the semantics of these attributes.

This refactoring arose out of discussions_ with Reid Kleckner in
response to a proposal to introduce the notion of symbol offsets to
enable handling of case (3).

.. _discussions: http://lists.cs.uiuc.edu/pipermail/llvmdev/2014-May/073235.html


Metadata is not a Value
-----------------------

Metadata nodes (``!{...}``) and strings (``!"..."``) are no longer values.
They have no use-lists, no type, cannot RAUW, and cannot be function-local.

Bridges between Value and Metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LLVM intrinsics can reference metadata using the ``metadata`` type, and
metadata nodes can reference constant values.

Function-local metadata is limited to direct arguments to LLVM intrinsics.

Metadata is typeless
^^^^^^^^^^^^^^^^^^^^

The following old IR:

.. code-block:: llvm

    @g = global i32 0

    define void @foo(i32 %v) {
    entry:
      call void @llvm.md(metadata !{i32 %v})
      call void @llvm.md(metadata !{i32* @global})
      call void @llvm.md(metadata !0)
      call void @llvm.md(metadata !{metadata !"string"})
      call void @llvm.md(metadata !{metadata !{metadata !1, metadata !"string"}})
      ret void, !bar !1, !baz !2
    }

    declare void @llvm.md(metadata)

    !0 = metadata !{metadata !1, metadata !2, metadata !3, metadata !"some string"}
    !1 = metadata !{metadata !2, null, metadata !"other", i32* @global, i32 7}
    !2 = metadata !{}

should now be written as:

.. code-block:: llvm

    @g = global i32 0

    define void @foo(i32 %v) {
    entry:
      call void @llvm.md(metadata i32 %v) ; The only legal place for function-local
                                          ; metadata.
      call void @llvm.md(metadata i32* @global)
      call void @llvm.md(metadata !0)
      call void @llvm.md(metadata !{!"string"})
      call void @llvm.md(metadata !{!{!1, !"string"}})
      ret void, !bar !1, !baz !2
    }

    declare void @llvm.md(metadata)

    !0 = !{!1, !2, !3, !"some string"}
    !1 = !{!2, null, !"other", i32* @global, i32 7}
    !2 = !{}

Distinct metadata nodes
^^^^^^^^^^^^^^^^^^^^^^^

Metadata nodes can opt-out of uniquing, using the keyword ``distinct``.
Distinct nodes are still owned by the context, but are stored in a side table,
and not uniqued.

In LLVM 3.5, metadata nodes would drop uniquing if an operand changed to
``null`` during optimizations.  This is no longer true.  However, if an operand
change causes a uniquing collision, they become ``distinct``.  Unlike LLVM 3.5,
where serializing to assembly or bitcode would re-unique the nodes, they now
remain ``distinct``.

The following IR:

.. code-block:: llvm

    !named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}

    !0 = !{}
    !1 = !{}
    !2 = distinct !{}
    !3 = distinct !{}
    !4 = !{!0}
    !5 = distinct !{!0}
    !6 = !{!4, !{}, !5}
    !7 = !{!{!0}, !0, !5}
    !8 = distinct !{!{!0}, !0, !5}

is equivalent to the following:

.. code-block:: llvm

    !named = !{!0, !0, !1, !2, !3, !4, !5, !5, !6}

    !0 = !{}
    !1 = distinct !{}
    !2 = distinct !{}
    !3 = !{!0}
    !4 = distinct !{!0}
    !5 = !{!3, !0, !4}
    !6 = distinct !{!3, !0, !4}

Constructing cyclic graphs
^^^^^^^^^^^^^^^^^^^^^^^^^^

During graph construction, if a metadata node transitively references a forward
declaration, the node itself is considered "unresolved" until the forward
declaration resolves.  An unresolved node can RAUW itself to support uniquing.
Nodes automatically resolve once all their operands have resolved.

However, cyclic graphs prevent the nodes from resolving.  An API client that
constructs a cyclic graph must call ``resolveCycles()`` to resolve nodes in the
cycle.

To save self-references from that burden, self-referencing nodes are implicitly
``distinct``.  So the following IR:

.. code-block:: llvm

    !named = !{!0, !1, !2, !3, !4}

    !0 = !{!0}
    !1 = !{!1}
    !2 = !{!2, !1}
    !3 = !{!2, !1}
    !4 = !{!2, !1}

is equivalent to:

.. code-block:: llvm

    !named = !{!0, !1, !2, !3, !3}

    !0 = distinct !{!0}
    !1 = distinct !{!1}
    !2 = distinct !{!2, !1}
    !3 = !{!2, !1}

MDLocation (aka DebugLoc aka DILocation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There's a new first-class metadata construct called ``MDLocation`` (to be
followed in subsequent releases by others).  It's used for the locations
referenced by ``!dbg`` metadata attachments.

For example, if an old ``!dbg`` attachment looked like this:

.. code-block:: llvm

    define i32 @foo(i32 %a, i32 %b) {
    entry:
      %add = add i32 %a, %b, !dbg !0
      ret %add, !dbg !1
    }

    !0 = metadata !{i32 10, i32 3, metadata !2, metadata !1)
    !1 = metadata !{i32 20, i32 7, metadata !3)
    !2 = metadata !{...}
    !3 = metadata !{...}

the new attachment looks like this:

.. code-block:: llvm

    define i32 @foo(i32 %a, i32 %b) {
    entry:
      %add = add i32 %a, %b, !dbg !0
      ret %add, !dbg !1
    }

    !0 = !MDLocation(line: 10, column: 3, scope: !2, inlinedAt: !1)
    !1 = !MDLocation(line: 20, column: 7, scope: !3)
    !2 = !{...}
    !3 = !{...}

The fields are named, can be reordered, and have sane defaults if left out
(although ``scope:`` is required).


Alias syntax change
-----------------------

The syntax for aliases is now closer to what is used for global variables

.. code-block:: llvm

    @a = weak global ...
    @b = weak alias ...

The order of the ``alias`` keyword and the linkage was swapped before.

The old JIT has been removed
----------------------------

All users should transition to MCJIT.


object::Binary doesn't owns the file buffer
-------------------------------------------

It is now just a wrapper, which simplifies using object::Binary with other
users of the underlying file.

IR in object files is now supported
-----------------------------------

Regular object files can contain IR in a section named ``.llvmbc``.


The gold plugin has been rewritten
----------------------------------

It is now implemented directly on top of lib/Linker instead of ``lib/LTO``.
The API of ``lib/LTO`` is sufficiently different from gold's view of the
linking process that some cases could not be conveniently implemented.

The new implementation is also lazier and has a ``save-temps`` option.


Change in the representation of lazy loaded funcs
-------------------------------------------------

Lazy loaded functions are now represented is a way that ``isDeclaration``
returns the correct answer even before reading the body.


The opt option -std-compile-opts was removed
--------------------------------------------

It was effectively an alias of -O3.


Python 2.7 is now required
--------------------------

This was done to simplify compatibility with python 3.

The leak detector has been removed
----------------------------------

In practice tools like asan and valgrind were finding way more bugs than
the old leak detector, so it was removed.


New comdat syntax
-----------------

The syntax of comdats was changed to

.. code-block:: llvm

    $c = comdat any
    @g = global i32 0, comdat($c)
    @c = global i32 0, comdat

The version without the parentheses is a syntatic sugar for a comdat with
the same name as the global.


Diagnotic infrastructure used by lib/Linker and lib/Bitcode
-----------------------------------------------------------

These libraries now use the diagnostic handler to print errors and warnings.
This provides better error messages and simpler error handling.


The PreserveSource linker mode was removed
------------------------------------------

It was fairly broken and was removed.



Changes to the ARM Backend
--------------------------

 During this release ...


Changes to the MIPS Target
--------------------------

During this release the MIPS target has reached a few major milestones. The
compiler has gained support for MIPS-II and MIPS-III; become ABI-compatible
with GCC for big and little endian O32, N32, and N64; and is now able to
compile the Linux kernel for 32-bit targets. Additionally, LLD now supports
microMIPS for the O32 ABI on little endian targets.

ABI
^^^

A large number of bugs have been fixed for big-endian MIPS targets using the
N32 and N64 ABI's as well as a small number of bugs affecting other ABI's.
Please note that some of these bugs will still affect LLVM-IR generated by
LLVM 3.5 since correct code generation depends on appropriate usage of the
``inreg``, ``signext``, and ``zeroext`` attributes on all function arguments
and returns.

There are far too many corrections to provide a complete list but here are a
few notable ones:

* Big-endian N32 and N64 now interlinks successfully with GCC compiled code.
  Previously this didn't work for the majority of cases.

* The registers used to return a structure containing a single 128-bit floating
  point member on the N32/N64 ABI's have been changed from those specified by
  the ABI documentation to match those used by GCC. The documentation specifies
  that ``$f0`` and ``$f2`` should be used but GCC has used ``$f0`` and ``$f1``
  for many years.

* Returning a zero-byte struct no longer causes arguments to be read from the
  wrong registers when using the O32 ABI.

* The exception personality has been changed for 64-bit MIPS targets to
  eliminate warnings about relocations in a read-only section.

* Incorrect usage of odd-numbered single-precision floating point registers
  has been fixed when the fastcc calling convention is used with 64-bit FPU's
  and -mno-odd-spreg.

LLVMLinux
^^^^^^^^^

It is now possible to compile the Linux kernel. This currently requires a small
number of kernel patches. See the `LLVMLinux project
<http://llvm.linuxfoundation.org/index.php/Main_Page>`_ for details.

* Added -mabicalls and -mno-abicalls. The implementation may not be complete
  but works sufficiently well for the Linux kernel.

* Fixed multiple compatibility issues between LLVM's inline assembly support
  and GCC's.

* Added support for a number of directives used by Linux to the Integrated
  Assembler.

Miscellaneous
^^^^^^^^^^^^^

* Attempting to disassemble l[wd]c[23], s[wd]c[23], cache, and pref no longer
  triggers an assertion.

* Added -muclibc and -mglibc to support toolchains that provide both uClibC and
  GLibC.

* __SIZEOF_INT128__ is no longer defined for 64-bit targets since 128-bit
  integers do not work at this time for this target.

* Using $t4-$t7 with the N32 and N64 ABI is deprecated when ``-fintegrated-as``
  is in use and will be removed in LLVM 3.7. These names have never been
  supported by the GNU Assembler for these ABI's.

Changes to the PowerPC Target
-----------------------------

There are numerous improvements to the PowerPC target in this release:

* LLVM now generates the Vector-Scalar eXtension (VSX) instructions from
  version 2.06 of the Power ISA, for both big- and little-endian targets.

* LLVM now has a POWER8 instruction scheduling description.

* Address Sanitizer (ASAN) support is now fully functional.

* Performance of simple atomic accesses has been greatly improved.

* Atomic fences now use light-weight syncs where possible, again providing
  significant performance benefit.

* The PowerPC target now supports PIC levels (-fPIC vs. -fpic).

* PPC32 SVR4 now supports small-model PIC.

* There have been many smaller bug fixes and performance improvements.

Changes to the OCaml bindings
-----------------------------

* The bindings now require OCaml >=4.00.0, ocamlfind,
  ctypes >=0.3.0 <0.4 and OUnit 2 if tests are enabled.

* The bindings can now be built using cmake as well as autoconf.

* LLVM 3.5 has, unfortunately, shipped a broken Llvm_executionengine
  implementation. In LLVM 3.6, the bindings now fully support MCJIT,
  however the interface is reworked from scratch using ctypes
  and is not backwards compatible.

* Llvm_linker.Mode was removed following the changes in LLVM.
  This breaks the interface of Llvm_linker.

* All combinations of ocamlc/ocamlc -custom/ocamlopt and shared/static
  builds of LLVM are now supported.

* Absolute paths are not embedded into the OCaml libraries anymore.
  Either OCaml >=4.02.2 must be used, which includes an rpath-like $ORIGIN
  mechanism, or META file must be updated for out-of-tree installations;
  see r221139.

* As usual, many more functions have been exposed to OCaml.

External Open Source Projects Using LLVM 3.6
============================================

An exciting aspect of LLVM is that it is used as an enabling technology for
a lot of other language and tools projects. This section lists some of the
projects that have already been updated to work with LLVM 3.6.

Portable Computing Language (pocl)
----------------------------------

In addition to producing an easily portable open source OpenCL
implementation, another major goal of `pocl <http://portablecl.org/>`_
is improving performance portability of OpenCL programs with
compiler optimizations, reducing the need for target-dependent manual
optimizations. An important part of pocl is a set of LLVM passes used to
statically parallelize multiple work-items with the kernel compiler, even in
the presence of work-group barriers. This enables static parallelization of
the fine-grained static concurrency in the work groups in multiple ways. 

TTA-based Co-design Environment (TCE)
-------------------------------------

`TCE <http://tce.cs.tut.fi/>`_ is a toolset for designing customized
exposed datapath processors based on the Transport triggered 
architecture (TTA). 

The toolset provides a complete co-design flow from C/C++
programs down to synthesizable VHDL/Verilog and parallel program binaries.
Processor customization points include the register files, function units,
supported operations, and the interconnection network.

TCE uses Clang and LLVM for C/C++/OpenCL C language support, target independent 
optimizations and also for parts of code generation. It generates
new LLVM-based code generators "on the fly" for the designed processors and
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

