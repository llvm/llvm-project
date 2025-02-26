<!-- This document is written in Markdown and uses extra directives provided by
MyST (https://myst-parser.readthedocs.io/en/latest/). -->

LLVM {{env.config.release}} Release Notes
=========================================

```{contents}
```

````{only} PreRelease
```{warning} These are in-progress notes for the upcoming LLVM {{env.config.release}}
             release. Release notes for previous releases can be found on
             [the Download Page](https://releases.llvm.org/download.html).
```
````

Introduction
============

This document contains the release notes for the LLVM Compiler Infrastructure,
release {{env.config.release}}.  Here we describe the status of LLVM, including
major improvements from the previous release, improvements in various subprojects
of LLVM, and some of the current users of the code.  All LLVM releases may be
downloaded from the [LLVM releases web site](https://llvm.org/releases/).

For more information about LLVM, including information about the latest
release, please check out the [main LLVM web site](https://llvm.org/).  If you
have questions or comments, the [Discourse forums](https://discourse.llvm.org)
is a good place to ask them.

Note that if you are reading this file from a Git checkout or the main
LLVM web page, this document applies to the *next* release, not the current
one.  To see the release notes for a specific release, please see the
[releases page](https://llvm.org/releases/).

Non-comprehensive list of changes in this release
=================================================

<!-- For small 1-3 sentence descriptions, just add an entry at the end of
this list. If your description won't fit comfortably in one bullet
point (e.g. maybe you would like to give an example of the
functionality, or simply have a lot to talk about), see the comment below
for adding a new subsection. -->

* ...

<!-- If you would like to document a larger change, then you can add a
subsection about it right here. You can copy the following boilerplate:

Special New Feature
-------------------

Makes programs 10x faster by doing Special New Thing.
-->

Changes to the LLVM IR
----------------------

* The `nocapture` attribute has been replaced by `captures(none)`.
* The constant expression variants of the following instructions have been
  removed:

  * `mul`

Changes to LLVM infrastructure
------------------------------

* Removed support for target intrinsics being defined in the target directories
  themselves (i.e., the `TargetIntrinsicInfo` class).

Changes to building LLVM
------------------------

Changes to TableGen
-------------------

Changes to Interprocedural Optimizations
----------------------------------------

Changes to the AArch64 Backend
------------------------------

Changes to the AMDGPU Backend
-----------------------------

Changes to the ARM Backend
--------------------------

Changes to the AVR Backend
--------------------------

Changes to the DirectX Backend
------------------------------

Changes to the Hexagon Backend
------------------------------

Changes to the LoongArch Backend
--------------------------------

Changes to the MIPS Backend
---------------------------

Changes to the PowerPC Backend
------------------------------

Changes to the RISC-V Backend
-----------------------------

* Adds experimental assembler support for the Qualcomm uC 'Xqcilia` (Large Immediate Arithmetic)
  extension.

Changes to the WebAssembly Backend
----------------------------------

Changes to the Windows Target
-----------------------------

Changes to the X86 Backend
--------------------------

Changes to the OCaml bindings
-----------------------------

Changes to the Python bindings
------------------------------

Changes to the C API
--------------------

* The following functions for creating constant expressions have been removed,
  because the underlying constant expressions are no longer supported. Instead,
  an instruction should be created using the `LLVMBuildXYZ` APIs, which will
  constant fold the operands if possible and create an instruction otherwise:

  * `LLVMConstMul`
  * `LLVMConstNUWMul`
  * `LLVMConstNSWMul`

Changes to the CodeGen infrastructure
-------------------------------------

Changes to the Metadata Info
---------------------------------

Changes to the Debug Info
---------------------------------

Changes to the LLVM tools
---------------------------------

* llvm-objcopy now supports the `--update-section` flag for intermediate Mach-O object files.

Changes to LLDB
---------------------------------

* When building LLDB with Python support, the minimum version of Python is now
  3.8.
* LLDB now supports hardware watchpoints for AArch64 Windows targets. Windows
  does not provide API to query the number of supported hardware watchpoints.
  Therefore current implementation allows only 1 watchpoint, as tested with
  Windows 11 on the Microsoft SQ2 and Snapdragon Elite X platforms.
* LLDB now steps through C++ thunks. This fixes an issue where previously, it
  wouldn't step into multiple inheritance virtual functions.

### Changes to lldb-dap

* Breakpoints can now be set for specific columns within a line.

Changes to BOLT
---------------------------------

Changes to Sanitizers
---------------------

Other Changes
-------------

External Open Source Projects Using LLVM {{env.config.release}}
===============================================================

* A project...

Additional Information
======================

A wide variety of additional information is available on the
[LLVM web page](https://llvm.org/), in particular in the
[documentation](https://llvm.org/docs/) section.  The web page also contains
versions of the API documentation which is up-to-date with the Git version of
the source code.  You can access versions of these documents specific to this
release by going into the `llvm/docs/` directory in the LLVM tree.

If you have any questions or comments about LLVM, please feel free to contact
us via the [Discourse forums](https://discourse.llvm.org).
