<!-- This document is written in Markdown and uses extra directives provided by
MyST (https://myst-parser.readthedocs.io/en/latest/). -->

<!-- If you want to modify sections/contents permanently, you should modify both
ReleaseNotes.md and ReleaseNotesTemplate.txt. -->

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

* The `ptrtoaddr` instruction was introduced. This instruction returns the
  address component of a pointer type variable but unlike `ptrtoint` does not
  capture provenance ([#125687](https://github.com/llvm/llvm-project/pull/125687)).
* The alignment argument of the `@llvm.masked.load`, `@llvm.masked.store`,
  `@llvm.masked.gather` and `@llvm.masked.scatter` intrinsics has been removed.
  Instead, the `align` attribute should be placed on the pointer (or vector of
  pointers) argument.
* A `load atomic` may now be used with vector types on x86.
* Added `@llvm.reloc.none` intrinsic to emit null relocations to symbols. This
  emits an undefined symbol reference without adding any dedicated code or data to
  to bear the relocation.
* Added `modular-format` attribute to dynamically pull in aspects of libc
  format string function implementations from statically-linked libc's based on
  the requirements of each call. Currently only `float` is supported; this can
  keep floating point support out of printf if it can be proven unused.
* Case values are no longer operands of `SwitchInst`.

Changes to LLVM infrastructure
------------------------------

Changes to building LLVM
------------------------

Changes to TableGen
-------------------

Changes to Interprocedural Optimizations
----------------------------------------

* Added `-enable-machine-outliner={optimistic-pgo,conservative-pgo}` to read
  profile data to guide the machine outliner
  ([#154437](https://github.com/llvm/llvm-project/pull/154437)).

Changes to Vectorizers
----------------------------------------

* Added initial support for copyable elements in SLP, which models copyable
  elements as add <element>, 0, i.e. uses identity constants for missing lanes.
* SLP vectorizer supports initial recognition of FMA/FMAD pattern

Changes to the AArch64 Backend
------------------------------

* Assembler/disassembler support has been added for Armv9.7-A (2025)
  architecture extensions.

* Assembler/disassembler support has been added for 'Virtual Tagging
  Extension (vMTE)' and 'Permission Overlay Extension version 2 (POE2)'
  Future Architecture Technologies extensions.

* `FEAT_TME` support has been removed, as it has been withdrawn from
   all future versions of the A-profile architecture.

* Added support for C1-Nano, C1-Pro, C1-Premium, and C1-Ultra CPUs.

Changes to the AMDGPU Backend
-----------------------------

* Removed `llvm.amdgcn.atomic.cond.sub.u32` and
  `llvm.amdgcn.atomic.csub.u32` intrinsics. Users should use the
  `atomicrmw` instruction with `usub_cond` and `usub_sat` instead.

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

* `half` now uses a soft float ABI, which works correctly in more cases.

Changes to the RISC-V Backend
-----------------------------

* The loop vectorizer now performs tail folding by default on RISC-V, which
  removes the need for a scalar epilogue loop. To restore the previous behaviour
  use `-prefer-predicate-over-epilogue=scalar-epilogue`.
* `llvm-objdump` now has basic support for switching between disassembling code
  and data using mapping symbols such as `$x` and `$d`. Switching architectures
  using `$x` with an architecture string suffix is not yet supported.
* Ssctr and Smctr extensions are no longer experimental.
* Add support for Zvfbfa (Additional BF16 vector compute support)
* Adds experimental support for the 'Zibi` (Branch with Immediate) extension.
* Add support for Zvfofp8min (OFP8 conversion extension)
* Adds assembler support for the Andes `XAndesvsinth` (Andes Vector Small Int Handling Extension).
* DWARF fission is now compatible with linker relaxations, allowing `-gsplit-dwarf` and `-mrelax`
  to be used together when building for the RISC-V platform.
* The Xqci Qualcomm uC Vendor Extension is no longger marked as experimental.

Changes to the WebAssembly Backend
----------------------------------

* `half` now uses a soft float lowering, which resolves various precision and
  bitcast issues.

Changes to the Windows Target
-----------------------------

* `-fpseudo-probe-for-profiling` is now supported for COFF.

Changes to the X86 Backend
--------------------------

* `-mcpu=wildcatlake` is now supported.
* `-mcpu=novalake` is now supported.

Changes to the OCaml bindings
-----------------------------

* The IR reader bindings renamed `parse_ir` to
  `parse_ir_bitcode_or_assembly` to clarify that the parser accepts both
  textual IR and bitcode. This rename is intentional to force existing code to
  update because the ownership semantics changed: the function no longer takes
  ownership of the input memory buffer.

Changes to the Python bindings
------------------------------

Changes to the C API
--------------------

* Add `LLVMGetOrInsertFunction` to get or insert a function, replacing the combination of `LLVMGetNamedFunction` and `LLVMAddFunction`.
* Allow `LLVMGetVolatile` to work with any kind of Instruction.
* Add `LLVMConstFPFromBits` to get a constant floating-point value from an array of 64 bit values.
* Add `LLVMParseIRInContext2`, which is equivalent to `LLVMParseIRInContext`
  but does not take ownership of the input `LLVMMemoryBufferRef`. This matches
  the underlying C++ API and avoids ownership surprises in language bindings
  and examples.
* Functions working on the global context have been deprecated. Use the
  functions that work on a specific context instead.

  * `LLVMGetGlobalContext` -> use `LLVMContextCreate` context instead
  * `LLVMInt1Type` -> `LLVMInt1TypeInContext`
  * `LLVMInt8Type` -> `LLVMInt8TypeInContext`
  * `LLVMInt16Type` -> `LLVMInt16TypeInContext`
  * `LLVMInt32Type` -> `LLVMInt32TypeInContext`
  * `LLVMInt64Type` -> `LLVMInt64TypeInContext`
  * `LLVMInt128Type` -> `LLVMInt128TypeInContext`
  * `LLVMIntType` -> `LLVMIntTypeInContext`
  * `LLVMHalfType` -> `LLVMHalfTypeInContext`
  * `LLVMBFloatType` -> `LLVMBFloatTypeInContext`
  * `LLVMFloatType` -> `LLVMFloatTypeInContext`
  * `LLVMDoubleType` -> `LLVMDoubleTypeInContext`
  * `LLVMX86FP80Type` -> `LLVMX86FP80TypeInContext`
  * `LLVMFP128Type` -> `LLVMFP128TypeInContext`
  * `LLVMPPCFP128Type` -> `LLVMPPCFP128TypeInContext`
  * `LLVMStructType` -> `LLVMStructTypeInContext`
  * `LLVMVoidType` -> `LLVMVoidTypeInContext`
  * `LLVMLabelType` -> `LLVMLabelTypeInContext`
  * `LLVMX86AMXType` -> `LLVMX86AMXTypeInContext`
  * `LLVMConstString` -> `LLVMConstStringInContext2`
  * `LLVMConstStruct` -> `LLVMConstStructInContext`
  * `LLVMMDString` -> `LLVMMDStringInContext2`
  * `LLVMMDNode` -> `LLVMMDNodeInContext2`
  * `LLVMAppendBasicBlock` -> `LLVMAppendBasicBlockInContext`
  * `LLVMInsertBasicBlock` -> `LLVMInsertBasicBlockInContext`
  * `LLVMCreateBuilder` -> `LLVMCreateBuilderInContext`
  * `LLVMIntPtrType` -> `LLVMIntPtrTypeInContext`
  * `LLVMIntPtrTypeForAS` -> `LLVMIntPtrTypeForASInContext`
  * `LLVMParseBitcode` -> `LLVMParseBitcodeInContext2`
  * `LLVMParseBitcode2` -> `LLVMParseBitcodeInContext2`
  * `LLVMGetBitcodeModule` -> `LLVMGetBitcodeModuleInContext2`
  * `LLVMGetBitcodeModule2` -> `LLVMGetBitcodeModuleInContext2`
* Add `LLVMGetSwitchCaseValue` and `LLVMSetSwitchCaseValue` to get and set switch case values; switch case values are no longer operands of the instruction.

Changes to the CodeGen infrastructure
-------------------------------------

Changes to the Metadata Info
---------------------------------

Changes to the Debug Info
---------------------------------

Changes to the LLVM tools
---------------------------------

* `llvm-profgen` now supports decoding pseudo probe for COFF binaries.

* `llvm-readelf` now dumps all hex format values in lower-case mode.
* Some code paths for supporting Python 2.7 in `llvm-lit` have been removed.
* Support for `%T` in lit has been removed.
* Add `--save-stats` option to `llc` to save LLVM statistics to a file. Compatible with the Clang option.
* Add `--save-stats` option to `opt` to save LLVM statistics to a file. Compatible with the Clang option.

* `llvm-config` gained a new flag `--quote-paths` which quotes and escapes paths
  emitted on stdout, to account for spaces or other special characters in path.
  (`#97305 <https://github.com/llvm/llvm-project/pull/97305>`_).

* `llvm-objdump` now supports using `--mcpu=help` and `--mattr=help` with the `--triple` option
  without requiring an input file or the `-d` (disassemble) flag.

Changes to LLDB
---------------------------------

* LLDB can now set breakpoints, show backtraces, and display variables when
  debugging Wasm with supported runtimes (WAMR and V8).
* LLDB now has a Wasm platform, which can be configured to run WebAssembly
  binaries directly under a Wasm runtime. Configurable through the
  platform.plugin.wasm settings.
* LLDB no longer stops processes by default when receiving SIGWINCH signals
  (window resize events) on Linux. This is the default on other Unix platforms.
  You can re-enable it using `process handle --notify=true --stop=true SIGWINCH`.
* The `show-progress` setting, which became a NOOP with the introduction of the
  statusline, now defaults to off and controls using OSC escape codes to show a
  native progress bar in supporting terminals like Ghostty and ConEmu.
* The default PDB reader on Windows was changed from DIA to native, which uses
  LLVM's PDB and CodeView support. You can switch back to the DIA reader with
  `settings set plugin.symbol-file.pdb.reader dia`. Note that support for the
  DIA reader will be removed in a future version of LLDB.
* A `--verbose` option was added to the `version` command. When `--verbose` is used,
  LLDB's build configuration is included in the command's output. This includes
  all the supported targets, along with the presence of (or lack of) optional
  features like XML parsing.

Changes to BOLT
---------------------------------

Changes to Sanitizers
---------------------

* Support running TypeSanitizer with UndefinedBehaviourSanitizer.
* TypeSanitizer no longer inlines all instrumentation by default. Added the
  `-f[no-]sanitize-type-outline-instrumentation` flags to give users control
  over this behaviour.

Other Changes
-------------

* Introduces the `AllocToken` pass, an instrumentation pass providing tokens to
  memory allocators enabling various heap organization strategies, such as heap
  partitioning.

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
