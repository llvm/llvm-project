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

* It is no longer permitted to inspect the uses of ConstantData. Use
  count APIs will behave as if they have no uses (i.e. use_empty() is
  always true).

* The `nocapture` attribute has been replaced by `captures(none)`.
* The constant expression variants of the following instructions have been
  removed:

  * `mul`

* Updated semantics of `llvm.type.checked.load.relative` to match that of
  `llvm.load.relative`.
* Inline asm calls no longer accept ``label`` arguments. Use ``callbr`` instead.

* Updated semantics of the `callbr` instruction to clarify that its
  'indirect labels' are not expected to be reached by indirect (as in
  register-controlled) branch instructions, and therefore are not
  guaranteed to start with a `bti` or `endbr64` instruction, where
  those exist.

Changes to LLVM infrastructure
------------------------------

* Removed support for target intrinsics being defined in the target directories
  themselves (i.e., the `TargetIntrinsicInfo` class).
* Fix Microsoft demangling of string literals to be stricter
  (#GH129970))
* Added the support for ``fmaximum`` and ``fminimum`` in ``atomicrmw`` instruction. The
  comparison is expected to match the behavior of ``llvm.maximum.*`` and
  ``llvm.minimum.*`` respectively.

Changes to building LLVM
------------------------

Changes to TableGen
-------------------

Changes to Interprocedural Optimizations
----------------------------------------

Changes to the AArch64 Backend
------------------------------

* Added the `execute-only` target feature, which indicates that the generated
  program code doesn't contain any inline data, and there are no data accesses
  to code sections. On ELF targets this property is indicated by the
  `SHF_AARCH64_PURECODE` section flag.
  ([#125687](https://github.com/llvm/llvm-project/pull/125687),
  [#132196](https://github.com/llvm/llvm-project/pull/132196),
  [#133084](https://github.com/llvm/llvm-project/pull/133084))

Changes to the AMDGPU Backend
-----------------------------

* Enabled the
  [FWD_PROGRESS bit](https://llvm.org/docs/AMDGPUUsage.html#code-object-v3-kernel-descriptor)
  for all GFX ISAs greater or equal to 10, for the AMDHSA OS.

* Bump the default `.amdhsa_code_object_version` to 6. ROCm 6.3 is required to run any program compiled with COV6.

* Add a new `amdgcn.load.to.lds` intrinsic that wraps the existing global.load.lds
intrinsic and has the same semantics. This intrinsic allows using buffer fat pointers
(`ptr addrspace(7)`) as arguments, allowing loads to LDS from these pointers to be
represented in the IR without needing to use buffer resource intrinsics directly.
This intrinsic is exposed to Clang as `__builtin_amdgcn_load_to_lds`, though
buffer fat pointers are not yet enabled in Clang. Migration to this intrinsic is
optional, and there are no plans to deprecate `amdgcn.global.load.lds`.

Changes to the ARM Backend
--------------------------

Changes to the AVR Backend
--------------------------

Changes to the DirectX Backend
------------------------------

Changes to the Hexagon Backend
------------------------------

* The default Hexagon architecture version in ELF object files produced by
  the tools such as llvm-mc is changed to v68. This version will be set if
  the user does not provide the CPU version in the command line.

Changes to the LoongArch Backend
--------------------------------

* Changing the default code model from `small` to `medium` for 64-bit.
* Added inline asm support for the `q` constraint.
* Added the `32s` target feature for LA32S ISA extensions.
* Added codegen support for atomic-ops (`cmpxchg`, `max`, `min`, `umax`, `umin`) on LA32.
* Added codegen support for the ILP32D calling convention.
* Added several codegen and vectorization optimizations.

Changes to the MIPS Backend
---------------------------

* `-mcpu=i6400` and `-mcpu=i6500` were added.

Changes to the PowerPC Backend
------------------------------

Changes to the RISC-V Backend
-----------------------------

* Adds experimental assembler support for the Qualcomm uC 'Xqcilb` (Long Branch)
  extension.
* Adds experimental assembler support for the Qualcomm uC 'Xqcili` (Load Large Immediate)
  extension.
* Adds experimental assembler support for the Qualcomm uC 'Xqcilia` (Large Immediate Arithmetic)
  extension.
* Adds experimental assembler support for the Qualcomm uC 'Xqcibm` (Bit Manipulation)
  extension.
* Adds experimental assembler support for the Qualcomm uC 'Xqcibi` (Branch Immediate)
  extension.
* Adds experimental assembler and code generation support for the Qualcomm
  'Xqccmp' extension, which is a frame-pointer convention compatible version of
  Zcmp.
* Added non-quadratic ``log-vrgather`` cost model for ``vrgather.vv`` instruction
* Adds experimental assembler support for the Qualcomm uC 'Xqcisim` (Simulation Hint)
  extension.
* Adds experimental assembler support for the Qualcomm uC 'Xqcisync` (Sync Delay)
  extension.
* Adds experimental assembler support for the Qualcomm uC 'Xqciio` (External Input Output)
  extension.
* Adds assembler support for the 'Zilsd` (Load/Store Pair Instructions)
  extension.
* Adds assembler support for the 'Zclsd` (Compressed Load/Store Pair Instructions)
  extension.
* Adds experimental assembler support for Zvqdotq.
* Adds Support for Qualcomm's `qci-nest` and `qci-nonest` interrupt types, which
  use instructions from `Xqciint` to save and restore some GPRs during interrupt
  handlers.
* When the experimental extension `Xqcili` is enabled, `qc.e.li` and `qc.li` may
  now be used to materialize immediates.
* Adds assembler support for ``.option exact``, which disables automatic compression,
  and branch and linker relaxation. This can be disabled with ``.option noexact``,
  which is also the default.
* `-mcpu=xiangshan-kunminghu` was added.
* `-mcpu=andes-n45` and `-mcpu=andes-nx45` were added.
* `-mcpu=andes-a45` and `-mcpu=andes-ax45` were added.
* Adds support for the 'Ziccamoc` (Main Memory Supports Atomics in Zacas) extension, which was introduced as an optional extension of the RISC-V Profiles specification.
* Adds experimental assembler support for SiFive CLIC CSRs, under the names
  `Zsfmclic` for the M-mode registers and `Zsfsclic` for the S-mode registers.
* Adds Support for SiFive CLIC interrupt attributes, which automate writing CLIC
  interrupt handlers without using inline assembly.
* Adds assembler support for the Andes `XAndesperf` (Andes Performance extension).
* `-mcpu=sifive-p870` was added.
* Adds assembler support for the Andes `XAndesvpackfph` (Andes Vector Packed FP16 extension).
* Adds assembler support for the Andes `XAndesvdot` (Andes Vector Dot Product extension).
* Adds assembler support for the standard `Q` (Quad-Precision Floating Point) 
  extension.
* Adds experimental assembler support for the SiFive Xsfmm* Attached Matrix
  Extensions.
* `-mcpu=andes-a25` and `-mcpu=andes-ax25` were added.
* The `Shlcofideleg` extension was added.
* `-mcpu=sifive-x390` was added.
* `-mtune=andes-45-series` was added.
* Adds assembler support for the Andes `XAndesvbfhcvt` (Andes Vector BFLOAT16 Conversion extension).
* `-mcpu=andes-ax45mpv` was added.
* Removed -mattr=+no-rvc-hints that could be used to disable parsing and generation of RVC hints.
* Adds assembler support for the Andes `XAndesvsintload` (Andes Vector INT4 Load extension).
* Adds assembler support for the Andes `XAndesbfhcvt` (Andes Scalar BFLOAT16 Conversion extension).

Changes to the WebAssembly Backend
----------------------------------

Changes to the Windows Target
-----------------------------

* `fp128` is now passed indirectly, meaning it uses the same calling convention
  as `i128`.

Changes to the X86 Backend
--------------------------

* `fp128` will now use `*f128` libcalls on 32-bit GNU targets as well.
* On x86-32, `fp128` and `i128` are now passed with the expected 16-byte stack
  alignment.

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

* Added `LLVMConstDataArray` and `LLVMGetRawDataValues` to allow creating and
  reading `ConstantDataArray` values without needing extra `LLVMValueRef`s for
  individual elements.

* Added ``LLVMDIBuilderCreateEnumeratorOfArbitraryPrecision`` for creating
  debugging metadata of enumerators larger than 64 bits.

* Added ``LLVMGetICmpSameSign`` and ``LLVMSetICmpSameSign`` for the `samesign`
  flag on `icmp` instructions.

Changes to the CodeGen infrastructure
-------------------------------------

Changes to the Metadata Info
---------------------------------

Changes to the Debug Info
---------------------------------

Changes to the LLVM tools
---------------------------------

* llvm-objcopy now supports the `--update-section` flag for intermediate Mach-O object files.
* llvm-strip now supports continuing to process files on encountering an error.
* In llvm-objcopy/llvm-strip's ELF port, `--discard-locals` and `--discard-all` now allow and preserve symbols referenced by relocations.
  ([#47468](https://github.com/llvm/llvm-project/issues/47468))
* llvm-addr2line now supports a `+` prefix when specifying an address.
* Support for `SHT_LLVM_BB_ADDR_MAP` versions 0 and 1 has been dropped.
* llvm-objdump now supports the `--debug-inlined-funcs` flag, which prints the
  locations of inlined functions alongside disassembly. The
  `--debug-vars-indent` flag has also been renamed to `--debug-indent`.

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
* A statusline was added to command-line LLDB to show progress events and
  information about the current state of the debugger at the bottom of the
  terminal. This is on by default and can be configured using the
  `show-statusline` and `statusline-format` settings. It is not currently
  supported on Windows.
* The `min-gdbserver-port` and `max-gdbserver-port` options have been removed
  from `lldb-server`'s platform mode. Since the changes to `lldb-server`'s port
  handling in LLDB 20, these options have had no effect.
* LLDB now supports `process continue --reverse` when used with debug servers
  supporting reverse execution, such as [rr](https://rr-project.org).
  When using reverse execution, `process continue --forward` returns to the
  forward execution.
* LLDB now supports RISC-V 32-bit ELF core files.
* LLDB now supports siginfo descriptions for Linux user-space signals. User space
  signals will now have descriptions describing the method and sender.
  ```
    stop reason = SIGSEGV: sent by tkill system call (sender pid=649752, uid=2667987)
  ```
* ELF Cores can now have their siginfo structures inspected using `thread siginfo`.
* LLDB now uses
  [DIL](https://discourse.llvm.org/t/rfc-data-inspection-language/69893) as the
  default implementation for 'frame variable'. This should not change the
  behavior of 'frame variable' at all, at this time. To revert to using the
  old implementation use: `settings set target.experimental.use-DIL false`.
* Disassembly of unknown instructions now produces `<unknown>` instead of
  nothing at all
* Changed the format of opcode bytes to match llvm-objdump when disassembling
  RISC-V code with `disassemble`'s `--byte` option.


### Changes to lldb-dap

* Breakpoints can now be set for specific columns within a line.
* Function return value is now displayed on step-out.

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
