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

* Added a new IRNormalizer pass which aims to transform LLVM modules into
  a normal form by reordering and renaming instructions while preserving the
  same semantics. The normalizer makes it easier to spot semantic differences
  when diffing two modules which have undergone different passes.

* ...

<!-- If you would like to document a larger change, then you can add a
subsection about it right here. You can copy the following boilerplate:

Special New Feature
-------------------

Makes programs 10x faster by doing Special New Thing.
-->

Changes to the LLVM IR
----------------------

* Types are no longer allowed to be recursive.

* The `x86_mmx` IR type has been removed. It will be translated to
  the standard vector type `<1 x i64>` in bitcode upgrade.
* Renamed `llvm.experimental.stepvector` intrinsic to `llvm.stepvector`.

* Added `usub_cond` and `usub_sat` operations to `atomicrmw`.

* Introduced `noalias.addrspace` metadata.

* Remove the following intrinsics which can be replaced with a `bitcast`:

  * `llvm.nvvm.bitcast.f2i`
  * `llvm.nvvm.bitcast.i2f`
  * `llvm.nvvm.bitcast.d2ll`
  * `llvm.nvvm.bitcast.ll2d`

* Remove the following intrinsics which can be replaced with a funnel-shift:

  * `llvm.nvvm.rotate.b32`
  * `llvm.nvvm.rotate.right.b64`
  * `llvm.nvvm.rotate.b64`

* Remove the following intrinsics which can be replaced with an
  `addrspacecast`:

  * `llvm.nvvm.ptr.gen.to.global`
  * `llvm.nvvm.ptr.gen.to.shared`
  * `llvm.nvvm.ptr.gen.to.constant`
  * `llvm.nvvm.ptr.gen.to.local`
  * `llvm.nvvm.ptr.global.to.gen`
  * `llvm.nvvm.ptr.shared.to.gen`
  * `llvm.nvvm.ptr.constant.to.gen`
  * `llvm.nvvm.ptr.local.to.gen`

* Remove the following intrinsics which can be relaced with a load from
  addrspace(1) with an !invariant.load metadata

  * `llvm.nvvm.ldg.global.i`
  * `llvm.nvvm.ldg.global.f`
  * `llvm.nvvm.ldg.global.p`

* Operand bundle values can now be metadata strings.

* Fast math flags are now permitted on `fptrunc` and `fpext`.

Changes to LLVM infrastructure
------------------------------

Changes to building LLVM
------------------------

* Raised the minimum MSVC version to Visual Studio 2019 16.8.

Changes to TableGen
-------------------

Changes to Interprocedural Optimizations
----------------------------------------

Changes to the AArch64 Backend
------------------------------

* `.balign N, 0`, `.p2align N, 0`, `.align N, 0` in code sections will now fill
  the required alignment space with a sequence of `0x0` bytes (the requested
  fill value) rather than NOPs.

* Assembler/disassembler support has been added for Armv9.6-A (2024)
  architecture extensions.

* Added support for the FUJITSU-MONAKA CPU.

Changes to the AMDGPU Backend
-----------------------------

* Removed `llvm.amdgcn.flat.atomic.fadd` and
  `llvm.amdgcn.global.atomic.fadd` intrinsics. Users should use the
  {ref}`atomicrmw <i_atomicrmw>` instruction with `fadd` and
  addrspace(0) or addrspace(1) instead.

Changes to the ARM Backend
--------------------------

* `.balign N, 0`, `.p2align N, 0`, `.align N, 0` in code sections will now fill
  the required alignment space with a sequence of `0x0` bytes (the requested
  fill value) rather than NOPs.

* The default behavior for frame pointers in leaf functions has been updated.
  When the `-fno-omit-frame-pointer` option is specified, `FPKeepKindStr` is
  set to `-mframe-pointer=all`, meaning the frame pointer (FP) is now retained
  in leaf functions by default. To eliminate the frame pointer in leaf functions,
  you must explicitly use the `-momit-leaf-frame-pointer` option.

* When using the `MOVT` or `MOVW` instructions, the Assembler will now check to
  ensure that any addend that is used is within a 16-bit signed value range. If the
  addend falls outside of this range, the LLVM backend will emit an error like so
  `Relocation Not In Range`.

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

* The Linux `ppc64` LLC default cpu is updated from `ppc` to `ppc64`.
* The AIX LLC default cpu is updated from `generic` to `pwr7`.

Changes to the RISC-V Backend
-----------------------------

* `.balign N, 0`, `.p2align N, 0`, `.align N, 0` in code sections will now fill
  the required alignment space with a sequence of `0x0` bytes (the requested
  fill value) rather than NOPs.
* Added Syntacore SCR4 and SCR5 CPUs: `-mcpu=syntacore-scr4/5-rv32/64`
* `-mcpu=sifive-p470` was added.
* Added Hazard3 CPU as taped out for RP2350: `-mcpu=rp2350-hazard3` (32-bit
  only).
* Fixed length vector support using RVV instructions now requires VLEN>=64. This
  means Zve32x and Zve32f will also require Zvl64b. The prior support was
  largely untested.
* The `Zvbc32e` and `Zvkgs` extensions are now supported experimentally.
* Added `Smctr`, `Ssctr` and `Svvptc` extensions.
* `-mcpu=syntacore-scr7` was added.
* `-mcpu=tt-ascalon-d8` was added.
* `-mcpu=mips-p8700` was added.
* `-mcpu=sifive-p550` was added.
* The `Zacas` extension is no longer marked as experimental.
* Added Smdbltrp, Ssdbltrp extensions to -march.
* The `Smmpm`, `Smnpm`, `Ssnpm`, `Supm`, and `Sspm` pointer masking extensions
  are no longer marked as experimental.
* The `Sha` extension is now supported.
* The RVA23U64, RVA23S64, RVB23U64, and RVB23S64 profiles are no longer marked
  as experimental.
* `.insn <length>, <raw encoding>` can be used to assemble 48- and 64-bit
  instructions from raw integer values.
* `.insn [<length>,] <raw encoding>` now accepts absolute expressions for both
  expressions, so that they can be computed from constants and absolute symbols.
* The following new inline assembly constraints and modifiers are accepted:
  * `cr` constraint meaning an RVC-encoding compatible GPR (`x8`-`x15`)
  * `cf` constraint meaning an RVC-encoding compatible FPR (`f8`-`f15`)
  * `R` constraint meaning an even-odd GPR pair (prints as the even register,
    but both registers in the pair are considered live).
  * `N` modifer meaning print the register encoding (0-31) rather than the name.
* `f` and `cf` inline assembly constraints, when using F-/D-/H-in-X extensions,
  will use the relevant GPR rather than FPR. This makes inline assembly portable
  between e.g. F and Zfinx code.
* Adds experimental assembler support for the Qualcomm uC 'Xqcicsr` (CSR)
  extension.
* Adds experimental assembler support for the Qualcomm uC 'Xqcisls` (Scaled Load Store)
  extension.
* Adds experimental assembler support for the Qualcomm uC 'Xqcia` (Arithmetic)
  extension.
* Adds experimental assembler support for the Qualcomm uC 'Xqciac` (Load-Store Address Calculation)
  extension.
* Adds experimental assembler support for the Qualcomm uC 'Xqcics` (Conditonal Select)
  extension.
* Adds experimental assembler support for the Qualcomm uC 'Xqcilsm` (Load Store Multiple)
  extension.
* Adds experimental assembler support for the Qualcomm uC 'Xqcicli` (Conditional Load Immediate)
  extension.
* Adds experimental assembler support for the Qualcomm uC 'Xqcicm` (Conditonal Move)
  extension.
* Added ``Sdext`` and ``Sdtrig`` extensions.

Changes to the WebAssembly Backend
----------------------------------

* The default target CPU, "generic", now enables the `-mnontrapping-fptoint`
  and `-mbulk-memory` flags, which correspond to the [Bulk Memory Operations]
  and [Non-trapping float-to-int Conversions] language features, which are
  [widely implemented in engines].

* A new Lime1 target CPU is added, `-mcpu=lime1`. This CPU follows the
  definition of the Lime1 CPU [here], and enables `-mmultivalue`,
  `-mmutable-globals`, `-mcall-indirect-overlong`, `-msign-ext`,
  `-mbulk-memory-opt`, `-mnontrapping-fptoint`, and `-mextended-const`.

[Bulk Memory Operations]: https://github.com/WebAssembly/bulk-memory-operations/blob/master/proposals/bulk-memory-operations/Overview.md
[Non-trapping float-to-int Conversions]: https://github.com/WebAssembly/spec/blob/master/proposals/nontrapping-float-to-int-conversion/Overview.md
[widely implemented in engines]: https://webassembly.org/features/
[here]: https://github.com/WebAssembly/tool-conventions/blob/main/Lime.md#lime1

Changes to the Windows Target
-----------------------------

Changes to the X86 Backend
--------------------------

* `.balign N, 0x90`, `.p2align N, 0x90`, and `.align N, 0x90` in code sections
  now fill the required alignment space with repeating `0x90` bytes, rather than
  using optimised NOP filling. Optimised NOP filling fills the space with NOP
  instructions of various widths, not just those that use the `0x90` byte
  encoding. To use optimised NOP filling in a code section, leave off the
  "fillval" argument, i.e. `.balign N`, `.p2align N` or `.align N` respectively.

* Due to the removal of the `x86_mmx` IR type, functions with
  `x86_mmx` arguments or return values will use a different,
  incompatible, calling convention ABI. Such functions are not
  generally seen in the wild (Clang never generates them!), so this is
  not expected to result in real-world compatibility problems.

* Support ISA of `AVX10.2-256` and `AVX10.2-512`.

* Supported instructions of `MOVRS AND AVX10.2`

* Supported ISA of `SM4(EVEX)`.

* Supported ISA of `MSR_IMM`.

* Supported ``-mcpu=diamondrapids``

* Supported emitting relocation types for x86-64 target:
  * `R_X86_64_CODE_4_GOTPCRELX`
  * `R_X86_64_CODE_4_GOTTPOFF`
  * `R_X86_64_CODE_4_GOTPC32_TLSDESC`
  * `R_X86_64_CODE_6_GOTTPOFF`


Changes to the OCaml bindings
-----------------------------

Changes to the Python bindings
------------------------------

Changes to the C API
--------------------

* The following symbols are deleted due to the removal of the `x86_mmx` IR type:

  * `LLVMX86_MMXTypeKind`
  * `LLVMX86MMXTypeInContext`
  * `LLVMX86MMXType`

 * The following functions are added to further support non-null-terminated strings:

  * `LLVMGetNamedFunctionWithLength`
  * `LLVMGetNamedGlobalWithLength`

* The following functions are added to access the `LLVMContextRef` associated
   with `LLVMValueRef` and `LLVMBuilderRef` objects:

  * `LLVMGetValueContext`
  * `LLVMGetBuilderContext`

* The new pass manager can now be invoked with a custom alias analysis pipeline, using
  the `LLVMPassBuilderOptionsSetAAPipeline` function.

* It is now also possible to run the new pass manager on a single function, by calling
  `LLVMRunPassesOnFunction` instead of `LLVMRunPasses`.

* Support for creating instructions with custom synchronization scopes has been added:

  * `LLVMGetSyncScopeID` to map a synchronization scope name to an ID.
  * `LLVMBuildFenceSyncScope`, `LLVMBuildAtomicRMWSyncScope` and
    `LLVMBuildAtomicCmpXchgSyncScope` versions of the existing builder functions
    with an additional synchronization scope ID parameter.
  * `LLVMGetAtomicSyncScopeID` and `LLVMSetAtomicSyncScopeID` to get and set the
    synchronization scope of any atomic instruction.
  * `LLVMIsAtomic` to check if an instruction is atomic, for use with the above functions.
    Because of backwards compatibility, `LLVMIsAtomicSingleThread` and
    `LLVMSetAtomicSingleThread` continue to work with any instruction type.

* The `LLVMSetPersonalityFn` and `LLVMSetInitializer` APIs now support clearing the
  personality function and initializer respectively by passing a null pointer.

* The following functions are added to allow iterating over debug records attached to
  instructions:

  * `LLVMGetFirstDbgRecord`
  * `LLVMGetLastDbgRecord`
  * `LLVMGetNextDbgRecord`
  * `LLVMGetPreviousDbgRecord`

* Added `LLVMAtomicRMWBinOpUSubCond` and `LLVMAtomicRMWBinOpUSubSat` to `LLVMAtomicRMWBinOp` enum for AtomicRMW instructions.

Changes to the CodeGen infrastructure
-------------------------------------

Changes to the Metadata Info
---------------------------------

Changes to the Debug Info
---------------------------------

Changes to the LLVM tools
---------------------------------

* llvm-objcopy now supports the following options for Mach-O:
  `--globalize-symbol`, `--globalize-symbols`,
  `--keep-global-symbol`, `--keep-global-symbols`,
  `--localize-symbol`, `--localize-symbols`,
  `--skip-symbol`, `--skip-symbols`.

Changes to LLDB
---------------------------------

* LLDB now supports inline diagnostics for the expression evaluator and command line parser.

  Old:
  ```
  (lldb) p a+b
  error: <user expression 0>:1:1: use of undeclared identifier 'a'
      1 | a+b
        | ^
  error: <user expression 0>:1:3: use of undeclared identifier 'b'
      1 | a+b
        |   ^
  ```

  New:

  ```
  (lldb) p a+b
           ˄ ˄
           │ ╰─ error: use of undeclared identifier 'b'
           ╰─ error: use of undeclared identifier 'a'
  ```

* LLDB can now read the `fpmr` register from AArch64 Linux processes and core
  files.

* Program stdout/stderr redirection will now open the file with O_TRUNC flag, make sure to truncate the file if path already exists.
  * eg. `settings set target.output-path/target.error-path <path/to/file>`

* A new setting `target.launch-working-dir` can be used to set a persistent cwd that is used by default by `process launch` and `run`.

* LLDB now parses shared libraries in parallel, resulting in an average 2x speedup when attaching (only available on Darwin platforms) and launching (available on all platforms).

* On the command line, LLDB now limits tab completions to your terminal width to avoid wrapping.

  Old:
  ```
  Available completions:
          _regexp-attach    -- Attach to process by ID or name.
          _regexp-break     -- Set a breakpoint using one of several shorthand
  formats.
          _regexp-bt        -- Show backtrace of the current thread's call sta
  ck. Any numeric argument displays at most that many frames. The argument 'al
  l' displays all threads. Use 'settings set frame-format' to customize the pr
  inting of individual frames and 'settings set thread-format' to customize th
  e thread header. Frame recognizers may filter thelist. Use 'thread backtrace
  -u (--unfiltered)' to see them all.
          _regexp-display   -- Evaluate an expression at every stop (see 'help
  target stop-hook'.)

  ```

  New:
  ```
  Available completions:
          _regexp-attach    -- Attach to process by ID or name.
          _regexp-break     -- Set a breakpoint using one of several shorth...
          _regexp-bt        -- Show backtrace of the current thread's call ...
          _regexp-display   -- Evaluate an expression at every stop (see 'h...
  ```

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
