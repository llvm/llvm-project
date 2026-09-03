<!-- This document is written in Markdown and uses extra directives provided by
MyST (https://myst-parser.readthedocs.io/en/latest/). -->

<!-- If you want to modify sections/contents permanently, you should modify both
ReleaseNotes.md and ReleaseNotesTemplate.txt. -->

# LLVM {{env.config.release}} Release Notes


::::{only} PreRelease
:::{warning} These are in-progress notes for the upcoming LLVM {{env.config.release}}
             release. Release notes for previous releases can be found on
             [the Download Page](https://releases.llvm.org/download.html).
:::
::::

## Introduction

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

## Non-comprehensive list of changes in this release

<!-- For small 1-3 sentence descriptions, just add an entry at the end of
this list. If your description won't fit comfortably in one bullet
point (e.g. maybe you would like to give an example of the
functionality, or simply have a lot to talk about), see the comment below
for adding a new subsection. -->

* ...

<!-- If you would like to document a larger change, then you can add a
subsection about it right here. You can copy the following boilerplate:

### Special New Feature

Makes programs 10x faster by doing Special New Thing.
-->

### Changes to the LLVM IR

* Added `llvm.vector.reduce.fmaximumnum` and `llvm.vector.reduce.fminimumnum`
  intrinsics, the reduction variants of `llvm.maximumnum` and
  `llvm.minimumnum`. 
* Added `nofreeobj` attribute for attributes and returns, which forbids
  freeing the underlying object (as opposed to only frees through that specific
  pointer). Renamed `!nofree` metadata to `!nofreeobj`, as it has the same
  semantics.
* The following VP intrinsics have been removed:
  * `llvm.vp.select.*`
  * `llvm.vp.add.*`
  * `llvm.vp.sub.*`
  * `llvm.vp.mul.*`
  * `llvm.vp.ashr.*`
  * `llvm.vp.lshr.*`
  * `llvm.vp.shl.*`
  * `llvm.vp.or.*`
  * `llvm.vp.and.*`
  * `llvm.vp.xor.*`
  * `llvm.vp.abs.*`
  * `llvm.vp.smax.*`
  * `llvm.vp.smin.*`
  * `llvm.vp.umax.*`
  * `llvm.vp.umin.*`
  * `llvm.vp.copysign.*`
  * `llvm.vp.minnum.*`
  * `llvm.vp.maxnum.*`
  * `llvm.vp.minimum.*`
  * `llvm.vp.maximum.*`
  * `llvm.vp.fadd.*`
  * `llvm.vp.fsub.*`
  * `llvm.vp.fmul.*`
  * `llvm.vp.fdiv.*`
  * `llvm.vp.frem.*`
  * `llvm.vp.fneg.*`
  * `llvm.vp.fabs.*`
  * `llvm.vp.sqrt.*`
  * `llvm.vp.fma.*`
  * `llvm.vp.fmuladd.*`
  * `llvm.vp.trunc.*`
  * `llvm.vp.zext.*`
  * `llvm.vp.sext.*`
  * `llvm.vp.fptrunc.*`
  * `llvm.vp.fpext.*`
  * `llvm.vp.fptoui.*`
  * `llvm.vp.fptosi.*`
  * `llvm.vp.uitofp.*`
  * `llvm.vp.sitofp.*`
  * `llvm.vp.ptrtoint.*`
  * `llvm.vp.inttoptr.*`
  * `llvm.vp.fcmp.*`
  * `llvm.vp.icmp.*`
  * `llvm.vp.ceil.*`
  * `llvm.vp.floor.*`
  * `llvm.vp.rint.*`
  * `llvm.vp.nearbyint.*`
  * `llvm.vp.round.*`
  * `llvm.vp.roundeven.*`
  * `llvm.vp.roundtozero.*`
  * `llvm.vp.lrint.*`
  * `llvm.vp.llrint.*`
  * `llvm.vp.bitreverse.*`
  * `llvm.vp.bswap.*`
  * `llvm.vp.ctpop.*`
  * `llvm.vp.ctlz.*`
  * `llvm.vp.cttz.*`
  * `llvm.vp.sadd.sat.*`
  * `llvm.vp.uadd.sat.*`
  * `llvm.vp.ssub.sat.*`
  * `llvm.vp.usub.sat.*`
  * `llvm.vp.fshl.*`
  * `llvm.vp.fshr.*`
  * `llvm.vp.is.fpclass.*`

  These intrinsics previously only set masked-off lanes to poison, and will be
  automatically upgraded to their non-VP equivalent.  On RISC-V the VL optimizer
  should automatically infer `vl` in most cases from a store or reduction
  instruction, so passing around an explicit EVL operand shouldn't be required.
  If needed a "root" EVL can be synthesized with `llvm.vp.merge`, e.g:

  ```llvm
  %x = add <vscale x 2 x i32> %y, %z
  %res = call <vscale x 2 x i32> @llvm.vp.merge(<vscale x 2 x i32> %x, <vscale x 2 x i32> poison, <vscale x 2 x i1> splat (i1 true), i32 %evl)
  ```

  The `llvm.vp.merge` will be folded away but the `%evl` will be propagated to
  the add instruction.

### Changes to LLVM infrastructure

* Removed `TargetOptions::FloatABIType`. The soft float ABI should be
  controlled by setting the `"float-abi"` module flag.

### Changes to building LLVM

* The DirectX backend is now an official target and has moved from
  `LLVM_ALL_EXPERIMENTAL_TARGETS` to `LLVM_ALL_TARGETS`. It is now built by
  default and no longer requires `LLVM_EXPERIMENTAL_TARGETS_TO_BUILD`.

### Changes to TableGen

* `!cond` operator short-circuits at the first `true` condition.  Subsequent
  `condition : value` pairs, along with their corresponding side effects,
  are left unresolved.

### Changes to Interprocedural Optimizations

- Interprocedural passes no longer rewrite the signature of functions marked
  `optnone`, so their argument list, return type, and calling convention are
  preserved. Interprocedural analysis and transformation of such functions is
  otherwise unaffected.

- The IR Outliner has been removed, due to lack of a maintainer and the presence
  of correctness issues.

### Changes to Vectorizers

### Changes to the AArch64 Backend

### Changes to the AMDGPU Backend

* Replaced `xnack` and `sramecc` target features with `amdgpu.xnack`
  and `amdgpu.sramecc` module flags.
* `llvm.amdgcn.make.buffer.rsrc` now accepts any integer width for its
  `numRecords` argument to account for targets that use 32-bit and 45-bit
  `numRecords` widths more accurately. If an integer of the incorrect width
  is used, it will be zero-extended or truncated as needed.

* These intrinsics have been removed in favour of `llvm.amdgcn.ballot`:
  * `llvm.amdgcn.icmp`
  * `llvm.amdgcn.fcmp`

### Changes to the ARM Backend

* Using the hard-float procedure call standard without floating-point registers
  is now an error. Previously this would fall back to the soft-float PCS while
  still emitting the hard-float ABI attribute tag.

### Changes to the AVR Backend

### Changes to the DirectX Backend

* The DirectX backend has been promoted from experimental to an official,
  fully supported LLVM target.

### Changes to the Hexagon Backend

### Changes to the LoongArch Backend

### Changes to the MIPS Backend

### Changes to the PowerPC Backend

### Changes to the RISC-V Backend

* Added experimental MC support for the `Smcsps` and `Sscsps`
  conditional stack pointer swap extensions.
* Adds experimental assembler/CodeGen support for the `Zilx` (Indexed Integer
  Load) extension.
* Added experimental MC support for the `Smijt` and `Ssijt` interrupt jump
  table extensions and the `Smehv` and `Ssehv` synchronous exception hardware
  vectoring extensions.
* Added experimental MC support for the `Smip` and `Ssip` interrupt handler
  push/pop extensions.
* Bump Svukte extension to 1.0.
* Remove experimental from Zicfiss.
* Added support for `Sspmp`, `Sspmpen` and `Smpmpdeleg` extensions.

### Changes to the WebAssembly Backend

* Added support for emitting common symbols (.comm) using the WASM_SYMBOL_BINDING_COMMON
  flag (see https://github.com/WebAssembly/tool-conventions/pull/267)
* Added `@llvm.wasm.memory.copy` and `@llvm.wasm.memory.fill` intrinsics for
  the WebAssembly `memory.copy` and `memory.fill` instructions.

### Changes to the Windows Target

### Changes to the X86 Backend

### Changes to the OCaml bindings

### Changes to the Python bindings

### Changes to the C API

### Changes to the CodeGen infrastructure

### Changes to the Metadata Info

### Changes to the Debug Info

### Changes to the LLVM tools

* llvm-mca no longer defaults -mcpu to "native"

### Changes to LLDB

* `platform.plugin.wasm.runtime-args` now precede the port argument on the Wasm
  runtime's command line instead of following it. A runtime that dispatches on a
  leading subcommand can therefore name that subcommand through this setting,
  rather than needing a wrapper script.

#### SBAPI

* A [bug](https://github.com/llvm/llvm-project/issues/211787) involving SBValues
  representing a register set was fixed. The methods `GetIndexOfChildWithName`
  and `GetChildMemberWithName` were incorrectly looking up values in all
  register sets. This meant that `GetIndexOfChildWithName` could return an index
  greater than the size of the set, and that `GetChildMemberWithName` could
  return values that were actually in a different set. Both methods are now fixed
  so that they are limited to the registers within the register set. Scripts
  using these methods may have to be updated as a result.

#### Windows

* Python 3.11 or later is now required for building LLDB 24 on Windows.
* For better performance, LLDB now turns off the Windows debug heap by default when debugging.
  If you need the debug heap enabled, set `platform.plugin.windows.disable-debug-heap` to `false`.

### Changes to BOLT

### Changes to Sanitizers

### Other Changes

* `cas::ObjectStore::getMemoryBuffer()` was documented as returning a buffer
  whose lifetime is independent of the CAS, but the buffer it returns may alias
  storage the CAS owns and so cannot outlive it. The documentation now matches
  the behavior, and the new `getStandaloneMemoryBuffer()` provides a buffer that
  does stay valid after the `ObjectStore` is destroyed.

## External Open Source Projects Using LLVM {{env.config.release}}

## Additional Information

A wide variety of additional information is available on the
[LLVM web page](https://llvm.org/), in particular in the
[documentation](https://llvm.org/docs/) section.  The web page also contains
versions of the API documentation which is up-to-date with the Git version of
the source code.  You can access versions of these documents specific to this
release by going into the `llvm/docs/` directory in the LLVM tree.

If you have any questions or comments about LLVM, please feel free to contact
us via the [Discourse forums](https://discourse.llvm.org).
