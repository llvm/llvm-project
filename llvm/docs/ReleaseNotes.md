<!-- This document is written in Markdown and uses extra directives provided by
MyST (https://myst-parser.readthedocs.io/en/latest/). -->

<!-- If you want to modify sections/contents permanently, you should modify both
ReleaseNotes.md and ReleaseNotesTemplate.txt. -->

# LLVM {{env.config.release}} Release Notes

```{contents}
```

````{only} PreRelease
```{warning} These are in-progress notes for the upcoming LLVM {{env.config.release}}
             release. Release notes for previous releases can be found on
             [the Download Page](https://releases.llvm.org/download.html).
```
````

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

* Removed `llvm.convert.to.fp16` and `llvm.convert.from.fp16`
  intrinsics. These are equivalent to `fptrunc` and `fpext` with half
  with a bitcast.

* "denormal-fp-math" and "denormal-fp-math-f32" string attributes were
  migrated to first-class denormal_fpenv attribute.

* The `"nooutline"` attribute is now writen as `nooutline`. Existing IR and
  bitcode will be automatically updated.

* `ConstantPointerNull` can now represent fixed and scalable vector splats of
  null pointers. Such constants may print as `splat (ptr null)` instead of
  `zeroinitializer`.

* LLVM IR floating-point literals have greatly changed:

  * The old hexadecimal bitwise representation is deprecated and will be removed
    in the next revision. It is replaced with a unified `f0x` prefix.

  * Hexadecimal literals akin to C99's syntax are supported.

  * Special values for infinities and NaNs, including NaN payloads, are added.

* The standard textual output for floating-point literals is changed to take
  advantage of the new floating-point literals formats.

* The resume/destroy functions emitted for the switch-resume ABI (C++20
  coroutines) now use `CallingConv::C` instead of `CallingConv::Fast`. This
  stabilizes the coroutine ABI across LLVM versions and aligns it with other
  vendors. The change is observationally identical on targets where `fastcc`
  and `ccc` agree for `void(ptr)` (x86_64, AArch64, RISC-V, ...) but is an ABI
  break on i686, MIPS O32, PowerPC64 ELFv1, and Lanai.

* Assume bundles now only accept attributes that are actually handled.
  Specifically, they are ``align``, ``cold``, ``dereferenceable``,
  ``dereferenceable_or_null``, ``nonnull``, ``noundef`` and
  ``separate_storage``.

* Fast math flags are now permitted on `uitofp` and `sitofp`.

* The ``modular-format`` attribute now supports the ``fixed`` aspect for C
  ISO 18037 fixed-point ``printf`` specifiers.

* Added `noipa` attribute which disables interprocedural analyses that inspect
  the definition of the function. This attribute does *not* control inlining or
  outlining. Add the `noinline` and `nooutline` attributes as well in cases
  where inlining and outlining should additionally be disabled.

* Added support for ``callgraph`` metadata. The `!callgraph` metadata
  associates a function definition with its type identifier and is used for call
  graph section generation. See the [callgraph Metadata](https://llvm.org/docs/LangRef.html#callgraph-metadata)
  section of the LangRef for details.

* The `dereferenceable` and `dereferenceable_or_null` attributes (and
  corresponding metadata) now apply only at the point of definition, instead of
  for the execution of the function (for arguments) or forever (for returns).

* `alwaysinline` no longer bypasses inlining compatibility checks based on
  target features. Inlining will only be performed if it is safe to do so.

* Module-level inline assembly now accepts optional `target_features` and
  `target_cpu` properties. This resolves errors during LTO on some
  architectures.

### Changes to LLVM infrastructure

* Removed ``Constant::isZeroValue``. It was functionally identical to
  ``Constant::isNullValue`` for all types except floating-point negative
  zero. All callers should use ``isNullValue`` instead. ``isZeroValue``
  will be reintroduced in the future with bitwise-all-zeros semantics
  to support non-zero null pointers.

* Added support for specifying the null pointer bit representation per
  address space in `DataLayout`. Pointer specifications (`p`) accept new
  flags: `z` (null is all-zeros) and `o` (null is all-ones). Address
  spaces without an explicit flag default to all-zeros. See the
  `DataLayout` section of the
  [LangRef](https://llvm.org/docs/LangRef.html#data-layout) for details.

* Removed TypePromoteFloat legalization from SelectionDAG

* Removed `bugpoint`. Usage has been replaced by `llvm-reduce` and
  `llvm/utils/reduce_pipeline.py`.

* The ``Br`` opcode was split into two opcodes separating unconditional
  (``UncondBr``) and conditional (``CondBr``) branches.

* ``BranchInst`` was deprecated in favor of ``UncondBrInst`` and ``CondBrInst``.

* The operand order of ``CondBr`` instructions was adjusted to match the
  successor order. This can cause subtle breakage when using ``getOperand`` or
  ``setOperand`` to access successors.

* The ``llvm::sys::fs`` link creation API has been refactored:

  * ``create_link`` now tries to create a symbolic link first, falling back to a
    hard link if that fails (previously it created a symlink on Unix and a hard
    link on Windows).
  * Added ``create_symlink``, which always creates a symbolic link. On windows
    this may fail if symlink permissions are not available.
  * Added ``readlink``, which reads the target of a symbolic link.

* Bitcode libraries can now implement compiler-managed library functions
  (libcalls) without causing incorrect API manipulation or undefined references
  ([#177046](https://github.com/llvm/llvm-project/pull/125687)). Note that
  there are still issues with invalid compiler reasoning about some functions
  in bitcode, e.g. `malloc`. Not yet supported on MachO or when using
  distributed ThinLTO.

* ``ConstantFP`` now supports vector types and is the canonical form returned by
  ``ConstantVector::getSplat(C)`` when ``C`` is a scalar ``ConstantFP``.

* ``DenseMap``, ``DenseSet``, ``StringMap``, and ``StringSet`` ``erase`` now
  invalidates all iterators and references into the container, not just the
  iterator for the erased element. Use the new ``remove_if`` member to erase
  matching elements in a single pass instead of erasing while iterating.

* ``TargetRegisterInfo::getMinimalPhysRegClass`` and related APIs have been
  refactored and no longer take a type. This API is also now precomputed in
  TableGen to improve compile-time.

* ``APInt::sqrt`` (square root rounded to nearest integer) has been replaced
  with ``APInt::sqrtFloor`` (floor of square root).

### Changes to building LLVM

### Changes to TableGen

* Outer let statements use ``ID{n-m}`` instead of ``ID<n-m>`` to be consistent
  with inner let statements.

### Changes to Interprocedural Optimizations

### Changes to Vectorizers

### Changes to the AArch64 Backend

* The `sysp`, `mrrs`, and `msrr` instructions are now accepted without
  requiring the `+d128` feature gating.
* Added a new internal option `-aarch64-emit-debug-tls-location` to allow the
  emission of `DW_AT_location` for thread-local variables. This is currently
  disabled by default to maintain compatibility with Binutils and LLVM older
  toolchains that do not define the `R_AARCH64_TLS_DTPREL64` static relocation
  type for TLS offsets.
* A bug was fixed that caused LLVM IR inline assembly clobbers of the x29 and
  x30 registers to be ignored when they were written using their xN names
  instead of the ABI names FP and LR. Note that LLVM IR produced by Clang
  always uses the ABI names, but other frontends may not.
  ([#167783](https://github.com/llvm/llvm-project/pull/167783))

### Changes to the AMDGPU Backend

* Initial support for gfx1310

* The `"amdgpu-num-sgpr"` and `"amdgpu-num-vgpr"` IR function attributes
  (generated by the Clang `amdgpu_num_sgpr` and `amdgpu_num_vgpr` attributes)
  are now deprecated. Use `"amdgpu-waves-per-eu"` instead. The backend still
  honors the attributes; Clang emits a `-Wdeprecated-declarations` warning when
  the source attributes are used.
* The `relaxed-buffer-oob-mode` subtarget feature has been replaced by two
  module flags, `amdgpu.buffer.oob.mode` and `amdgpu.tbuffer.oob.mode`, which
  control out-of-bounds semantics (see the AMDGPU User Guide). Frontends that
  previously relied on the subtarget feature to enable misaligned buffer merging
  must now set the corresponding module flag to `1` (relaxed). An absent flag is
  treated as strict by the backend.

* Changed triple naming system from "amdgcn" to "amdgpu" plus a subarch suffix.

### Changes to the ARM Backend

* The `r14` register can now be used as an alias for the link register `lr`
  in inline assembly. Clang always canonicalizes the name to `lr`, but other
  frontends may not.
* `subs pc, lr, #imm` can now be predicated in Thumb2.

### Changes to the AVR Backend

### Changes to the DirectX Backend

### Changes to the Hexagon Backend

### Changes to the LoongArch Backend

* DWARF fission is now compatible with linker relaxations, allowing `-gsplit-dwarf` and `-mrelax`
  to be used together when building for the LoongArch platform.

### Changes to the MIPS Backend

### Changes to the NVPTX Backend

* The default SM version has been changed from `sm_30` to `sm_75`. `sm_75` is
  the oldest GPU variant compatible with the widest range of recent major CUDA
  Toolkit versions (11/12/13).

### Changes to the PowerPC Backend

### Changes to the RISC-V Backend

* `llvm-objdump` now has support for `--symbolize-operands` with RISC-V.
* `-mcpu=spacemit-x100` was added.
* Change P extension version to match the 0.21 draft specification.
* Mnemonics for MOP/HINT-based instructions (`lpad`, `pause`, `ntl.*`, `c.ntl.*`,
  `sspush`, `sspopchk`, `ssrdp`, `c.sspush`, `c.sspopchk`) are now always
  available in the assembler and disassembler without requiring their respective
  extensions.
* Adds experimental assembler support for the 'Zvabd` (RISC-V Integer Vector
  Absolute Difference) extension.
* Adds CodeGen support for the 'Zvabd` extension.
* `-mcpu=spacemit-a100` was added.
* Adds assembler support for the `XSMTVDotII` (SpacemiT Vector Extension for Matrix 2.0) extension.
* The opt-in `-riscv-enable-p-ext-simd-codegen` flag has been removed. P extension SIMD code generation is now enabled automatically if the P extension is supported.
* `-mcpu=xt-c910v2` and `-mcpu=xt-c920v2` were added.
* Adds experimental assembler support for the 'Zvzip` (RISC-V Vector
  Reordering Structured Data) extension.
* `-mcpu=sifive-x160` and `-mcpu=sifive-x180` were added.
* Support for the experimental `XRivosVisni` vendor extension has been removed.
* Support for the experimental `XRivosVizip` vendor extension has been removed.
* Adds experimental assembler support for the 'Zvvmm` (RISC-V Integer Matrix Multiply-Accumulate) extension.
* Adds experimental assembler support for the 'Zvvfmm` (RISC-V Floating-Point Matrix Multiply-Accumulate) extension.
* Adds experimental assembler support for the 'Zvvmtls` and 'Zvvmttls` (RISC-V
  Matrix Tile Load/Store) extensions.
* Adds support for 'Ziccid' (Instruction/Data Coherence and Consistency) extension.
* Adds experimental assembler support for the `Xqccmt` (Qualcomm 16-bit Table Jump) vendor extension.
* `-mcpu=sifive-870` has been renamed `-mcpu=sifive-p870-d`.
* Adds experimental assembler support for batched dot-product extensions(Zvqwbdota8i, Zvqwbdota16i, Zvfwbdota16bf, Zvfqwbdota8f and Zvfbdota32f).
* Adds experimental assembler support for dot-product extensions(Zvqwdota8i, Zvqwdota16i, Zvfwdota16bf and Zvfqwdota8f).
* `-mtune=generic` now uses the scheduling model from SpacemiT X60 instead of an empty scheduling model.
* The Xqcilo pseudos now emit sequences that can be relaxed.

### Changes to the WebAssembly Backend

* WebAssembly reference types are now represented in LLVM IR as the target
  extension types `target("wasm.externref")` and `target("wasm.funcref")`,
  rather than as pointers in address spaces 10 and 20 (`ptr addrspace(10)` /
  `ptr addrspace(20)`).
* As a consequence of the representation change, reference types are no longer
  treated as vectorizable pointers. This fixes a crash in the SLP vectorizer,
  which previously would attempt to gather `externref`/`funcref` values into a
  vector and then crash.

### Changes to the Windows Target

* The `.seh_startchained` and `.seh_endchained` assembly instructions have been removed and replaced
  with a new `.seh_splitchained` instruction.

### Changes to the X86 Backend

* `.att_syntax` directive is now emitted for assembly files when AT&T syntax is
  in use. This matches the behaviour of Intel syntax and aids with
  compatibility when changing the default Clang syntax to the Intel syntax.

* EGPR (R16-R31) now requires V3 unwind info on Windows x64. Using EGPR
  without V3 unwind produces a fatal error.
* Implemented Win64 APX ABI callee-saved registers: R30 and R31 are now
  treated as non-volatile in the Win64 calling convention when APX is
  available, per the Microsoft x64 calling convention specification.

* Functions using setjmp with Win64 APX ABI now reserve R30/R31 from
  register allocation, as the unwinder cannot restore APX extended
  registers across longjmp. A warning is emitted for large functions
  where this reservation may impact performance.

* Added ``.seh_push2regs`` assembly directive for explicitly encoding a
  two-register push in Windows x64 V3 unwind info. The directive takes two
  register operands: ``.seh_push2regs %r12, %r13``.

* The `fp128` type is now returned via sret instead of XMM0 for some calling
  conventions to match GCC. The C and Win64 calling conventions now use
  sret, other calling conventions do not need to be compatible with
  GCC and still return via XMM0.

### Changes to the OCaml bindings

### Changes to the Python bindings

### Changes to the C API

* Replaced opcode ``LLVMBr`` with ``LLVMUncondBr`` and ``LLVMCondBr``.

* The operand order of ``CondBr`` instructions was adjusted to match the
  successor order. This can cause subtle breakage when using ``LLVMGetOperand``
  or ``LLVMSetOperand`` to access successors.

### Changes to the CodeGen infrastructure

* Renamed ISD::CTLZ_ZERO_UNDEF to ISD::CTLZ_ZERO_POISON opcode to make it clear that
  a zero input results in poison.

* Renamed ISD::CTTZ_ZERO_UNDEF to ISD::CTTZ_ZERO_POISON opcode to make it clear that
  a zero input results in poison.

* LLVM intrinsics can now declare required target features using the
  ``TargetFeatures`` TableGen field. SelectionDAG and GlobalISel use this
  metadata to reject unsupported target intrinsics generically, so targets do
  not need custom lowering checks for each annotated intrinsic.

### Changes to the GlobalISel infrastructure

* Renamed G_CTLZ_ZERO_UNDEF to G_CTLZ_ZERO_POISON opcode to make it clear that
  a zero input results in poison.

* Renamed G_CTTZ_ZERO_UNDEF to G_CTTZ_ZERO_POISON opcode to make it clear that
  a zero input results in poison.

* GlobalISel's IRTranslator now supports the LLVM IR byte type (`bN`). Byte
  values are translated as the equi-sized integer scalar (`sN`), `ConstantByte`
  is materialised via `G_CONSTANT`, and `bitcast` between a byte type and a
  pointer is lowered to `G_INTTOPTR` / `G_PTRTOINT` rather than the
  verifier-illegal `G_BITCAST`.

### Changes to the Metadata Info

### Changes to the Debug Info

### Changes to the LLVM tools

* llvm-ml now supports the `/unwindv3` flag to enable V3 unwind information
  format for x64 exception handling.
* llvm-ml now supports the `@UnwindVersion` built-in symbol, which returns the
  current unwind version (1 by default, 3 when `/unwindv3` is specified).
* llvm-ml now supports the `.push2reg`, `.pop2reg`, `.beginepilog`, and
  `.endepilog` MASM directives for V3 unwind information.
* llvm-ml now supports the `.popreg`, `.freestack`, `.restorereg`,
  `.restorexmm128`, and `.unsetframe` MASM epilog directives. These are the
  epilog counterparts of `.pushreg`, `.allocstack`, `.savereg`,
  `.savexmm128`, and `.setframe` respectively, and are valid only inside
  `.beginepilog`/`.endepilog` blocks and only for V3 unwind information.
* llvm-ml now supports `.pushframe code` syntax (without the `@` prefix)
  for interrupt handlers with error codes.
* llvm-ml now diagnoses:
  - Prolog directives (including `.allocstack`) used outside of prologs (after `.endprolog`).
  - Epilog directives used outside of epilogs (outside of `.beginepilog` + `.endepilog` blocks).
  - Beginning an epilog (`.beginepilog`) inside another epilog.

* `llvm-profgen` now supports ETM trace decoding using the OpenCSD library for Cortex-M targets. OpenCSD version 1.5.4 or higher is required.

* `llvm-objcopy` no longer corrupts the symbol table when `--update-section` is called for ELF files.
* `llvm-objcopy` now reports an error when `--compress-sections` requests unavailable zlib or zstd support.
  The diagnostic is emitted while parsing the option, matching `--compress-debug-sections`.
  Such commands may now fail even if the input file contains no sections that would be compressed.
* `FileCheck` option `-check-prefix` now accepts a comma-separated list of
  prefixes, making it an alias of the existing `-check-prefixes` option.
* Add `-mtune` option to `llc`.
* Add `-mtune` option to `opt`.
* Fixed `llvm-ar` to correctly handle the `N` count modifier on Windows for archive members whose names differ only
  in case (e.g. `FOO.OBJ` and `foo.obj`). Previously, `-N 2` would fail with "not found" even when two matching members existed.
* `llvm-readobj` and `llvm-readelf` now support the `--call-graph-section` option to dump the contents of the experimental [call graph section](CallGraphSection.md).

### Changes to LLDB

* A new ``webinspector-wasm`` platform was added to list and attach to WebAssembly processes in Safari.
* The default for `load-script-from-symbol-file` was changed from `warn` to `trusted`. This means that scripts from
  code signed dSYM bundles are now loaded automatically, while untrusted bundles continue to produce a warning.
* Pressing enter after `frame variable` repeats the command with an incremented `--depth` option, allowing quick
  expansion of nested data.
* Breakpoint commands now accept `.` to refer to the location(s) at which the current thread is stopped. For
  example, `breakpoint disable .` disables the just-hit breakpoint location. Another usage is to automate a
  command to run at the current location: `breakpoint command add -o 'p my_var' .`.
* The `apropos` command now:
  * Highlights matching keywords in its output when color is enabled.
  * Searches the components of settings paths. For example `apropos qemu-user` will now
    show `platform.plugin.qemu-user` as one of the results.
* Reading global and static variables on WebAssembly targets now works correctly. Previously their
  values could not be read because data sections were mapped to the wrong address space.
* A new `diagnostics report` command (aliased `bugreport`) assembles a diagnostics bundle and files
  a pre-filled GitHub issue, pointing at the bundle to attach. The GitHub reporter is built by
  default and can be disabled with the `LLDB_ENABLE_GITHUB_BUG_REPORTER=OFF` CMake option.

#### Deprecated APIs

* ``SBTarget::GetDataByteSize()``, ``SBTarget::GetCodeByteSize()``, and ``SBSection::GetTargetByteSize()``
  have been deprecated. They always return `1`, as before.

#### FreeBSD

##### Userspace Debugging

* Support for MIPS64 has been removed.
* The minimum assumed FreeBSD version is now 14. The effect of which is that watchpoints are
  assumed to be supported.

##### Kernel Debugging

* The plugin that analyzes FreeBSD kernel core dump and live core has been renamed from `freebsd-kernel` to
 `freebsd-kernel-core`. Remote kernel debugging is still handled by the `gdb-remote` plugin.
* Support for libfbsdvmcore has been removed. As a result, FreeBSD kernel dump debugging is now only
  available on FreeBSD hosts. Live kernel debugging through the GDB remote protocol is still available
  from any platform.
* Support for ARM, PPC64le, and RISCV64 has been added.
* The crashed thread is now automatically selected on start.
* Threads are listed in incrmental order by pid then by tid.
* Unread kernel messages saved in `msgbufp` are now printed when LLDB starts.
* Writing to the core is now supported. For safety reasons, this feature is off by default. To enable it,
  `plugin.process.freebsd-kernel-core.read-only` must be set to `false`. This setting is available when
  using `/dev/mem` or a kernel dump. However, since `kvm_write()` does not support writing to kernel dumps,
  writes to a kernel dump will still fail when the setting is false.
* Added a command `process plugin refresh-threads`, enabling on-demand thread-list reconstruction from `/dev/mem`
  so users can resync live kernel thread state without restarting LLDB. Note that this has no impact on full dump
  and minidump files.

#### Linux

* On Arm Linux, the `tpidruro` register can now be read. Writing to this register is not supported.
* Thread local variables are now supported on Arm and RISC-V Linux if the program being debugged is using glibc.
* LLDB now supports AArch64 Linux systems that only have SME (as opposed to
  SVE and SME). See the AArch64 Linux [documentation](https://lldb.llvm.org/use/aarch64-linux.html#sme-only-systems)
  for more details.

  Prior to this version of LLDB, there was a bug that caused LLDB to crash on
  startup on these systems ([#138717](https://github.com/llvm/llvm-project/issues/138717)).
  This affected LLDB versions from 18 up to and including 22. 17 and below are not affected.
  If you are using such a system and cannot change LLDB version, or want to package
  an affected version in a way that is compatible with these systems, the issue
  contains details of backports that could be done to fix the affected versions.
* When an ELF core file is loaded, LLDB now shows the command line that created the core file.
  If you need to see it again, use the command `process status -v`.
* LLDB now supports debugging Linux [Memory Protection Keys](https://docs.kernel.org/core-api/protection-keys.html)
  on AArch64 systems that have the Permission Overlay Extension (POE / FEAT_S1POE).
  See the [LLDB on AArch64 Linux](https://lldb.llvm.org/use/aarch64-linux.html#permission-overlay-extension-poe)
  guide for more information.

#### Windows

* Python 3.11 or later is now recommended for building LLDB 23 on Windows. From LLDB 24, Python 3.11 or later will be required.
* Messages from `OutputDebugString[A|W]` are now shown inline when using LLDB
  from the command-line and in the output window when using lldb-dap.


### Changes to BOLT

### Changes to Sanitizers

* Add a random delay into ThreadSanitizer to help find rare thread interleavings.

### Other Changes

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
