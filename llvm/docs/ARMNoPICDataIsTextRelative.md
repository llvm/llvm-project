# ARM `-mno-pic-data-is-text-relative`

This document records the ARM semantics expected from Clang and LLVM for
`-mno-pic-data-is-text-relative`. The contract is based on GCC's ARM option
model, with the initial LLVM implementation deliberately scoped to the fixed
EABI static-base form.

## Problem Model

Normal ARM PIC code may assume that the displacement between the text segment
and the data segment is fixed at static link time. With that assumption, the
compiler can address eligible local data with PC-relative relocations. This is
the default GCC behavior for non-VxWorks RTP ARM targets.

`-mno-pic-data-is-text-relative` removes that assumption. Code may remain fixed
while the writable data segment is independently relocated. Data references
that cannot be proven text-relative must therefore be addressed through a GOT
base instead of through the text segment.

The option is meaningful only for PIC or PIE code generation. Clang diagnoses
`-mno-pic-data-is-text-relative` without `-fpic`, `-fPIC`, `-fpie`, or `-fPIE`.

## GCC ARM Semantics

GCC exposes three related ARM options:

* `-mpic-data-is-text-relative` and `-mno-pic-data-is-text-relative` select
  whether known data may be addressed relative to text.
* `-msingle-pic-base` tells GCC that the PIC base register is read-only and is
  initialized by the runtime before execution.
* `-mpic-register=<reg>` chooses the PIC base register.

For GCC ARM EABI targets, disabling text-relative data also enables
`-msingle-pic-base` by default, unless the user explicitly selected
`-mno-single-pic-base`. In that default single-base mode, GCC uses `r9` as the
PIC base register. A runtime or loader must initialize `r9` before executing the
generated code.

GCC still allows an explicit `-mno-single-pic-base` override. In that mode it
does not rely on a pre-initialized `r9`; it computes the GOT base in generated
code. GCC also allows `-mpic-register=<reg>` for explicit PIC register
selection, subject to target validity checks.

GCC's address selection has these important properties:

* Local non-weak data symbols and labels may use text-relative addressing when
  `-mpic-data-is-text-relative` is active.
* External data, interposable data, and weak data use GOT addressing.
* With `-mno-pic-data-is-text-relative`, data references use GOT addressing.
* Non-FDPIC function references are not data references and may remain
  text-relative.
* Text constant-pool entries follow the same distinction: GCC emits GOT-style
  relocations when text-relative data is disabled, and GOTOFF-style relocations
  for eligible local references when it is enabled.

## LLVM Scope

The first LLVM implementation follows GCC's default EABI behavior for
`-mno-pic-data-is-text-relative`: data is addressed through the GOT using `r9`
as the static base register. This matches the GCC mode that is selected by
default when text-relative data is disabled on ARM EABI targets.

The LLVM option surface is intentionally narrower than GCC's for this branch:

* Clang accepts `-mpic-data-is-text-relative` and
  `-mno-pic-data-is-text-relative` for ARM, Thumb, and SystemZ.
* LLVM does not currently expose ARM `-msingle-pic-base`,
  `-mno-single-pic-base`, or `-mpic-register=<reg>`.
* Because custom PIC registers are not exposed, the static base register is
  fixed to `r9`.

This means LLVM does not yet duplicate GCC's explicit `-mno-single-pic-base`
mode, where the GOT base is recomputed instead of read from a fixed runtime
register. It also does not yet duplicate GCC's user-selected PIC register mode.

## Code Generation Contract

For ARM and Thumb PIC or PIE code:

* The default is `-mpic-data-is-text-relative`.
* Under `-mpic-data-is-text-relative`, local non-weak data may use
  text-relative addressing.
* Under `-mno-pic-data-is-text-relative`, data addresses must be materialized
  through GOT relocations and the static base register.
* External, interposable, and weak data must use GOT addressing in both modes.
* Non-FDPIC local function addresses may stay text-relative in both modes.
* `r9` must be reserved from general register allocation and treated as an
  incoming ABI register whenever it is used as the static base.
* The generated object must describe `r9` as the static base register through
  the ARM build attribute `Tag_ABI_PCS_R9_use = R9IsSB` when the option causes
  `r9` to be used as the static base.

The runtime contract is the same as GCC's default single-base EABI contract:
when `-mno-pic-data-is-text-relative` is used, startup or loader code must set
`r9` to the GOT/static-base value expected by the linked image before entering
compiled code.

## Implemented Interactions

The implemented driver and code-generation behavior covers these cases:

* `-fpic`, `-fPIC`, `-fpie`, and `-fPIE` driver forwarding.
* Multilib flag normalization for ARM PIC data mode, so runtime variants can
  distinguish text-relative PIC data from static-base GOT data.
* Rejection or diagnosis when the option is used without PIC or PIE.
* Rejection when `-mno-pic-data-is-text-relative` is combined with
  `-ffixed-r9`, because the GCC-compatible default mode requires `r9` as the
  static base register.
* Existing ROPI/RWPI driver validation prevents combinations with PIC or PIE,
  so `-mno-pic-data-is-text-relative` cannot be combined with `-fropi`,
  `-frwpi`, or `-fropi -frwpi`.
* ARM and Thumb code generation, including Thumb-1.
* SelectionDAG, FastISel, and GlobalISel lowering paths.
* Local data, hidden data, external data, weak data, and local function
  addresses.
* Object attributes for `Tag_ABI_PCS_R9_use`.
* The Clang predefined macro `__ARM_PIC_DATA_STATIC_BASE=1` when ARM or Thumb
  PIC or PIE code uses the fixed-`r9` static-base data mode. GCC does not define
  an equivalent macro for `-mno-pic-data-is-text-relative`; this macro is added
  so C or assembly startup code can conditionally initialize the static base.

## Interactions To Review

Review and test coverage should explicitly cover these areas before upstream
submission:

* Targets that reserve `r9` for platform reasons other than `-ffixed-r9`.
* Assembly or runtime code that accesses global data without compiler-generated
  address materialization.

The initial runtime assembly audit found compiler-rt examples that need
conditional static-base handling when those runtimes are built for this mode:

* `compiler-rt/lib/sanitizer_common/sanitizer_common_interceptors_vfork_arm.inc.S`
  uses `__ARM_PIC_DATA_STATIC_BASE` to select a GOT-offset load of the
  `real_vfork` data symbol through `r9`.
* `compiler-rt/lib/xray/xray_trampoline_arm.S` loads the
  `__xray::XRayPatchedFunction` data symbol through an `r9`-relative GOT entry
  when `__ARM_PIC_DATA_STATIC_BASE` is defined.
* `libunwind/src/UnwindRegistersSave.S` and
  `libunwind/src/UnwindRegistersRestore.S` save and restore `r9` as part of ARM
  register state, but do not by themselves access global data through a
  hand-written address sequence in the audited ARM paths.

The explicit GCC options `-m[no-]single-pic-base` and `-mpic-register=<reg>`
are design decisions rather than accidental omissions. If LLVM needs complete
GCC command-line compatibility, those options require separate implementation
and tests.
