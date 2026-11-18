; Placement + addressing for `constant` globals under the ROPI/RWPI family,
; distinguishing six shapes:
;
;  * @relconst -- a const holding a DIRECT POINTER to another symbol (the
;    reduced shape of a Rust `dyn` vtable). needsDynamicRelocation() == true.
;    Under ropi-rwpi ONLY, it must (a) be placed in the writable,
;    relocation-bearing .data.rel.ro (so a loader can fix the slot after
;    placement is known) and (b) be addressed SB-relative via r9 -- the two
;    decisions must agree. Under static/ropi/rwpi it keeps stock behavior.
;
;  * @reltab -- a const holding a RELATIVE POINTER (ptrtoint(A)-ptrtoint(B),
;    both dso_local). needsRelocation() == true but needsDynamicRelocation()
;    == false: the entry is link-time-final, so it must stay in .rodata and
;    be addressed like read-only data (PC-relative under ropi/ropi-rwpi,
;    absolute otherwise) in EVERY model. Addressing this via r9 while it sits
;    in .rodata would compute S+Delta(rw) for a datum that moved with the RO
;    region -- the rust#95871 class of defect.
;
;  * @ext / @extmut -- the DECLARATION CONTRACT. A declaration exposes no
;    initializer, so the region classification must ride on the `constant`
;    flag, which the frontend asserts:
;      - `external constant` (@ext): the frontend promises the definition is
;        RO-region data -> PC-relative. rustc emits this under ropi-rwpi for
;        cross-CGU/cross-crate declarations of pure-data immutable statics
;        (whose definitions stay in .rodata).
;      - `external global` (@extmut): definition lives in the RW image
;        (mutable data, or a reloc-bearing const diverted to .data.rel.ro)
;        -> SB-relative via r9.
;    A `constant` declaration whose definition is actually reloc-bearing
;    (e.g. C `extern const`, or ThinLTO's EliminateAvailableExternally
;    stripping an initializer while keeping the constant bit) is
;    misclassified by this contract -- that remains the documented cross-TU
;    limitation: the truth is not computable from such a declaration.
;
;  * @pinned -- a reloc-bearing const with an EXPLICIT SECTION. The user owns
;    placement (linker scripts route pinned sections by NAME, conventionally
;    into the RO region), so BOTH halves keep stock behavior under ropi-rwpi:
;    the named section keeps "a" flags (no SHF_WRITE stamping) and the
;    address stays PC-relative, NOT r9. Diverting either half alone would
;    reintroduce the addressing-vs-placement mismatch.
;
;  * @mix -- a const holding BOTH a direct pointer and a relative pointer
;    whose target is a FUNCTION (text). One dynamic slot poisons the whole
;    object (max-over-operands), so the entire datum goes to .data.rel.ro +
;    r9 under ropi-rwpi. NOTE the loader contract implication: the
;    link-time-final `take_addr_relconst-mix` word then lives in the RW-slid
;    region while its target stays in text, so a loader for independently
;    sliding regions must rewrite such surviving REL32-class entries by
;    (Delta(target region) - Delta_rw) -- here (Delta_text - Delta_rw) --
;    or reject mixed objects. (A relative entry whose target is in the SAME
;    region as the slot, e.g. data-to-data, survives sliding unchanged.)
;
; RUN: llc -relocation-model=static    -mtriple=armv7a--none-eabi   < %s | FileCheck %s --check-prefixes=CHECK,ARM_ABS,RODATA
; RUN: llc -relocation-model=ropi      -mtriple=armv7a--none-eabi   < %s | FileCheck %s --check-prefixes=CHECK,ARM_PC,RODATA
; RUN: llc -relocation-model=rwpi      -mtriple=armv7a--none-eabi   < %s | FileCheck %s --check-prefixes=CHECK,ARM_ABS,RODATA
; RUN: llc -relocation-model=ropi-rwpi -mtriple=armv7a--none-eabi   < %s | FileCheck %s --check-prefixes=CHECK,ARM_RR,RELRO
; RUN: llc -relocation-model=rwpi      -mtriple=thumbv7m--none-eabi < %s | FileCheck %s --check-prefixes=CHECK,T2_ABS,RODATA
; RUN: llc -relocation-model=ropi-rwpi -mtriple=thumbv7m--none-eabi < %s | FileCheck %s --check-prefixes=CHECK,T2_RR,RELRO
;
; No-movt targets must take the literal-pool SBREL path for the diverted
; class; Thumb1 (v6m) additionally has no register-offset ADD with r9.
; RUN: llc -relocation-model=ropi-rwpi -mtriple=armv7a--none-eabi -mattr=+no-movt < %s | FileCheck %s --check-prefixes=NOMOVT_RR,RELRO
; RUN: llc -relocation-model=ropi-rwpi -mtriple=thumbv6m--none-eabi < %s | FileCheck %s --check-prefixes=T1_RR,RELRO
;
; GlobalISel resolves RO-ness through the same ARMTargetLowering::isReadOnly
; hook, so parity (including the explicit-section carve-out) is automatic;
; abort=1 also asserts nothing in this file falls back to SelectionDAG.
; RUN: llc -relocation-model=ropi-rwpi -mtriple=armv7a--none-eabi -global-isel -global-isel-abort=1 < %s | FileCheck %s --check-prefixes=GISEL_RR,RELRO

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

@target = external global i32, align 4
@relconst = constant ptr @target, align 4

@f = dso_local global i32 1, align 4
@reltab = dso_local constant i32 sub (i32 ptrtoint (ptr @f to i32), i32 ptrtoint (ptr @reltab to i32)), align 4

@ext = external constant ptr
@extmut = external global i32

@pinned = dso_local constant ptr @target, section ".mysec", align 4

@mix = dso_local constant { ptr, i32 } { ptr @target, i32 sub (i32 ptrtoint (ptr @take_addr_relconst to i32), i32 ptrtoint (ptr @mix to i32)) }, align 4

define ptr @take_addr_relconst() {
entry:
  ret ptr @relconst
; CHECK-LABEL: take_addr_relconst:

; ARM_ABS: movw    r0, :lower16:relconst{{$}}
; ARM_ABS: movt    r0, :upper16:relconst{{$}}

; ARM_PC: movw    r0, :lower16:(relconst-([[LPC:.LPC[0-9]+_[0-9]+]]+8))
; ARM_PC: movt    r0, :upper16:(relconst-([[LPC]]+8))
; ARM_PC: [[LPC]]:
; ARM_PC-NEXT: add     r0, pc, r0

; ARM_RR: movw    r0, :lower16:relconst(sbrel)
; ARM_RR: movt    r0, :upper16:relconst(sbrel)
; ARM_RR: add     r0, r9, r0

; T2_ABS: movw    r0, :lower16:relconst{{$}}
; T2_ABS: movt    r0, :upper16:relconst{{$}}

; T2_RR: movw    r0, :lower16:relconst(sbrel)
; T2_RR: movt    r0, :upper16:relconst(sbrel)
; T2_RR: add     r0, r9

; NOMOVT_RR-LABEL: take_addr_relconst:
; NOMOVT_RR: ldr     r0, [[CPI:.LCPI[0-9]+_[0-9]+]]
; NOMOVT_RR-NEXT: add     r0, r9, r0
; NOMOVT_RR: [[CPI]]:
; NOMOVT_RR-NEXT: .long   relconst(sbrel)

; T1_RR-LABEL: take_addr_relconst:
; T1_RR: ldr     r0, [[CPI:.LCPI[0-9]+_[0-9]+]]
; T1_RR-NEXT: mov     r1, r9
; T1_RR-NEXT: adds    r0, r1, r0
; T1_RR: [[CPI]]:
; T1_RR-NEXT: .long   relconst(sbrel)

; GISEL_RR-LABEL: take_addr_relconst:
; GISEL_RR: movw    r0, :lower16:relconst(sbrel)
; GISEL_RR: movt    r0, :upper16:relconst(sbrel)
; GISEL_RR: add     r0, r9, r0

; CHECK: bx lr
}

define ptr @take_addr_reltab() {
entry:
  ret ptr @reltab
; CHECK-LABEL: take_addr_reltab:

; ARM_ABS: movw    r0, :lower16:{{(\.Lreltab\$local|reltab)$}}
; ARM_ABS: movt    r0, :upper16:{{(\.Lreltab\$local|reltab)$}}

; ARM_PC: movw    r0, :lower16:(.Lreltab$local-([[LPC1:.LPC[0-9]+_[0-9]+]]+8))
; ARM_PC: movt    r0, :upper16:(.Lreltab$local-([[LPC1]]+8))
; ARM_PC: [[LPC1]]:
; ARM_PC-NEXT: add     r0, pc, r0

; The relative-pointer const stays RO under ropi-rwpi: PC-relative, NOT r9.
; ARM_RR: movw    r0, :lower16:(.Lreltab$local-([[LPC1:.LPC[0-9]+_[0-9]+]]+8))
; ARM_RR: movt    r0, :upper16:(.Lreltab$local-([[LPC1]]+8))
; ARM_RR: [[LPC1]]:
; ARM_RR-NEXT: add     r0, pc, r0

; T2_ABS: movw    r0, :lower16:.Lreltab$local{{$}}
; T2_ABS: movt    r0, :upper16:.Lreltab$local{{$}}

; T2_RR: movw    r0, :lower16:(.Lreltab$local-([[LPC1:.LPC[0-9]+_[0-9]+]]+4))
; T2_RR: movt    r0, :upper16:(.Lreltab$local-([[LPC1]]+4))
; T2_RR: [[LPC1]]:
; T2_RR-NEXT: add     r0, pc

; CHECK: bx lr
}

define ptr @take_addr_ext() {
entry:
  ret ptr @ext
; CHECK-LABEL: take_addr_ext:

; ARM_ABS: movw    r0, :lower16:ext{{$}}
; ARM_ABS: movt    r0, :upper16:ext{{$}}

; ARM_PC: movw    r0, :lower16:(ext-([[LPCE:.LPC[0-9]+_[0-9]+]]+8))
; ARM_PC: movt    r0, :upper16:(ext-([[LPCE]]+8))
; ARM_PC: [[LPCE]]:
; ARM_PC-NEXT: add     r0, pc, r0

; Declaration contract, RO arm: `external constant` -> PC-relative. This is
; what rustc's decl-marking relies on for pure-data statics; it is WRONG for
; a reloc-bearing definition (the documented cross-TU limitation).
; ARM_RR: movw    r0, :lower16:(ext-([[LPCE:.LPC[0-9]+_[0-9]+]]+8))
; ARM_RR: movt    r0, :upper16:(ext-([[LPCE]]+8))
; ARM_RR: [[LPCE]]:
; ARM_RR-NEXT: add     r0, pc, r0

; T2_ABS: movw    r0, :lower16:ext{{$}}
; T2_ABS: movt    r0, :upper16:ext{{$}}

; T2_RR: movw    r0, :lower16:(ext-([[LPCE:.LPC[0-9]+_[0-9]+]]+4))
; T2_RR: movt    r0, :upper16:(ext-([[LPCE]]+4))
; T2_RR: [[LPCE]]:
; T2_RR-NEXT: add     r0, pc

; NOMOVT_RR-LABEL: take_addr_ext:
; NOMOVT_RR: ldr     r0, [[CPIE:.LCPI[0-9]+_[0-9]+]]
; NOMOVT_RR: [[LPCE:.LPC[0-9]+_[0-9]+]]:
; NOMOVT_RR-NEXT: add     r0, pc, r0
; NOMOVT_RR: [[CPIE]]:
; NOMOVT_RR-NEXT: .long   ext-([[LPCE]]+8)

; T1_RR-LABEL: take_addr_ext:
; T1_RR: ldr     r0, [[CPIE:.LCPI[0-9]+_[0-9]+]]
; T1_RR: [[LPCE:.LPC[0-9]+_[0-9]+]]:
; T1_RR-NEXT: add     r0, pc
; T1_RR: [[CPIE]]:
; T1_RR-NEXT: .long   ext-([[LPCE]]+4)

; GISEL_RR-LABEL: take_addr_ext:
; GISEL_RR: movw    r0, :lower16:(ext-([[LPCGE:.LPC[0-9]+_[0-9]+]]+8))
; GISEL_RR: movt    r0, :upper16:(ext-([[LPCGE]]+8))
; GISEL_RR: [[LPCGE]]:
; GISEL_RR-NEXT: add     r0, pc, r0

; CHECK: bx lr
}

define ptr @take_addr_extmut() {
entry:
  ret ptr @extmut
; CHECK-LABEL: take_addr_extmut:

; Declaration contract, RW arm: `external global` (no constant flag) -> SB
; via r9. rustc emits this shape for declarations of mutable statics AND of
; reloc-bearing immutable statics (whose definitions are diverted to
; .data.rel.ro) -- both live in the r9-addressed RW image.
; ARM_RR: movw    r0, :lower16:extmut(sbrel)
; ARM_RR: movt    r0, :upper16:extmut(sbrel)
; ARM_RR: add     r0, r9, r0

; T2_RR: movw    r0, :lower16:extmut(sbrel)
; T2_RR: movt    r0, :upper16:extmut(sbrel)
; T2_RR: add     r0, r9

; GISEL_RR-LABEL: take_addr_extmut:
; GISEL_RR: movw    r0, :lower16:extmut(sbrel)
; GISEL_RR: movt    r0, :upper16:extmut(sbrel)
; GISEL_RR: add     r0, r9, r0

; NOMOVT_RR-LABEL: take_addr_extmut:
; NOMOVT_RR: ldr     r0, [[CPIEM:.LCPI[0-9]+_[0-9]+]]
; NOMOVT_RR-NEXT: add     r0, r9, r0
; NOMOVT_RR: [[CPIEM]]:
; NOMOVT_RR-NEXT: .long   extmut(sbrel)

; T1_RR-LABEL: take_addr_extmut:
; T1_RR: ldr     r0, [[CPIEM:.LCPI[0-9]+_[0-9]+]]
; T1_RR-NEXT: mov     r1, r9
; T1_RR-NEXT: adds    r0, r1, r0
; T1_RR: [[CPIEM]]:
; T1_RR-NEXT: .long   extmut(sbrel)

; CHECK: bx lr
}

define ptr @take_addr_pinned() {
entry:
  ret ptr @pinned
; CHECK-LABEL: take_addr_pinned:

; ARM_ABS: movw    r0, :lower16:{{(\.Lpinned\$local|pinned)$}}
; ARM_ABS: movt    r0, :upper16:{{(\.Lpinned\$local|pinned)$}}

; ARM_PC: movw    r0, :lower16:(.Lpinned$local-([[LPCP:.LPC[0-9]+_[0-9]+]]+8))
; ARM_PC: movt    r0, :upper16:(.Lpinned$local-([[LPCP]]+8))
; ARM_PC: [[LPCP]]:
; ARM_PC-NEXT: add     r0, pc, r0

; Explicit-section carve-out: the user pinned the section, so BOTH halves
; keep stock behavior under ropi-rwpi -- PC-relative address (NOT r9), and
; the named section below keeps "a" flags (no SHF_WRITE stamping).
; ARM_RR: movw    r0, :lower16:(.Lpinned$local-([[LPCP:.LPC[0-9]+_[0-9]+]]+8))
; ARM_RR: movt    r0, :upper16:(.Lpinned$local-([[LPCP]]+8))
; ARM_RR: [[LPCP]]:
; ARM_RR-NEXT: add     r0, pc, r0

; T2_ABS: movw    r0, :lower16:.Lpinned$local{{$}}
; T2_ABS: movt    r0, :upper16:.Lpinned$local{{$}}

; T2_RR: movw    r0, :lower16:(.Lpinned$local-([[LPCP:.LPC[0-9]+_[0-9]+]]+4))
; T2_RR: movt    r0, :upper16:(.Lpinned$local-([[LPCP]]+4))
; T2_RR: [[LPCP]]:
; T2_RR-NEXT: add     r0, pc

; GISEL_RR-LABEL: take_addr_pinned:
; GISEL_RR: movw    r0, :lower16:(.Lpinned$local-([[LPCGP:.LPC[0-9]+_[0-9]+]]+8))
; GISEL_RR: movt    r0, :upper16:(.Lpinned$local-([[LPCGP]]+8))
; GISEL_RR: [[LPCGP]]:
; GISEL_RR-NEXT: add     r0, pc, r0

; CHECK: bx lr
}

define ptr @take_addr_mix() {
entry:
  ret ptr @mix
; CHECK-LABEL: take_addr_mix:

; ARM_ABS: movw    r0, :lower16:{{(\.Lmix\$local|mix)$}}
; ARM_ABS: movt    r0, :upper16:{{(\.Lmix\$local|mix)$}}

; ARM_PC: movw    r0, :lower16:(.Lmix$local-([[LPCM:.LPC[0-9]+_[0-9]+]]+8))
; ARM_PC: movt    r0, :upper16:(.Lmix$local-([[LPCM]]+8))
; ARM_PC: [[LPCM]]:
; ARM_PC-NEXT: add     r0, pc, r0

; One dynamic slot poisons the whole object: .data.rel.ro + r9.
; ARM_RR: movw    r0, :lower16:.Lmix$local(sbrel)
; ARM_RR: movt    r0, :upper16:.Lmix$local(sbrel)
; ARM_RR: add     r0, r9, r0

; T2_ABS: movw    r0, :lower16:.Lmix$local{{$}}
; T2_ABS: movt    r0, :upper16:.Lmix$local{{$}}

; T2_RR: movw    r0, :lower16:.Lmix$local(sbrel)
; T2_RR: movt    r0, :upper16:.Lmix$local(sbrel)
; T2_RR: add     r0, r9

; CHECK: bx lr
}

; Section placement. Emission order in all configs:
; relconst, f (.data), reltab, pinned (.mysec), mix.

; RODATA: .section .rodata,"a",%progbits
; RODATA: relconst:
; RODATA-NEXT: .long target
; RODATA: reltab:
; RODATA: .long f-reltab
; RODATA: .section .mysec,"a",%progbits
; RODATA: pinned:
; RODATA: .long target
; RODATA: mix:
; RODATA: .long target
; RODATA-NEXT: .long take_addr_relconst-mix

; RELRO: .section .data.rel.ro,"aw",%progbits
; RELRO: relconst:
; RELRO-NEXT: .long target
; RELRO: .section .rodata,"a",%progbits
; RELRO: reltab:
; RELRO: .long f-reltab
; RELRO: .section .mysec,"a",%progbits
; RELRO: pinned:
; RELRO: .long target
; RELRO: .section .data.rel.ro,"aw",%progbits
; RELRO: mix:
; RELRO: .long target
; RELRO-NEXT: .long take_addr_relconst-mix
