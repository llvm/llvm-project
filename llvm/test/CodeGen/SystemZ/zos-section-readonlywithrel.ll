; Test that a read-only global containing pointer relocations (e.g. a C++
; vtable) is placed in the writable static area (C_WSA64) and NOT in the
; constant/text section (C_CODE64) on z/OS.
;
; z/OS uses DynamicNoPIC as its effective relocation model.  This causes
; getKindForGlobal() to skip the Reloc::Static short-circuit and classify
; constant globals that needsDynamicRelocation() as ReadOnlyWithRel instead
; of ReadOnly.  SelectSectionForGlobal then routes ReadOnlyWithRel to a
; per-symbol C_WSA64 PR section (ESD_LB_Deferred) rather than C_CODE64
; (ESD_LB_Initial, read-only/executable).
;
; RUN: llc < %s -mtriple=s390x-ibm-zos | FileCheck %s

@_ZTV4Base = hidden constant [2 x ptr] [ptr null, ptr @_ZTI4Base], align 8
@_ZTI4Base = external hidden global ptr

; The vtable gets its own SD and a C_WSA64 ED (DEFLOAD, NOTEXECUTABLE).
; CHECK: _ZTV4Base CSECT
; CHECK-NEXT: C_WSA64 CATTR {{.*}}DEFLOAD,NOTEXECUTABLE
; CHECK: _ZTV4Base XATTR LINKAGE(XPLINK),REFERENCE(DATA)

; The vtable data (null + reloc to _ZTI4Base) appears inside the WSA section.
; CHECK: DC XL8'0000000000000000'
; CHECK-NEXT: DC AD(_ZTI4Base)

; Crucially, the vtable must NOT be emitted directly inside C_CODE64.
; CHECK-NOT: C_CODE64{{.*}}_ZTV4Base
