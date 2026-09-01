; REQUIRES: x86-registered-target, amdgpu-registered-target

; RUN: opt -mtriple=x86_64 -passes=pre-isel-intrinsic-lowering -S %s -o - | FileCheck %s --check-prefix=FALLBACK
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -passes=pre-isel-intrinsic-lowering -S %s -o - | FileCheck %s --check-prefix=AMD
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -passes=pre-isel-intrinsic-lowering -S %s -o - | FileCheck %s --check-prefix=NO-MAI
; RUN: opt -passes=pre-isel-intrinsic-lowering -S %s -o - | FileCheck %s --check-prefix=NO-TARGET

define i32 @vgpr(i32 %x) {
  %y = call i32 @llvm.experimental.regalloc.handoff(i32 %x, metadata !0)
  ret i32 %y
}

; FALLBACK-LABEL: define i32 @vgpr
; FALLBACK-NEXT: ret i32 %x
; AMD-LABEL: define i32 @vgpr
; AMD-NEXT: %y = call i32 @llvm.experimental.regalloc.handoff(i32 %x, metadata !0)
; AMD-NEXT: ret i32 %y
; NO-MAI-LABEL: define i32 @vgpr
; NO-MAI-NEXT: %y = call i32 @llvm.experimental.regalloc.handoff(i32 %x, metadata !0)
; NO-MAI-NEXT: ret i32 %y
; NO-TARGET-LABEL: define i32 @vgpr
; NO-TARGET-NEXT: %y = call i32 @llvm.experimental.regalloc.handoff(i32 %x, metadata !0)
; NO-TARGET-NEXT: ret i32 %y

define i32 @agpr(i32 %x) {
  %y = call i32 @llvm.experimental.regalloc.handoff(i32 %x, metadata !1)
  ret i32 %y
}

; FALLBACK-LABEL: define i32 @agpr
; FALLBACK-NEXT: ret i32 %x
; AMD-LABEL: define i32 @agpr
; AMD-NEXT: %y = call i32 @llvm.experimental.regalloc.handoff(i32 %x, metadata !1)
; AMD-NEXT: ret i32 %y
; NO-MAI-LABEL: define i32 @agpr
; NO-MAI-NEXT: ret i32 %x
; NO-TARGET-LABEL: define i32 @agpr
; NO-TARGET-NEXT: %y = call i32 @llvm.experimental.regalloc.handoff(i32 %x, metadata !1)
; NO-TARGET-NEXT: ret i32 %y

define i32 @unknown(i32 %x) {
  %y = call i32 @llvm.experimental.regalloc.handoff(i32 %x, metadata !2)
  ret i32 %y
}

; FALLBACK-LABEL: define i32 @unknown
; FALLBACK-NEXT: ret i32 %x
; AMD-LABEL: define i32 @unknown
; AMD-NEXT: ret i32 %x

define i32 @empty_constraint(i32 %x) {
  %y = call i32 @llvm.experimental.regalloc.handoff(i32 %x, metadata !3)
  ret i32 %y
}

; AMD-LABEL: define i32 @empty_constraint
; AMD-NEXT: ret i32 %x

define i32 @multi_element_constraint(i32 %x) {
  %y = call i32 @llvm.experimental.regalloc.handoff(i32 %x, metadata !4)
  ret i32 %y
}

; AMD-LABEL: define i32 @multi_element_constraint
; AMD-NEXT: ret i32 %x

define i32 @non_string_constraint(i32 %x) {
  %y = call i32 @llvm.experimental.regalloc.handoff(i32 %x, metadata !5)
  ret i32 %y
}

; AMD-LABEL: define i32 @non_string_constraint
; AMD-NEXT: ret i32 %x

declare i32 @llvm.experimental.regalloc.handoff(i32, metadata)

!0 = !{!"amdgpu.vgpr"}
!1 = !{!"amdgpu.agpr"}
!2 = !{!"amdgpu.unknown"}
!3 = !{}
!4 = !{!"amdgpu.vgpr", !"extra"}
!5 = !{i32 0}
