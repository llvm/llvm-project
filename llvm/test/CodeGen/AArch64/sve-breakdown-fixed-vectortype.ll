; RUN: llc -mtriple=aarch64--linux < %s | FileCheck %s --check-prefix=SVE-ABI

; Fixed-length vectors forced into SVE Z registers still need ABI copies
; broken down into Q register pairs when lowering fixed-width memory ops.
define void @sve_fixed_vector_breakdown_v8i32(ptr %x, ptr %y) {
; SVE-ABI-LABEL: sve_fixed_vector_breakdown_v8i32:
; SVE-ABI:       // %bb.0:
; SVE-ABI-NEXT:    //APP
; SVE-ABI-NEXT:    //NO_APP
; SVE-ABI-NEXT:    stp	q1, q2, [x1]
; SVE-ABI-NEXT:    ldp	q0, q1, [x0]
; SVE-ABI-NEXT:    //APP
; SVE-ABI-NEXT:    //NO_APP
; SVE-ABI-NEXT:    ret
entry:
  %y.y = call <8 x i32> asm sideeffect "", "={z1}"()
  store <8 x i32> %y.y, ptr %y, align 16
  %x.x = load <8 x i32>, ptr %x, align 16
  call void asm sideeffect "", "{z0}"(<8 x i32> %x.x)
  ret void
}

define void @sve_fixed_vector_breakdown_v4i64(ptr %x, ptr %y) {
; SVE-ABI-LABEL: sve_fixed_vector_breakdown_v4i64:
; SVE-ABI:       // %bb.0:
; SVE-ABI-NEXT:    //APP
; SVE-ABI-NEXT:    //NO_APP
; SVE-ABI-NEXT:    stp	q1, q2, [x1]
; SVE-ABI-NEXT:    ldp	q0, q1, [x0]
; SVE-ABI-NEXT:    //APP
; SVE-ABI-NEXT:    //NO_APP
; SVE-ABI-NEXT:    ret
entry:
  %y.y = call <4 x i64> asm sideeffect "", "={z1}"()
  store <4 x i64> %y.y, ptr %y, align 16
  %x.x = load <4 x i64>, ptr %x, align 16
  call void asm sideeffect "", "{z0}"(<4 x i64> %x.x)
  ret void
}

define void @sve_fixed_vector_breakdown_v16i8(ptr %x, ptr %y) {
; SVE-ABI-LABEL: sve_fixed_vector_breakdown_v16i8:
; SVE-ABI:       // %bb.0:
; SVE-ABI-NEXT:    //APP
; SVE-ABI-NEXT:    //NO_APP
; SVE-ABI-NEXT:    str q1, [x1]
; SVE-ABI-NEXT:    ldr q0, [x0]
; SVE-ABI-NEXT:    //APP
; SVE-ABI-NEXT:    //NO_APP
; SVE-ABI-NEXT:    ret
entry:
  %y.y = call <16 x i8> asm sideeffect "", "={z1}"()
  store <16 x i8> %y.y, ptr %y, align 16
  %x.x = load <16 x i8>, ptr %x, align 16
  call void asm sideeffect "", "{z0}"(<16 x i8> %x.x)
  ret void
}
