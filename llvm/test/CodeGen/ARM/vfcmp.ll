; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

; This tests fcmp operations that do not map directly to NEON instructions.

; une is implemented with VCEQ/VMVN
define <2 x i32> @vcunef32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: vcunef32:
;CHECK: vceq.f32
;CHECK-NEXT: vmvn
  %tmp1 = load <2 x float>, <2 x float>* %A
  %tmp2 = load <2 x float>, <2 x float>* %B
  %tmp3 = fcmp une <2 x float> %tmp1, %tmp2
  %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
  ret <2 x i32> %tmp4
}

; olt is implemented with VCGT
define <2 x i32> @vcoltf32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: vcoltf32:
;CHECK: vcgt.f32
  %tmp1 = load <2 x float>, <2 x float>* %A
  %tmp2 = load <2 x float>, <2 x float>* %B
  %tmp3 = fcmp olt <2 x float> %tmp1, %tmp2
  %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
  ret <2 x i32> %tmp4
}

; ole is implemented with VCGE
define <2 x i32> @vcolef32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: vcolef32:
;CHECK: vcge.f32
  %tmp1 = load <2 x float>, <2 x float>* %A
  %tmp2 = load <2 x float>, <2 x float>* %B
  %tmp3 = fcmp ole <2 x float> %tmp1, %tmp2
  %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
  ret <2 x i32> %tmp4
}

; uge is implemented with VCGT/VMVN
define <2 x i32> @vcugef32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: vcugef32:
;CHECK: vcgt.f32
;CHECK-NEXT: vmvn
  %tmp1 = load <2 x float>, <2 x float>* %A
  %tmp2 = load <2 x float>, <2 x float>* %B
  %tmp3 = fcmp uge <2 x float> %tmp1, %tmp2
  %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
  ret <2 x i32> %tmp4
}

; ule is implemented with VCGT/VMVN
define <2 x i32> @vculef32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: vculef32:
;CHECK: vcgt.f32
;CHECK-NEXT: vmvn
  %tmp1 = load <2 x float>, <2 x float>* %A
  %tmp2 = load <2 x float>, <2 x float>* %B
  %tmp3 = fcmp ule <2 x float> %tmp1, %tmp2
  %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
  ret <2 x i32> %tmp4
}

; ugt is implemented with VCGE/VMVN
define <2 x i32> @vcugtf32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: vcugtf32:
;CHECK: vcge.f32
;CHECK-NEXT: vmvn
  %tmp1 = load <2 x float>, <2 x float>* %A
  %tmp2 = load <2 x float>, <2 x float>* %B
  %tmp3 = fcmp ugt <2 x float> %tmp1, %tmp2
  %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
  ret <2 x i32> %tmp4
}

; ult is implemented with VCGE/VMVN
define <2 x i32> @vcultf32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: vcultf32:
;CHECK: vcge.f32
;CHECK-NEXT: vmvn
  %tmp1 = load <2 x float>, <2 x float>* %A
  %tmp2 = load <2 x float>, <2 x float>* %B
  %tmp3 = fcmp ult <2 x float> %tmp1, %tmp2
  %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
  ret <2 x i32> %tmp4
}

; ueq is implemented with VCGT/VCGT/VORR/VMVN
define <2 x i32> @vcueqf32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: vcueqf32:
;CHECK: vcgt.f32
;CHECK-NEXT: vcgt.f32
;CHECK-NEXT: vorr
;CHECK-NEXT: vmvn
  %tmp1 = load <2 x float>, <2 x float>* %A
  %tmp2 = load <2 x float>, <2 x float>* %B
  %tmp3 = fcmp ueq <2 x float> %tmp1, %tmp2
  %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
  ret <2 x i32> %tmp4
}

; one is implemented with VCGT/VCGT/VORR
define <2 x i32> @vconef32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: vconef32:
;CHECK: vcgt.f32
;CHECK-NEXT: vcgt.f32
;CHECK-NEXT: vorr
  %tmp1 = load <2 x float>, <2 x float>* %A
  %tmp2 = load <2 x float>, <2 x float>* %B
  %tmp3 = fcmp one <2 x float> %tmp1, %tmp2
  %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
  ret <2 x i32> %tmp4
}

; uno is implemented with VCGT/VCGE/VORR/VMVN
define <2 x i32> @vcunof32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: vcunof32:
;CHECK: vcge.f32
;CHECK-NEXT: vcgt.f32
;CHECK-NEXT: vorr
;CHECK-NEXT: vmvn
  %tmp1 = load <2 x float>, <2 x float>* %A
  %tmp2 = load <2 x float>, <2 x float>* %B
  %tmp3 = fcmp uno <2 x float> %tmp1, %tmp2
  %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
  ret <2 x i32> %tmp4
}

; ord is implemented with VCGT/VCGE/VORR
define <2 x i32> @vcordf32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: vcordf32:
;CHECK: vcge.f32
;CHECK-NEXT: vcgt.f32
;CHECK-NEXT: vorr
  %tmp1 = load <2 x float>, <2 x float>* %A
  %tmp2 = load <2 x float>, <2 x float>* %B
  %tmp3 = fcmp ord <2 x float> %tmp1, %tmp2
  %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
  ret <2 x i32> %tmp4
}
