; RUN: opt -wholeprogramdevirt -whole-program-visibility -wholeprogramdevirt-summary-action=export -wholeprogramdevirt-read-summary=%S/Inputs/export.yaml -wholeprogramdevirt-write-summary=%t -S -o - %s | FileCheck %s
; RUN: FileCheck --check-prefix=SUMMARY %s < %t

; SUMMARY:      TypeIdMap:
; SUMMARY-NEXT:   typeid3:
; SUMMARY-NEXT:     TTRes:
; SUMMARY-NEXT:       Kind:            Unsat
; SUMMARY-NEXT:       SizeM1BitWidth:  0
; SUMMARY-NEXT:       AlignLog2:       0
; SUMMARY-NEXT:       SizeM1:          0
; SUMMARY-NEXT:       BitMask:         0
; SUMMARY-NEXT:       InlineBits:      0
; SUMMARY-NEXT:     WPDRes:
; SUMMARY-NEXT:       0:
; SUMMARY-NEXT:         Kind:            SingleImpl
; SUMMARY-NEXT:         SingleImplName:  'vf3$merged'
; SUMMARY-NEXT:         ResByArg:
; SUMMARY-NEXT:   typeid1:
; SUMMARY-NEXT:     TTRes:
; SUMMARY-NEXT:       Kind:            Unsat
; SUMMARY-NEXT:       SizeM1BitWidth:  0
; SUMMARY-NEXT:       AlignLog2:       0
; SUMMARY-NEXT:       SizeM1:          0
; SUMMARY-NEXT:       BitMask:         0
; SUMMARY-NEXT:       InlineBits:      0
; SUMMARY-NEXT:     WPDRes:
; SUMMARY-NEXT:       0:
; SUMMARY-NEXT:         Kind:            SingleImpl
; SUMMARY-NEXT:         SingleImplName:  vf1
; SUMMARY-NEXT:         ResByArg:
; SUMMARY-NEXT:   typeid2:
; SUMMARY-NEXT:     TTRes:
; SUMMARY-NEXT:       Kind:            Unsat
; SUMMARY-NEXT:       SizeM1BitWidth:  0
; SUMMARY-NEXT:       AlignLog2:       0
; SUMMARY-NEXT:       SizeM1:          0
; SUMMARY-NEXT:       BitMask:         0
; SUMMARY-NEXT:       InlineBits:      0
; SUMMARY-NEXT:     WPDRes:
; SUMMARY-NEXT:       0:
; SUMMARY-NEXT:         Kind:            SingleImpl
; SUMMARY-NEXT:         SingleImplName:  vf2
; SUMMARY-NEXT:         ResByArg:
; SUMMARY-NEXT:   typeid4:
; SUMMARY-NEXT:     TTRes:
; SUMMARY-NEXT:       Kind:            Unsat
; SUMMARY-NEXT:       SizeM1BitWidth:  0
; SUMMARY-NEXT:       AlignLog2:       0
; SUMMARY-NEXT:       SizeM1:          0
; SUMMARY-NEXT:       BitMask:         0
; SUMMARY-NEXT:       InlineBits:      0
; SUMMARY-NEXT:     WPDRes:
; SUMMARY-NEXT:       0:
; SUMMARY-NEXT:         Kind:            SingleImpl
; SUMMARY-NEXT:         SingleImplName:  'vf4$merged'
; SUMMARY-NEXT:         ResByArg:
; SUMMARY-NEXT: WithGlobalValueDeadStripping: false
; SUMMARY-NEXT: ...

; CHECK: $"vf4$merged" = comdat largest
$vf4 = comdat largest

; CHECK: @vt1 = constant void (i8*)* @vf1
@vt1 = constant void (i8*)* @vf1, !type !0

; CHECK: @vt2 = constant void (i8*)* @vf2
@vt2 = constant void (i8*)* @vf2, !type !1

@vt3 = constant void (i8*)* @vf3, !type !2

; CHECK: @vt4 = constant void (i8*)* @"vf4$merged", comdat($"vf4$merged")
@vt4 = constant void (i8*)* @vf4, comdat($vf4), !type !3

@vt5 = constant void (i8*)* @vf5, !type !4

; CHECK: declare void @vf1(i8*)
declare void @vf1(i8*)

; CHECK: define void @vf2(i8* %0)
define void @vf2(i8*) {
  ret void
}

; CHECK: define hidden void @"vf3$merged"(i8* %0) {
define internal void @vf3(i8*) {
  ret void
}

; CHECK: define hidden void @"vf4$merged"(i8* %0) comdat {
define internal void @vf4(i8*) comdat {
  ret void
}

declare void @vf5(i8*)

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 0, !"typeid2"}
!2 = !{i32 0, !"typeid3"}
!3 = !{i32 0, !"typeid4"}
!4 = !{i32 0, !5}
!5 = distinct !{}
