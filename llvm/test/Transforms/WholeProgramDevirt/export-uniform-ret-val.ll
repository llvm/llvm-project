; RUN: opt -wholeprogramdevirt -whole-program-visibility -wholeprogramdevirt-summary-action=export -wholeprogramdevirt-read-summary=%S/Inputs/export.yaml -wholeprogramdevirt-write-summary=%t -S -o - %s | FileCheck %s
; RUN: FileCheck --check-prefix=SUMMARY %s < %t

; SUMMARY-NOT: TypeTests:

; SUMMARY:      TypeIdMap:
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
; SUMMARY-NEXT:         Kind:            Indir
; SUMMARY-NEXT:         SingleImplName:  ''
; SUMMARY-NEXT:         ResByArg:
; SUMMARY-NEXT:           24,12:
; SUMMARY-NEXT:             Kind:            UniformRetVal
; SUMMARY-NEXT:             Info:            36
; SUMMARY-NEXT:             Byte:            0
; SUMMARY-NEXT:             Bit:             0

; CHECK: @vt4a = constant i32 (i8*, i32, i32)* @vf4a
@vt4a = constant i32 (i8*, i32, i32)* @vf4a, !type !0

; CHECK: @vt4b = constant i32 (i8*, i32, i32)* @vf4b
@vt4b = constant i32 (i8*, i32, i32)* @vf4b, !type !0

define i32 @vf4a(i8*, i32 %x, i32 %y) {
  %z = add i32 %x, %y
  ret i32 %z
}

define i32 @vf4b(i8*, i32 %x, i32 %y) {
  ret i32 36
}

!0 = !{i32 0, !"typeid4"}
