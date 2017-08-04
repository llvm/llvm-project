; RUN: opt -wholeprogramdevirt -wholeprogramdevirt-summary-action=export -wholeprogramdevirt-read-summary=%S/Inputs/export.yaml -wholeprogramdevirt-write-summary=%t -S -o - %s | FileCheck %s
; RUN: FileCheck --check-prefix=SUMMARY %s < %t

; SUMMARY-NOT:  TypeTests:

; SUMMARY:      TypeIdMap:
; SUMMARY-NEXT:   typeid3:
; SUMMARY-NEXT:     TTRes:
; SUMMARY-NEXT:       Kind:            Unsat
; SUMMARY-NEXT:       SizeM1BitWidth:  0
; SUMMARY-NEXT:     WPDRes:
; SUMMARY-NEXT:       0:
; SUMMARY-NEXT:         Kind:            Indir
; SUMMARY-NEXT:         SingleImplName:  ''
; SUMMARY-NEXT:         ResByArg:
; SUMMARY-NEXT:           12,24:
; SUMMARY-NEXT:             Kind:            UniqueRetVal
; SUMMARY-NEXT:             Info:            0
; SUMMARY-NEXT:   typeid4:
; SUMMARY-NEXT:     TTRes:
; SUMMARY-NEXT:       Kind:            Unsat
; SUMMARY-NEXT:       SizeM1BitWidth:  0
; SUMMARY-NEXT:     WPDRes:
; SUMMARY-NEXT:       0:
; SUMMARY-NEXT:         Kind:            Indir
; SUMMARY-NEXT:         SingleImplName:  ''
; SUMMARY-NEXT:         ResByArg:
; SUMMARY-NEXT:           24,12:
; SUMMARY-NEXT:             Kind:            UniqueRetVal
; SUMMARY-NEXT:             Info:            1

; CHECK: @vt3a = constant i1 (i8*, i32, i32)* @vf3a
@vt3a = constant i1 (i8*, i32, i32)* @vf3a, !type !0

; CHECK: @vt3b = constant i1 (i8*, i32, i32)* @vf3b
@vt3b = constant i1 (i8*, i32, i32)* @vf3b, !type !0

; CHECK: @vt3c = constant i1 (i8*, i32, i32)* @vf3c
@vt3c = constant i1 (i8*, i32, i32)* @vf3c, !type !0

; CHECK: @vt4a = constant i1 (i8*, i32, i32)* @vf4a
@vt4a = constant i1 (i8*, i32, i32)* @vf4a, !type !1

; CHECK: @vt4b = constant i1 (i8*, i32, i32)* @vf4b
@vt4b = constant i1 (i8*, i32, i32)* @vf4b, !type !1

; CHECK: @vt4c = constant i1 (i8*, i32, i32)* @vf4c
@vt4c = constant i1 (i8*, i32, i32)* @vf4c, !type !1

; CHECK: @__typeid_typeid3_0_12_24_unique_member = hidden alias i8, bitcast (i1 (i8*, i32, i32)** @vt3b to i8*)
; CHECK: @__typeid_typeid4_0_24_12_unique_member = hidden alias i8, bitcast (i1 (i8*, i32, i32)** @vt4b to i8*)

define i1 @vf3a(i8*, i32, i32) {
  ret i1 true
}

define i1 @vf3b(i8*, i32, i32) {
  ret i1 false
}

define i1 @vf3c(i8*, i32, i32) {
  ret i1 true
}

define i1 @vf4a(i8*, i32, i32) {
  ret i1 false
}

define i1 @vf4b(i8*, i32, i32) {
  ret i1 true
}

define i1 @vf4c(i8*, i32, i32) {
  ret i1 false
}

!0 = !{i32 0, !"typeid3"}
!1 = !{i32 0, !"typeid4"}
