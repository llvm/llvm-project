; RUN: opt -wholeprogramdevirt -whole-program-visibility -wholeprogramdevirt-summary-action=export -wholeprogramdevirt-read-summary=%S/Inputs/export.yaml -wholeprogramdevirt-write-summary=%t -o /dev/null %s
; RUN: FileCheck %s < %t

; CHECK:       TypeTests: [ 15427464259790519041, 17525413373118030901 ]
; CHECK-NEXT:  TypeTestAssumeVCalls:

@vt1a = constant void (i8*)* @vf1a, !type !0
@vt1b = constant void (i8*)* @vf1b, !type !0
@vt2a = constant void (i8*)* @vf2a, !type !1
@vt2b = constant void (i8*)* @vf2b, !type !1
@vt3a = constant void (i8*)* @vf3a, !type !2
@vt3b = constant void (i8*)* @vf3b, !type !2
@vt4a = constant void (i8*)* @vf4a, !type !3
@vt4b = constant void (i8*)* @vf4b, !type !3

declare void @vf1a(i8*)
declare void @vf1b(i8*)
declare void @vf2a(i8*)
declare void @vf2b(i8*)
declare void @vf3a(i8*)
declare void @vf3b(i8*)
declare void @vf4a(i8*)
declare void @vf4b(i8*)

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 0, !"typeid2"}
!2 = !{i32 0, !"typeid3"}
!3 = !{i32 0, !"typeid4"}
