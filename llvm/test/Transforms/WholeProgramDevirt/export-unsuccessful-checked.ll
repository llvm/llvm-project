; RUN: opt -passes=wholeprogramdevirt -whole-program-visibility -wholeprogramdevirt-summary-action=export -wholeprogramdevirt-read-summary=%S/Inputs/export.yaml -wholeprogramdevirt-write-summary=%t -o /dev/null %s
; RUN: FileCheck %s < %t

; CHECK:       TypeTests: [ 15427464259790519041, 17525413373118030901 ]
; CHECK-NEXT:  TypeTestAssumeVCalls:

@vt1a = constant ptr @vf1a, !type !0
@vt1b = constant ptr @vf1b, !type !0
@vt2a = constant ptr @vf2a, !type !1
@vt2b = constant ptr @vf2b, !type !1
@vt3a = constant ptr @vf3a, !type !2
@vt3b = constant ptr @vf3b, !type !2
@vt4a = constant ptr @vf4a, !type !3
@vt4b = constant ptr @vf4b, !type !3

declare void @vf1a(ptr)
declare void @vf1b(ptr)
declare void @vf2a(ptr)
declare void @vf2b(ptr)
declare void @vf3a(ptr)
declare void @vf3b(ptr)
declare void @vf4a(ptr)
declare void @vf4b(ptr)

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 0, !"typeid2"}
!2 = !{i32 0, !"typeid3"}
!3 = !{i32 0, !"typeid4"}
