; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; Check that module flags are structurally correct.
;
; CHECK: incorrect number of operands in module flag
; CHECK: !0
!0 = !{i32 1}
; CHECK: invalid behavior operand in module flag (expected constant integer)
; CHECK: !"foo"
!1 = !{!"foo", !"foo", i32 42}
; CHECK: invalid behavior operand in module flag (unexpected constant)
; CHECK: i32 999
!2 = !{i32 999, !"foo", i32 43}
; CHECK: invalid ID operand in module flag (expected metadata string)
; CHECK: i32 1
!3 = !{i32 1, i32 1, i32 44}
; CHECK: invalid value for 'require' module flag (expected metadata pair)
; CHECK: i32 45
!4 = !{i32 3, !"bla", i32 45}
; CHECK: invalid value for 'require' module flag (expected metadata pair)
; CHECK: !
!5 = !{i32 3, !"bla", !{i32 46}}
; CHECK: invalid value for 'require' module flag (first value operand should be a string)
; CHECK: i32 47
!6 = !{i32 3, !"bla", !{i32 47, i32 48}}

; Check that module flags only have unique IDs.
;
; CHECK: module flag identifiers must be unique (or of 'require' type)
!7 = !{i32 1, !"foo", i32 49}
!8 = !{i32 2, !"foo", i32 50}
; CHECK-NOT: module flag identifiers must be unique
!9 = !{i32 2, !"bar", i32 51}
!10 = !{i32 3, !"bar", !{!"bar", i32 51}}

; Check that any 'append'-type module flags are valid.
; CHECK: invalid value for 'append'-type module flag (expected a metadata node)
!16 = !{i32 5, !"flag-2", i32 56}
; CHECK: invalid value for 'append'-type module flag (expected a metadata node)
!17 = !{i32 5, !"flag-3", i32 57}
; CHECK-NOT: invalid value for 'append'-type module flag (expected a metadata node)
!18 = !{i32 5, !"flag-4", !{i32 57}}

; Check that any 'max' module flags are valid.
; CHECK: invalid value for 'max' module flag (expected constant integer)
!19 = !{i32 7, !"max", !"max"}

; Check that any 'min' module flags are valid.
; CHECK: invalid value for 'min' module flag (expected constant integer)
!20 = !{i32 8, !"min", !"min"}

; Check that any 'require' module flags are valid.
; CHECK: invalid requirement on flag, flag is not present in module
!11 = !{i32 3, !"bar", !{!"no-such-flag", i32 52}}
; CHECK: invalid requirement on flag, flag does not have the required value
!12 = !{i32 1, !"flag-0", i32 53}
!13 = !{i32 3, !"bar", !{!"flag-0", i32 54}}
; CHECK-NOT: invalid requirement on flag, flag is not present in module
; CHECK-NOT: invalid requirement on flag, flag does not have the required value
!14 = !{i32 1, !"flag-1", i32 55}
!15 = !{i32 3, !"bar", !{!"flag-1", i32 55}}

!llvm.module.flags = !{
  !0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15,
  !16, !17, !18, !19, !20 }
