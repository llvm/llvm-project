; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

target datalayout = "p0:64:64-p1:32:32"


@absolute_empty_arguments = external global i32, !absolute_symbol !0

@absolute_one_argument = external global i32, !absolute_symbol !1

@absolute_three_arguments = external global i32, !absolute_symbol !2



@absolute_one_argument_wrong_width = external global i32, !absolute_symbol !3
@absolute_two_arguments_wrong_width = external global i32, !absolute_symbol !4

@absolute_two_arguments_one_wrong_width0 = external global i32, !absolute_symbol !5
@absolute_two_arguments_one_wrong_width1 = external global i32, !absolute_symbol !6

@absolute_zero_zero = external global i32, !absolute_symbol !7

@absolute_equal_other = external global i32, !absolute_symbol !8

@absolute_wrong_width_non0_as = external addrspace(1) global i32, !absolute_symbol !9

; Test other kinds of symbols besides GlobalVariable
define void @absolute_func_empty_arguments() !absolute_symbol !0 {
  ret void
}

@absolute_is_fp = external global i32, !absolute_symbol !10
@absolute_is_vector = external global i32, !absolute_symbol !11
@absolute_is_ptr = external global i32, !absolute_symbol !12
@absolute_is_ptr0 = external global i32, !absolute_symbol !13
@absolute_is_ptr1 = external global i32, !absolute_symbol !14

@absolute_wrong_order = external global i32, !absolute_symbol !15

; CHECK: It should have at least one range!
; CHECK-NEXT: !0 = !{}
; CHECK: It should have at least one range!
; CHECK-NEXT: !0 = !{}
!0 = !{}

; CHECK-NEXT: Unfinished range!
; CHECK-NEXT: !1 = !{i64 128}
!1 = !{i64 128}

; CHECK-NEXT: Unfinished range!
; CHECK-NEXT: !2 = !{i64 128, i64 256, i64 512}
!2 = !{i64 128, i64 256, i64 512}

; CHECK-NEXT: Unfinished range!
; CHECK-NEXT: !3 = !{i32 256}
!3 = !{i32 256}

; CHECK-NEXT: Range types must match instruction type!
; CHECK-NEXT: ptr @absolute_two_arguments_wrong_width
!4 = !{i32 256, i32 512}

; CHECK-NEXT: Range types must match instruction type!
; CHECK-NEXT: ptr @absolute_two_arguments_one_wrong_width0
!5 = !{i32 256, i64 512}

; CHECK-NEXT: Range types must match instruction type!
; CHECK-NEXT: ptr @absolute_two_arguments_one_wrong_width1
!6 = !{i64 256, i32 512}

; CHECK-NEXT: Range must not be empty!
; CHECK-NEXT: !7 = !{i64 0, i64 0}
!7 = !{i64 0, i64 0}

; CHECK-NEXT: The upper and lower limits cannot be the same value
; CHECK-NEXT: ptr @absolute_equal_other
!8 = !{i64 123, i64 123}

; CHECK-NEXT: Range types must match instruction type!
; CHECK-NEXT: ptr addrspace(1) @absolute_wrong_width_non0_as
!9 = !{i64 512, i64 256}

; CHECK-NEXT: The lower limit must be an integer!
!10 = !{float 0.0, float 256.0}

; CHECK-NEXT: The lower limit must be an integer!
!11 = !{<2 x i64> zeroinitializer, <2 x i64> <i64 256, i64 256>}

; CHECK-NEXT: The lower limit must be an integer!
!12 = !{ptr null, ptr inttoptr (i64 256 to ptr)}

; CHECK-NEXT: The lower limit must be an integer!
!13 = !{ptr null, i64 456}

; CHECK-NEXT: The upper limit must be an integer!
!14 = !{i64 456, ptr inttoptr (i64 512 to ptr)}
!15 = !{i64 1024, i64 128}

