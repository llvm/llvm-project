;; Verify that @LPStart is omitted when there are no landing pads. This test
;; uses an unkown personality to force emitting the exception table.

; RUN: llc -basic-block-sections=all -mtriple=x86_64 < %s | FileCheck %s

declare void @throwit()
declare i32 @__unknown_ehpersonality(...)

define void @foo(i1 %cond) uwtable personality ptr @__unknown_ehpersonality {
entry:
  br i1 %cond, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  call void @throwit()
  unreachable

cond.false:                                         ; preds = %entry
  ret void
}

; CHECK:      GCC_except_table0:
; CHECK-NEXT: .Lexception0:
; CHECK-NEXT:   .byte	255                             # @LPStart Encoding = omit
; CHECK-NEXT:   .byte	255                             # @TType Encoding = omit
; CHECK-NEXT:   .byte	1                               # Call site Encoding = uleb128
; CHECK-NEXT:   .uleb128 .Laction_table_base0-.Lcst_begin0
; CHECK-NEXT: .Lcst_begin0:
; CHECK-NEXT: .Lexception1:
; CHECK-NEXT:   .byte	255                             # @LPStart Encoding = omit
; CHECK-NEXT:   .byte	255                             # @TType Encoding = omit
; CHECK-NEXT:   .byte	1                               # Call site Encoding = uleb128
; CHECK-NEXT:   .uleb128 .Laction_table_base0-.Lcst_begin1
; CHECK-NEXT: .Lcst_begin1:
; CHECK-NEXT: .Lexception2:
; CHECK-NEXT:   .byte	255                             # @LPStart Encoding = omit
; CHECK-NEXT:   .byte	255                             # @TType Encoding = omit
; CHECK-NEXT:   .byte	1                               # Call site Encoding = uleb128
; CHECK-NEXT:   .uleb128 .Laction_table_base0-.Lcst_begin2
; CHECK-NEXT: .Lcst_begin2:
; CHECK-NEXT:   .uleb128 foo.__part.2-foo.__part.2      # >> Call Site 1 <<
; CHECK-NEXT:   .uleb128 .LBB_END0_2-foo.__part.2       #   Call between foo.__part.2 and .LBB_END0_2
; CHECK-NEXT:   .byte	0                               #     has no landing pad
; CHECK-NEXT:   .byte	0                               #   On action: cleanup
; CHECK-NEXT: .Laction_table_base0:
