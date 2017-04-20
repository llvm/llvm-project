; RUN: opt < %s -analyze -scalar-evolution | FileCheck %s

declare void @z(i32)
declare void @z2(i64)

define void @fun(i1 %bool, i32 %x) {
entry:
        br label %body
body:
        %i = phi i32 [ 0, %entry ], [ %i.next, %body ]
        %bottom_zero = mul i32 %i, 2
        %a = or i32 %bottom_zero, 1
        call void @z(i32 %a)
        %bool_ext = zext i1 %bool to i32
        %b = or i32 %bool_ext, %bottom_zero
        call void @z(i32 %b)
        %shifted = lshr i32 %x, 31
        %c = or i32 %shifted, %bottom_zero
        call void @z(i32 %c)
        %i_ext = zext i32 %i to i64
        %d = or i64 %i_ext, 4294967296
        call void @z2(i64 %d)
        %i.next = add i32 %i, 1
        %cond = icmp eq i32 %i.next, 10
        br i1 %cond, label %exit, label %body
exit:
        ret void
}

; CHECK: %a = or i32 %bottom_zero, 1
; CHECK-NEXT: -->  {1,+,2}<%body>
; CHECK: %b = or i32 %bool_ext, %bottom_zero
; CHECK-NEXT: -->  {(zext i1 %bool to i32),+,2}
; CHECK: %c = or i32 %shifted, %bottom_zero
; CHECK-NEXT: -->  {(%x /u -2147483648),+,2}<%body>
; CHECK: %d = or i64 %i_ext, 4294967296
; CHECK-NEXT: -->  {4294967296,+,1}<nuw><nsw><%body>

