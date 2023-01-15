; RUN: llc < %s -O0 -relocation-model=pic -mtriple=thumbv7-apple-ios | FileCheck %s
; <rdar://problem/12529625>

@foo = global i32 34879, align 4
@DWJumpTable2808 = global [2 x i32] [i32 sub (i32 ptrtoint (ptr blockaddress(@func, %14) to i32), i32 ptrtoint (ptr blockaddress(@func, %4) to i32)), i32 sub (i32 ptrtoint (ptr blockaddress(@func, %13) to i32), i32 ptrtoint (ptr blockaddress(@func, %4) to i32))]
@0 = internal constant [45 x i8] c"func XXXXXXXXXXX :: bb xxxxxxxxxxxxxxxxxxxx\0A\00"

; The indirect branch has the two destinations as successors. The lone PHI
; statement shouldn't be implicitly defined.

; CHECK-LABEL:      func:
; CHECK:      Ltmp1:    @ Block address taken
; CHECK-NOT:            @ implicit-def: R0
; CHECK:                @ 4-byte Reload

define i32 @func() nounwind ssp {
  %1 = alloca i32, align 4
  %2 = load i32, ptr @foo, align 4
  %3 = icmp eq i32 %2, 34879
  br label %4

; <label>:4                                       ; preds = %0
  %5 = zext i1 %3 to i32
  %6 = mul i32 %5, 287
  %7 = add i32 %6, 2
  %8 = getelementptr [2 x i32], ptr @DWJumpTable2808, i32 0, i32 %5
  %9 = load i32, ptr %8
  %10 = add i32 %9, ptrtoint (ptr blockaddress(@func, %4) to i32)
  %11 = inttoptr i32 %10 to ptr
  %12 = call i32 (ptr, ...) @printf(ptr @0)
  indirectbr ptr %11, [label %13, label %14]

; <label>:13                                      ; preds = %4
  %tmp14 = phi i32 [ %7, %4 ]
  store i32 23958, ptr @foo, align 4
  %tmp15 = load i32, ptr %1, align 4
  %tmp16 = icmp eq i32 %tmp15, 0
  %tmp17 = zext i1 %tmp16 to i32
  %tmp21 = add i32 %tmp17, %tmp14
  ret i32 %tmp21

; <label>:14                                      ; preds = %4
  ret i32 42
}

declare i32 @printf(ptr, ...)
