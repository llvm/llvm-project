; RUN: opt < %s -passes=instcombine -S | FileCheck %s

@.str = internal constant [5 x i8] c"foo\0A\00"
@.str1 = internal constant [5 x i8] c"bar\0A\00"

define i32 @main() nounwind  {
entry:
  %x = call i32 @func_11() nounwind
  %tmp3 = or i32 %x, -5
  %tmp5 = urem i32 251, %tmp3
  %tmp6 = icmp ne i32 %tmp5, 0
  %tmp67 = zext i1 %tmp6 to i32
  %tmp9 = urem i32 %tmp67, 95
  %tmp10 = and i32 %tmp9, 1
  %tmp12 = icmp eq i32 %tmp10, 0
  br i1 %tmp12, label %bb14, label %bb

bb:
  br label %bb15

bb14:
  br label %bb15

bb15:
  %iftmp.0.0 = phi i8* [ getelementptr ([5 x i8], [5 x i8]* @.str1, i32 0, i32 0), %bb14 ], [ getelementptr ([5 x i8], [5 x i8]* @.str, i32 0, i32 0), %bb ]
  %tmp17 = call i32 (i8*, ...) @printf(i8* %iftmp.0.0) nounwind
  ret i32 0
}

; CHECK-LABEL: define i32 @main(
; CHECK: call i32 @func_11()
; CHECK-NEXT: br i1 false, label %bb14, label %bb

declare i32 @func_11()

declare i32 @printf(i8*, ...) nounwind
