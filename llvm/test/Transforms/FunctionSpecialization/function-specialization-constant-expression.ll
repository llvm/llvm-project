; Test function specialization wouldn't crash due to constant expression.
; Note that this test case shows that function specialization pass would
; transform the function even if no specialization happened.
; RUN: opt -function-specialization -S < %s | FileCheck %s

; CHECK: plus:
; CHECK-NEXT:  %{{.*}} = call i64 @func2(i64* getelementptr inbounds (%struct, %struct* @Global, i32 0, i32 3))
; CHECK: minus:
; CHECK-NEXT: %{{.*}} = call i64 @func2(i64* getelementptr inbounds (%struct, %struct* @Global, i32 0, i32 4))

%struct = type { i8, i16, i32, i64, i64}
@Global = internal constant %struct {i8 0, i16 1, i32 2, i64 3, i64 4}
define internal i64 @zoo(i1 %flag) {
entry:
  br i1 %flag, label %plus, label %minus

plus:
  %arg = getelementptr %struct, %struct* @Global, i32 0, i32 3
  %tmp0 = call i64 @func2(i64* %arg)
  br label %merge

minus:
  %arg2 = getelementptr %struct, %struct* @Global, i32 0, i32 4
  %tmp1 = call i64 @func2(i64* %arg2)
  br label %merge

merge:
  %tmp2 = phi i64 [ %tmp0, %plus ], [ %tmp1, %minus]
  ret i64 %tmp2
}

define internal i64 @func2(i64 *%x) {
entry:
  %val = ptrtoint i64* %x to i64
  ret i64 %val
}

define internal i64 @func(i64 *%x, i64 (i64*)* %binop) {
entry:
  %tmp0 = call i64 %binop(i64* %x)
  ret i64 %tmp0
}

define i64 @main() {
    %1 = call i64 @zoo(i1 0)
    %2 = call i64 @zoo(i1 1)
    %3 = add i64 %1, %2
    ret i64 %3
}