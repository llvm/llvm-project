; RUN: opt -passes="ipsccp<func-spec>" -force-specialization -S < %s | FileCheck %s

; Test function specialization wouldn't crash due to constant expression.
; Note that this test case shows that function specialization pass would
; transform the function even if no specialization happened.

%struct = type { i8, i16, i32, i64, i64}
@Global = internal constant %struct {i8 0, i16 1, i32 2, i64 3, i64 4}

define internal i64 @func2(ptr %x) {
entry:
  %val = ptrtoint ptr %x to i64
  ret i64 %val
}

define internal i64 @func(ptr %x, ptr %binop) {
; CHECK-LABEL: @func(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    unreachable
;
entry:
  %tmp0 = call i64 %binop(ptr %x)
  ret i64 %tmp0
}

define internal i64 @zoo(i1 %flag) {
entry:
  br i1 %flag, label %plus, label %minus

plus:
  %arg = getelementptr %struct, ptr @Global, i32 0, i32 3
  %tmp0 = call i64 @func2(ptr %arg)
  br label %merge

minus:
  %arg2 = getelementptr %struct, ptr @Global, i32 0, i32 4
  %tmp1 = call i64 @func2(ptr %arg2)
  br label %merge

merge:
  %tmp2 = phi i64 [ %tmp0, %plus ], [ %tmp1, %minus]
  ret i64 %tmp2
}


define i64 @main() {
; CHECK-LABEL: @main(
; CHECK-NEXT:    [[TMP1:%.*]] = call i64 @zoo.4(i1 false)
; CHECK-NEXT:    [[TMP2:%.*]] = call i64 @zoo.3(i1 true)
; CHECK-NEXT:    ret i64 add (i64 ptrtoint (ptr getelementptr inbounds ([[STRUCT:%.*]], ptr @Global, i32 0, i32 4) to i64), i64 ptrtoint (ptr getelementptr inbounds ([[STRUCT]], ptr @Global, i32 0, i32 3) to i64))
;
  %1 = call i64 @zoo(i1 0)
  %2 = call i64 @zoo(i1 1)
  %3 = add i64 %1, %2
  ret i64 %3
}

; CHECK-LABEL: @func2.1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i64 undef

; CHECK-LABEL: @func2.2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i64 undef

; CHECK-LABEL: @zoo.3(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[PLUS:%.*]]
; CHECK:       plus:
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @func2.2(ptr getelementptr inbounds ([[STRUCT:%.*]], ptr @Global, i32 0, i32 3))
; CHECK-NEXT:  br label [[MERGE:%.*]]
; CHECK:       merge:
; CHECK-NEXT:    ret i64 undef

; CHECK-LABEL: @zoo.4(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[MINUS:%.*]]
; CHECK:       minus:
; CHECK-NEXT:    [[TMP1:%.*]] = call i64 @func2.1(ptr getelementptr inbounds ([[STRUCT:%.*]], ptr @Global, i32 0, i32 4))
; CHECK-NEXT:  br label [[MERGE:%.*]]
; CHECK:       merge:
; CHECK-NEXT:    ret i64 undef

