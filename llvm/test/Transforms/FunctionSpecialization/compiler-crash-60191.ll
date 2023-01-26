; RUN: opt -S --passes="ipsccp<func-spec>" -force-function-specialization < %s | FileCheck %s

@A = private constant [6 x i32] [i32 1, i32 2, i32 0, i32 0, i32 0, i32 0], align 16
@B = external global ptr, align 8

define i32 @caller() {
entry:
  %c1 = call fastcc i32 @func(ptr @f0, i32 0, ptr null)
  %c2 = call fastcc i32 @func(ptr @f1, i32 1, ptr @A)
  %c3 = call fastcc i32 @func(ptr @f2, i32 2, ptr @A)
  %add = add i32 %c1, %c2
  %sub = sub i32 %add, %c3
  ret i32 %sub
}

define internal fastcc i32 @func(ptr %f, i32 %N, ptr %A) {
entry:
  switch i32 %N, label %sw.epilog [
    i32 2, label %sw.bb
    i32 1, label %sw.bb2
    i32 0, label %sw.bb4
  ]

sw.bb:                                            ; preds = %entry
  %0 = getelementptr inbounds i32, ptr %A, i64 1
  %1 = load i32, ptr %0, align 4
  %2 = call i32 %f(i32 %1)
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  %3 = load i32, ptr %A, align 4
  %4 = zext i32 %3 to i64
  %5 = call i32 %f(i64 %4)
  br label %sw.epilog

sw.bb4:                                           ; preds = %entry
  %6 = call i32 %f()
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.bb, %sw.bb2, %sw.bb4, %entry
  %7 = phi i32 [undef, %entry], [%2, %sw.bb], [%5, %sw.bb2], [%6, %sw.bb4]
  ret i32 %7
}

define i32 @f0() {
  %ld = load i32, ptr @B, align 4
  ret i32 %ld
}

define i32 @f1(i64 %offset) {
  %gep = getelementptr inbounds i32, ptr @B, i64 %offset
  %ld = load i32, ptr %gep, align 4
  ret i32 %ld
}

define i32 @f2(i32 %offset) {
  %zext = zext i32 %offset to i64 
  %call = call i32 @f1(i64 %zext)
  ret i32 %call
}

; Tests that `func` has been specialized and it didn't cause compiler crash.
; CHECK-DAG: func.1
; CHECK-DAG: func.2
; CHECK-DAG: func.3

