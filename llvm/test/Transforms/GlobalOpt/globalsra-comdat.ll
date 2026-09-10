; RUN: opt < %s -S -passes=globalopt | FileCheck %s

$x = comdat any
@x = internal global [2 x i32] zeroinitializer, comdat, align 4
; CHECK: @x = internal unnamed_addr global i1 false, comdat
; CHECK: @x.0 = internal unnamed_addr global i32 0, comdat($x), align 4
; CHECK: @x.1 = internal unnamed_addr global i32 0, comdat($x), align 4

define dso_local i32 @f() {
  %1 = load i32, ptr @x, align 4
  %2 = add nsw i32 %1, 1
  store i32 %2, ptr @x, align 4
  ret i32 %2
}

define dso_local i32 @f2() {
  %1 = load i32, ptr getelementptr inbounds ([2 x i32], ptr @x, i64 0, i64 1), align 4
  %2 = add nsw i32 %1, 1
  store i32 %2, ptr getelementptr inbounds ([2 x i32], ptr @x, i64 0, i64 1), align 4
  ret i32 %2
}