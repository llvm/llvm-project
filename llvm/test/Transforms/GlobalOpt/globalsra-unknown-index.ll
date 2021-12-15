; RUN: opt < %s -passes=globalopt -S | FileCheck %s

; globalopt should not sra the global, because it can't see the index.

%struct.X = type { [3 x i32], [3 x i32] }

; CHECK: @Y = internal unnamed_addr global [3 x %struct.X] zeroinitializer
@Y = internal global [3 x %struct.X] zeroinitializer

@addr = external global i8

define void @frob() {
  store i32 1, i32* getelementptr inbounds ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 ptrtoint (i8* @addr to i64)), align 4
  ret void
}

; CHECK-LABEL: @borf
; CHECK: %a = load
; CHECK: %b = load
; CHECK: add i32 %a, %b
define i32 @borf(i64 %i, i64 %j) {
  %p = getelementptr inbounds [3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 0
  %a = load i32, i32* %p
  %q = getelementptr inbounds [3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 1, i64 0
  %b = load i32, i32* %q
  %c = add i32 %a, %b
  ret i32 %c
}

; CHECK-LABEL: @borg
; CHECK: %a = load
; CHECK: %b = load
; CHECK: add i32 %a, %b
define i32 @borg(i64 %i, i64 %j) {
  %p = getelementptr inbounds [3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 1, i32 0, i64 1
  %a = load i32, i32* %p
  %q = getelementptr inbounds [3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 1, i32 1, i64 1
  %b = load i32, i32* %q
  %c = add i32 %a, %b
  ret i32 %c
}

; CHECK-LABEL: @borh
; CHECK: %a = load
; CHECK: %b = load
; CHECK: add i32 %a, %b
define i32 @borh(i64 %i, i64 %j) {
  %p = getelementptr inbounds [3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 2, i32 0, i64 2
  %a = load i32, i32* %p
  %q = getelementptr inbounds [3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 2, i32 1, i64 2
  %b = load i32, i32* %q
  %c = add i32 %a, %b
  ret i32 %c
}
