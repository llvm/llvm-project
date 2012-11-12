; RUN: opt < %s -globalopt -S > %t
; RUN: grep "@Y = internal unnamed_addr global \[3 x [%]struct.X\] zeroinitializer" %t
; RUN: grep load %t | count 6
; RUN: grep "add i32 [%]a, [%]b" %t | count 3

; globalopt should not sra the global, because it can't see the index.

%struct.X = type { [3 x i32], [3 x i32] }

@Y = internal global [3 x %struct.X] zeroinitializer

@addr = external global i8

define void @frob() {
  store i32 1, i32* getelementptr inbounds ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 ptrtoint (i8* @addr to i64)), align 4
  ret void
}
define i32 @borf(i64 %i, i64 %j) {
  %p = getelementptr inbounds [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 0
  %a = load i32* %p
  %q = getelementptr inbounds [3 x %struct.X]* @Y, i64 0, i64 0, i32 1, i64 0
  %b = load i32* %q
  %c = add i32 %a, %b
  ret i32 %c
}
define i32 @borg(i64 %i, i64 %j) {
  %p = getelementptr inbounds [3 x %struct.X]* @Y, i64 0, i64 1, i32 0, i64 1
  %a = load i32* %p
  %q = getelementptr inbounds [3 x %struct.X]* @Y, i64 0, i64 1, i32 1, i64 1
  %b = load i32* %q
  %c = add i32 %a, %b
  ret i32 %c
}
define i32 @borh(i64 %i, i64 %j) {
  %p = getelementptr inbounds [3 x %struct.X]* @Y, i64 0, i64 2, i32 0, i64 2
  %a = load i32* %p
  %q = getelementptr inbounds [3 x %struct.X]* @Y, i64 0, i64 2, i32 1, i64 2
  %b = load i32* %q
  %c = add i32 %a, %b
  ret i32 %c
}
