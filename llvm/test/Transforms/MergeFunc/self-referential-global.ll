; RUN: opt -passes=mergefunc -disable-output < %s

; A linked list type and simple payload
%LL = type { %S, %LL* }
%S = type { void (%S*, i32)* }

; Table refers to itself via GEP
@Table = internal global [3 x %LL] [%LL { %S { void (%S*, i32)* @B }, %LL* getelementptr inbounds ([3 x %LL], [3 x %LL]* @Table, i32 0, i32 0) }, %LL { %S { void (%S*, i32)* @A }, %LL* getelementptr inbounds ([3 x %LL], [3 x %LL]* @Table, i32 0, i32 0) }, %LL { %S { void (%S*, i32)* @A }, %LL* getelementptr inbounds ([3 x %LL], [3 x %LL]* @Table, i32 0, i32 0) }], align 16

; The body of this is irrelevant; it is long so that mergefunc doesn't skip it as a small function.
define internal void @A(%S* %self, i32 %a) {
  %1 = add i32 %a, 32
  %2 = add i32 %1, 32
  %3 = add i32 %2, 32
  %4 = add i32 %3, 32
  %5 = add i32 %4, 32
  %6 = add i32 %5, 32
  %7 = add i32 %6, 32
  %8 = add i32 %7, 32
  %9 = add i32 %8, 32
  %10 = add i32 %9, 32
  %11 = add i32 %10, 32
  ret void
}

define internal void @B(%S* %self, i32 %a) {
  %1 = add i32 %a, 32
  %2 = add i32 %1, 32
  %3 = add i32 %2, 32
  %4 = add i32 %3, 32
  %5 = add i32 %4, 32
  %6 = add i32 %5, 32
  %7 = add i32 %6, 32
  %8 = add i32 %7, 32
  %9 = add i32 %8, 32
  %10 = add i32 %9, 32
  %11 = add i32 %10, 32
  ret void
}

