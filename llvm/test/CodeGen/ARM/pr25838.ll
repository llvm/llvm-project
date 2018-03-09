; RUN: llc -verify-machineinstrs < %s
; PR25838

target triple = "armv7--linux-android"

%0 = type { i32, i32 }

define i32 @foo(%0* readonly) {
  br i1 undef, label %12, label %2

; <label>:2
  %3 = trunc i64 undef to i32
  %4 = icmp eq i32 undef, 0
  br i1 %4, label %5, label %9

; <label>:5
  %6 = icmp slt i32 %3, 0
  %7 = sub nsw i32 0, %3
  %8 = select i1 %6, i32 %7, i32 %3
  br label %12

; <label>:9
  br i1 undef, label %12, label %10

; <label>:10
  %11 = tail call i32 @bar(i32 undef)
  unreachable

; <label>:12
  %13 = phi i32 [ %8, %5 ], [ 0, %1 ], [ undef, %9 ]
  ret i32 %13
}

declare i32 @bar(i32)
