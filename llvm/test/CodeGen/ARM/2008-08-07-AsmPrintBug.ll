; RUN: llc -mtriple arm-apple-darwin -mattr=+v6 -relocation-model pic -filetype asm -o - %s | FileCheck %s

%struct.FILE = type { ptr, i32, i32, i16, i16, %struct.__sbuf, i32, ptr, ptr, ptr, ptr, ptr, %struct.__sbuf, ptr, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
%struct.__gcov_var = type { ptr, i32, i32, i32, i32, i32, i32, [1025 x i32] }
%struct.__sFILEX = type opaque
%struct.__sbuf = type { ptr, i32 }
@__gcov_var = common global %struct.__gcov_var zeroinitializer

define i32 @__gcov_close() nounwind {
entry:
  load i32, ptr getelementptr (%struct.__gcov_var, ptr @__gcov_var, i32 0, i32 5), align 4
  ret i32 %0
}

; CHECK: comm

