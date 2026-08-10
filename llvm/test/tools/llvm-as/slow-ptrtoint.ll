; RUN: llvm-as %s -o - | llvm-dis -o - | FileCheck %s

%0 = type { %1, %1, %1, %1, %1, %1, %1, %1 }
%1 = type { %2, %2, %2, %2, %2, %2, %2, %2 }
%2 = type { %3, %3, %3, %3, %3, %3, %3, %3 }
%3 = type { %4, %4, %4, %4, %4, %4, %4, %4 }
%4 = type { %5, %5, %5, %5, %5, %5, %5, %5 }
%5 = type { %6, %6, %6, %6, %6, %6, %6, %6 }
%6 = type { %7, %7, %7, %7, %7, %7, %7, %7 }
%7 = type { %8, %8, %8, %8, %8, %8, %8, %8 }
%8 = type { %9, %9, %9, %9, %9, %9, %9, %9 }
%9 = type { %10, %10, %10, %10, %10, %10, %10, %10 }
%10 = type { %11, %11, %11, %11, %11, %11, %11, %11 }
%11 = type { %12, %12, %12, %12, %12, %12, %12, %12 }
%12 = type { %13, %13, %13, %13, %13, %13, %13, %13 }
%13 = type { i32, i32 }

; it would take a naive recursive implementation ~4 days
; to constant fold the size of %0
define i64 @f_i64() {
; CHECK-LABEL: @f_i64
; CHECK:         ret i64 ptrtoint (ptr getelementptr (%0, ptr null, i32 1) to i64)
  ret i64 ptrtoint (ptr getelementptr (%0, ptr null, i32 1) to i64)
}

define i32 @f_i32() {
; CHECK-LABEL: @f_i32
; CHECK:         ret i32 ptrtoint (ptr getelementptr (%3, ptr null, i32 1) to i32)
  ret i32 ptrtoint (ptr getelementptr (%3, ptr null, i32 1) to i32)
}
