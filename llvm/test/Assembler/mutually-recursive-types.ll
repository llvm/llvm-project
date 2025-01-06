; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: error: identified structure type 'rt3' is recursive

%rt1 = type { i32, { i8, %rt2, i8 }, i32 }
%rt2 = type { i64, { i6, %rt3 } }
%rt3 = type { %rt1 }
