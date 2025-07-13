; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: huge 'byval' arguments are unsupported
define void @f(ptr byval([2147483648 x i16])) { ret void }
