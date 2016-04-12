; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: 'allocsize' element size argument is out of bounds
declare i8* @a(i32) allocsize(1)

; CHECK: 'allocsize' element size argument must refer to an integer parameter
declare i8* @b(i32*) allocsize(0)

; CHECK: 'allocsize' number of elements argument is out of bounds
declare i8* @c(i32) allocsize(0, 1)

; CHECK: 'allocsize' number of elements argument must refer to an integer parameter
declare i8* @d(i32, i32*) allocsize(0, 1)

; CHECK: 'allocsize' number of elements argument is out of bounds
declare i8* @e(i32, i32) allocsize(1, 2)
