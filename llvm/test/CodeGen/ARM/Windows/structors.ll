; RUN: llc -mtriple thumbv7-windows-itanium -o - %s \
; RUN:   | FileCheck %s -check-prefix CHECK-WIN

; RUN: llc -mtriple thumbv7-windows-gnu -o - %s \
; RUN:   | FileCheck %s -check-prefix CHECK-GNU

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @function, ptr null }]

define arm_aapcs_vfpcc void @function() {
entry:
  ret void
}

; CHECK-WIN: .section .CRT$XCU,"dr"
; CHECK-WIN: .long function

; CHECK-GNU: .section .ctors,"dw"
; CHECK-GNU: .long function
