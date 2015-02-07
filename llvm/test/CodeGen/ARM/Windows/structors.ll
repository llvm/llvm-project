; RUN: llc -mtriple thumbv7-windows-itanium -o - %s | FileCheck %s

@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @function, i8* null }]

define arm_aapcs_vfpcc void @function() {
entry:
  ret void
}

; CHECK: .section .CRT$XCU,"dr"
; CHECK: .long function

