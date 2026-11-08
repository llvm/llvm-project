; RUN: split-file %s %t
; RUN: not llvm-as -disable-output %t/struct.ll 2>&1 | FileCheck %t/struct.ll
; RUN: not llvm-as -disable-output %t/non-byte-size.ll 2>&1 | FileCheck %t/non-byte-size.ll

;--- struct.ll
; CHECK: error: atomicrmw xchg operand must be an integer type, a floating-point type, a pointer type, or a fixed vector of any of these types
define void @f(ptr %ptr) {
  atomicrmw xchg ptr %ptr, { i32 } zeroinitializer seq_cst
  ret void
}

;--- non-byte-size.ll
; CHECK: atomic memory access' size must be byte-sized
define void @f(ptr %ptr, <4 x i1> %val) {
  atomicrmw xchg ptr %ptr, <4 x i1> %val seq_cst
  ret void
}
