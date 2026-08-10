; RUN: llc < %s -mtriple=x86_64-- -mattr=+x87,-sse,-sse2 -global-isel -global-isel-abort=1 | FileCheck %s --check-prefix=X64
; RUN: llc < %s -mtriple=i686-- -mattr=+x87,-sse,-sse2 -global-isel -global-isel-abort=1 | FileCheck %s --check-prefix=X86

; fails with: llc: llvm-project/llvm/lib/Target/X86/X86FloatingPoint.cpp:326: unsigned int getFPReg(const MachineOperand &): Assertion `Reg >= X86::FP0 && Reg <= X86::FP6 && "Expected FP register!"' failed.
define i32 @store_and_fcmp_f32(ptr %p, ptr %q) nounwind {
  %v = load float, ptr %p
  store float %v, ptr %q
  %c = fcmp oeq float %v, 0.0
  %r = zext i1 %c to i32
  ret i32 %r
}
