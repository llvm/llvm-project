; RUN: llc -verify-machineinstrs < %s -march=msp430

define i16 @foo() {
entry:
  %0 = call i32 @llvm.get.rounding()
  %1 = trunc i32 %0 to i16
  ret i16 %1
}

declare i32 @llvm.get.rounding() nounwind
