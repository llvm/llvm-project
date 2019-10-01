; RUN: not llc -march=mips < %s 2>&1 | FileCheck %s

declare i8* @llvm.frameaddress(i32) nounwind readnone

define i8* @f() nounwind {
entry:
  %0 = call i8* @llvm.frameaddress(i32 1)
  ret i8* %0

; CHECK: error: return address can be determined only for current frame
}
