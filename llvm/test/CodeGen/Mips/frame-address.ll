; RUN: llc -march=mipsel < %s | FileCheck %s

declare i8* @llvm.frameaddress(i32) nounwind readnone

define i8* @f() nounwind {
entry:
  %0 = call i8* @llvm.frameaddress(i32 0)
  ret i8* %0

; CHECK:   addu    $fp, $sp, $zero
; CHECK:   addu    $2, $zero, $fp
}
