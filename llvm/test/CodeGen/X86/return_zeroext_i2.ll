; RUN: llc -mtriple=i386-pc-win32 < %s | FileCheck %s 
; Check that the testcase does not crash
define zeroext i2 @crash () {
  ret i2 0
}
; CHECK: xorl	%eax, %eax
; CHECK-NEXT: retl
