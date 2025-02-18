; RUN: llc -mtriple=mipsel -verify-machineinstrs < %s | FileCheck %s

define ptr @f1() nounwind {
entry:
  %0 = call ptr @llvm.returnaddress(i32 0)
  ret ptr %0

; CHECK:    move  $2, $ra
}

define ptr @f2() nounwind {
entry:
  call void @g()
  %0 = call ptr @llvm.returnaddress(i32 0)
  ret ptr %0

; CHECK:    move  $[[R0:[0-9]+]], $ra
; CHECK:    jal
; CHECK:    move  $2, $[[R0]]
}

declare ptr @llvm.returnaddress(i32) nounwind readnone
declare void @g()
