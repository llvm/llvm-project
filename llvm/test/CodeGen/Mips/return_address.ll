; RUN: llc -march=mipsel -verify-machineinstrs < %s | FileCheck %s

define i8* @f1() nounwind {
entry:
  %0 = call i8* @llvm.returnaddress(i32 0)
  ret i8* %0

; CHECK:    move  $2, $ra
}

define i8* @f2() nounwind {
entry:
  call void @g()
  %0 = call i8* @llvm.returnaddress(i32 0)
  ret i8* %0

; CHECK:    move  $[[R0:[0-9]+]], $ra
; CHECK:    jal
; CHECK:    move  $2, $[[R0]]
}

declare i8* @llvm.returnaddress(i32) nounwind readnone
declare void @g()
