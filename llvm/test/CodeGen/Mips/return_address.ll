; RUN: llc -march=mipsel < %s | FileCheck %s

define i8* @f1() nounwind {
entry:
  %0 = call i8* @llvm.returnaddress(i32 0)
  ret i8* %0

; CHECK:    addu    $2, $zero, $ra
}

define i8* @f2() nounwind {
entry:
  call void @g()
  %0 = call i8* @llvm.returnaddress(i32 0)
  ret i8* %0

; CHECK:    addu    $[[R0:[0-9]+]], $zero, $ra
; CHECK:    jal
; CHECK:    addu    $2,  $zero, $[[R0]]
}

declare i8* @llvm.returnaddress(i32) nounwind readnone
declare void @g()
