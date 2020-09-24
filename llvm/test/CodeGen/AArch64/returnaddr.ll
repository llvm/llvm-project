; RUN: llc -o - %s -mtriple=arm64-apple-ios7.0 | FileCheck %s

define i8* @rt0(i32 %x) nounwind readnone {
entry:
; CHECK-LABEL: rt0:
; CHECK: hint #7
; CHECK: mov x0, x30
  %0 = tail call i8* @llvm.returnaddress(i32 0)
  ret i8* %0
}

define i8* @rt2() nounwind readnone {
entry:
; CHECK-LABEL: rt2:
; CHECK: ldr x[[reg:[0-9]+]], [x29]
; CHECK: ldr x[[reg]], [x[[reg]]]
; CHECK: ldr x30, [x[[reg]], #8]
; CHECK: hint #7
; CHECK: mov x0, x30
  %0 = tail call i8* @llvm.returnaddress(i32 2)
  ret i8* %0
}

declare i8* @llvm.returnaddress(i32) nounwind readnone
