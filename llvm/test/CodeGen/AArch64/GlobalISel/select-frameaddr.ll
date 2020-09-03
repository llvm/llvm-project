; RUN: llc -mtriple=arm64-apple-ios -global-isel -o - %s | FileCheck %s

define i8* @rt0(i32 %x) nounwind readnone {
entry:
; CHECK-LABEL: rt0:
; CHECK: mov x0, x29
  %0 = tail call i8* @llvm.frameaddress(i32 0)
  ret i8* %0
}

define i8* @rt2() nounwind readnone {
entry:
; CHECK-LABEL: rt2:
; CHECK: ldr x[[reg:[0-9]+]], [x29]
; CHECK: ldr x0, [x[[reg]]]
  %0 = tail call i8* @llvm.frameaddress(i32 2)
  ret i8* %0
}

declare i8* @llvm.frameaddress(i32) nounwind readnone
