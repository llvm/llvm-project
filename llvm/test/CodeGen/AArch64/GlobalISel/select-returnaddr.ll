; RUN: llc -mtriple=arm64-apple-ios -global-isel -o - %s | FileCheck %s

define i8* @rt0(i32 %x) nounwind readnone {
entry:
; CHECK-LABEL: rt0:
; CHECK:       hint #7
; CHECK-NEXT:  mov x0, x30
  %0 = tail call i8* @llvm.returnaddress(i32 0)
  ret i8* %0
}

define i8* @rt0_call_clobber(i32 %x) nounwind readnone {
entry:
; CHECK-LABEL: rt0_call_clobber:
; CHECK:       stp x20, x19, [sp, #-32]!
; CHECK:       stp x29, x30, [sp, #16]
; CHECK:       mov x19, x30
; CHECK:       bl _foo
; CHECK:       mov x30, x19
; CHECK-NEXT:  hint #7
; CHECK-NEXT:  mov x0, x30
; CHECK-NOT:   x0
; CHECK:       ret
  %ret = call i32 @foo()
  %0 = tail call i8* @llvm.returnaddress(i32 0)
  ret i8* %0
}

define i8* @rt2() nounwind readnone {
entry:
; CHECK-LABEL: rt2:
; CHECK:       ldr x[[reg:[0-9]+]], [x29]
; CHECK:       ldr x[[reg]], [x[[reg]]]
; CHECK:       ldr x30, [x[[reg]], #8]
; CHECK:       hint #7
; CHECK:       mov x0, x30
; CHECK-NOT:   x0
; CHECK:       ret
  %0 = tail call i8* @llvm.returnaddress(i32 2)
  ret i8* %0
}


declare i32 @foo()
declare i8* @llvm.returnaddress(i32) nounwind readnone
