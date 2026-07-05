; Function Attrs: nounwind readnone
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-unknown \
; RUN:   -mcpu=pwr9 < %s | FileCheck %s

define signext i32 @ctw(i32 signext %a) {
entry:
  %0 = tail call i32 @llvm.cttz.i32(i32 %a, i1 false)
  ret i32 %0
; CHECK-LABEL: ctw
; CHECK: cnttzw 3, 3
; CHECK-NEXT: blr
}

; Function Attrs: nounwind readnone
declare i32 @llvm.cttz.i32(i32, i1)

; Function Attrs: nounwind readnone
define signext i32 @clw(i32 signext %a) {
entry:
  %0 = tail call i32 @llvm.ctlz.i32(i32 %a, i1 false)
  ret i32 %0
; CHECK-LABEL: clw
; CHECK: cntlzw 3, 3
; CHECK-NEXT: blr
}

; Function Attrs: nounwind readnone
declare i32 @llvm.ctlz.i32(i32, i1)

; Function Attrs: nounwind readnone
define i64 @ctd(i64 %a) {
entry:
  %0 = tail call i64 @llvm.cttz.i64(i64 %a, i1 false)
  ret i64 %0
; CHECK-LABEL: ctd
; CHECK: cnttzd 3, 3
; CHECK-NEXT: blr
}

; Function Attrs: nounwind readnone
declare i64 @llvm.cttz.i64(i64, i1)

; Function Attrs: nounwind readnone
define i64 @cld(i64 %a) {
entry:
  %0 = tail call i64 @llvm.ctlz.i64(i64 %a, i1 false)
  ret i64 %0
; CHECK-LABEL: cld
; CHECK: cntlzd 3, 3
; CHECK-NEXT: blr
}

; Function Attrs: nounwind readnone
declare i64 @llvm.ctlz.i64(i64, i1)
