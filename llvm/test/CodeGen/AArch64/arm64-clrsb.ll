; RUN: llc < %s -mtriple=arm64-apple-ios7.0.0 |  FileCheck %s
; RUN: llc < %s -mtriple=arm64-apple-ios7.0.0 -O0 -pass-remarks-missed=gisel* -global-isel-abort=2 |  FileCheck %s --check-prefixes=GISEL,FALLBACK

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; Function Attrs: nounwind readnone
declare i32 @llvm.ctlz.i32(i32, i1) #0
declare i64 @llvm.ctlz.i64(i64, i1) #1

; Function Attrs: nounwind ssp
; FALLBACK-NOT: remark{{.*}}clrsb32
define i32 @clrsb32(i32 %x) #2 {
entry:
  %shr = ashr i32 %x, 31
  %xor = xor i32 %shr, %x
  %mul = shl i32 %xor, 1
  %add = or i32 %mul, 1
  %0 = tail call i32 @llvm.ctlz.i32(i32 %add, i1 false)

  ret i32 %0
; CHECK-LABEL: clrsb32
; CHECK:   cls [[TEMP:w[0-9]+]], [[TEMP]]

; FIXME: We should produce the same result here to save some code size. After
; that, we can remove the GISEL special casing.
; GISEL-LABEL: clrsb32
; GISEL: clz
}

; Function Attrs: nounwind ssp
; FALLBACK-NOT: remark{{.*}}clrsb64
define i64 @clrsb64(i64 %x) #3 {
entry:
  %shr = ashr i64 %x, 63
  %xor = xor i64 %shr, %x
  %mul = shl nsw i64 %xor, 1
  %add = or i64 %mul, 1
  %0 = tail call i64 @llvm.ctlz.i64(i64 %add, i1 false)

  ret i64 %0
; CHECK-LABEL: clrsb64
; CHECK:   cls [[TEMP:x[0-9]+]], [[TEMP]]
; GISEL-LABEL: clrsb64
; GISEL:   cls [[TEMP:x[0-9]+]], [[TEMP]]
}
