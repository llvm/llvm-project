; RUN: llc -mtriple=hexagon < %s | FileCheck %s

target triple = "hexagon"

; Function Attrs: nounwind readnone
define i32 @f0(i32 %a0) #0 {
b0:
; CHECK: cl0
  %v0 = tail call i32 @llvm.ctlz.i32(i32 %a0, i1 true)
  ret i32 %v0
}

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.ctlz.i32(i32, i1) #1

; Function Attrs: nounwind readnone speculatable
declare i64 @llvm.ctlz.i64(i64, i1) #1

; Function Attrs: nounwind readnone
define i32 @f1(i32 %a0) #0 {
b0:
; CHECK: ct0
  %v0 = tail call i32 @llvm.cttz.i32(i32 %a0, i1 true)
  ret i32 %v0
}

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.cttz.i32(i32, i1) #1

; Function Attrs: nounwind readnone speculatable
declare i64 @llvm.cttz.i64(i64, i1) #1

; Function Attrs: nounwind readnone
define i32 @f2(i64 %a0) #0 {
b0:
; CHECK: cl0
  %v0 = tail call i64 @llvm.ctlz.i64(i64 %a0, i1 true)
  %v1 = trunc i64 %v0 to i32
  ret i32 %v1
}

; Function Attrs: nounwind readnone
define i32 @f3(i64 %a0) #0 {
b0:
; CHECK: ct0
  %v0 = tail call i64 @llvm.cttz.i64(i64 %a0, i1 true)
  %v1 = trunc i64 %v0 to i32
  ret i32 %v1
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind readnone speculatable }
