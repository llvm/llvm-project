; RUN: llc -mtriple=arm64ec-pc-windows-msvc < %s | FileCheck %s

define void @f1(ptr %p, i64 %n) {
; CHECK-LABEL: "#f1":
; CHECK: bl "#memset"
  call void @llvm.memset.p0.i64(ptr %p, i8 0, i64 %n, i1 false)
  ret void
}

define void @f2(ptr %p1, ptr %p2, i64 %n) {
; CHECK-LABEL: "#f2":
; CHECK: bl "#memcpy"
  call void @llvm.memcpy.p0.i64(ptr %p1, ptr %p2, i64 %n, i1 false)
  ret void
}

define double @f3(double %x, double %y) {
; CHECK-LABEL: "#f3":
; CHECK: b "#fmod"
  %r = frem double %x, %y
  ret double %r
}


define i128 @f4(i128 %x, i128 %y) {
; CHECK-LABEL: "#f4":
; CHECK: bl "#__divti3"
  %r = sdiv i128 %x, %y
  ret i128 %r
}

