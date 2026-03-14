; RUN: llc < %s -mtriple=aarch64 -mattr=+mte | FileCheck %s

define void @stgp0(i64 %a, i64 %b, ptr %p) {
entry:
; CHECK-LABEL: stgp0:
; CHECK: stgp x0, x1, [x2]
; CHECK: ret
  call void @llvm.aarch64.stgp(ptr %p, i64 %a, i64 %b)
  ret void
}

define void @stgp1004(i64 %a, i64 %b, ptr %p) {
entry:
; CHECK-LABEL: stgp1004:
; CHECK: add [[R:x[0-9]+]], x2, #1004
; CHECK: stgp x0, x1, [[[R]]]
; CHECK: ret
  %q = getelementptr i8, ptr %p, i32 1004
  call void @llvm.aarch64.stgp(ptr %q, i64 %a, i64 %b)
  ret void
}

define void @stgp1008(i64 %a, i64 %b, ptr %p) {
entry:
; CHECK-LABEL: stgp1008:
; CHECK: stgp x0, x1, [x2, #1008]
; CHECK: ret
  %q = getelementptr i8, ptr %p, i32 1008
  call void @llvm.aarch64.stgp(ptr %q, i64 %a, i64 %b)
  ret void
}

define void @stgp1024(i64 %a, i64 %b, ptr %p) {
entry:
; CHECK-LABEL: stgp1024:
; CHECK: add [[R:x[0-9]+]], x2, #1024
; CHECK: stgp x0, x1, [[[R]]]
; CHECK: ret
  %q = getelementptr i8, ptr %p, i32 1024
  call void @llvm.aarch64.stgp(ptr %q, i64 %a, i64 %b)
  ret void
}

define void @stgp_1024(i64 %a, i64 %b, ptr %p) {
entry:
; CHECK-LABEL: stgp_1024:
; CHECK: stgp x0, x1, [x2, #-1024]
; CHECK: ret
  %q = getelementptr i8, ptr %p, i32 -1024
  call void @llvm.aarch64.stgp(ptr %q, i64 %a, i64 %b)
  ret void
}

define void @stgp_1040(i64 %a, i64 %b, ptr %p) {
entry:
; CHECK-LABEL: stgp_1040:
; CHECK: sub [[R:x[0-9]+]], x2, #1040
; CHECK: stgp x0, x1, [x{{.*}}]
; CHECK: ret
  %q = getelementptr i8, ptr %p, i32 -1040
  call void @llvm.aarch64.stgp(ptr %q, i64 %a, i64 %b)
  ret void
}

define void @stgp_alloca(i64 %a, i64 %b) {
entry:
; CHECK-LABEL: stgp_alloca:
; CHECK: stgp x0, x1, [sp, #-32]!
; CHECK: stgp x1, x0, [sp, #16]
; CHECK: ret
  %x = alloca i8, i32 32, align 16
  call void @llvm.aarch64.stgp(ptr %x, i64 %a, i64 %b)
  %x1 = getelementptr i8, ptr %x, i32 16
  call void @llvm.aarch64.stgp(ptr %x1, i64 %b, i64 %a)
  ret void
}

declare void @llvm.aarch64.stgp(ptr %p, i64 %a, i64 %b)
