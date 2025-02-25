; RUN: mlir-translate -import-llvm %s -split-input-file | FileCheck %s

declare i64 @llvm.aarch64.ldxr.p0(ptr)

define dso_local void @t0(ptr %a) {
  %x = call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i8) %a)
  ret void
}

; CHECK-LABEL: llvm.func @llvm.aarch64.ldxr.p0(!llvm.ptr)
; CHECK-LABEL: llvm.func @t0
; CHECK:   llvm.call_intrinsic "llvm.aarch64.ldxr.p0"({{.*}}) : (!llvm.ptr) -> i64
; CHECK:   llvm.return
; CHECK: }

; -----

declare <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8>, <8 x i8>)

define dso_local <8 x i8> @t1(<8 x i8> %lhs, <8 x i8> %rhs) {
  %r = call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> %lhs, <8 x i8> %rhs)
  ret <8 x i8> %r
}

; CHECK: llvm.func @t1(%[[A0:.*]]: vector<8xi8>, %[[A1:.*]]: vector<8xi8>) -> vector<8xi8> {{.*}} {
; CHECK:   %[[R:.*]] = llvm.call_intrinsic "llvm.aarch64.neon.uabd.v8i8"(%[[A0]], %[[A1]]) : (vector<8xi8>, vector<8xi8>) -> vector<8xi8>
; CHECK:   llvm.return %[[R]] : vector<8xi8>
; CHECK: }

; -----

declare void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8>, <8 x i8>, ptr)

define dso_local void @t2(<8 x i8> %lhs, <8 x i8> %rhs, ptr %a) {
  call void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8> %lhs, <8 x i8> %rhs, ptr %a)
  ret void
}

; CHECK: llvm.func @t2(%[[A0:.*]]: vector<8xi8>, %[[A1:.*]]: vector<8xi8>, %[[A2:.*]]: !llvm.ptr) {{.*}} {
; CHECK:   llvm.call_intrinsic "llvm.aarch64.neon.st2.v8i8.p0"(%[[A0]], %[[A1]], %[[A2]]) : (vector<8xi8>, vector<8xi8>, !llvm.ptr) -> !llvm.void
; CHECK:   llvm.return
; CHECK: }

; -----

declare void @llvm.gcroot(ptr %arg1, ptr %arg2)
define void @gctest() gc "example" {
  %arg1 = alloca ptr
  call void @llvm.gcroot(ptr %arg1, ptr null)
  ret void
}

; CHECK-LABEL: @gctest
; CHECK: llvm.call_intrinsic "llvm.gcroot"({{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr) -> !llvm.void

; -----

; Test we get the supported version, not the unregistered one.

declare i32 @llvm.lround.i32.f32(float)

; CHECK-LABEL: llvm.func @lround_test
define void @lround_test(float %0, double %1) {
  ; CHECK-NOT: llvm.call_intrinsic "llvm.lround
  ; CHECK: llvm.intr.lround(%{{.*}}) : (f32) -> i32
  %3 = call i32 @llvm.lround.i32.f32(float %0)
  ret void
}
