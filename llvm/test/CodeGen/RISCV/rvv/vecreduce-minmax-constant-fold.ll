; RUN: llc -mtriple=riscv64 -mattr=+v < %s | FileCheck %s

define i64 @test_const_smax() {
; CHECK-LABEL: test_const_smax:
; CHECK:       # %bb.0:
; CHECK-NEXT:    li a0, 127
; CHECK-NEXT:    ret
  %r = call i64 @llvm.vector.reduce.smax.v4i64(<4 x i64> <i64 -128, i64 -1, i64 0, i64 127>)
  ret i64 %r
}

define i64 @test_const_smin() {
; CHECK-LABEL: test_const_smin:
; CHECK:       # %bb.0:
; CHECK-NEXT:    li a0, -128
; CHECK-NEXT:    ret
  %r = call i64 @llvm.vector.reduce.smin.v4i64(<4 x i64> <i64 -128, i64 -1, i64 0, i64 127>)
  ret i64 %r
}

define i64 @test_const_umax() {
; CHECK-LABEL: test_const_umax:
; CHECK:       # %bb.0:
; CHECK-NEXT:    li a0, -1
; CHECK-NEXT:    ret
  %r = call i64 @llvm.vector.reduce.umax.v4i64(<4 x i64> <i64 -128, i64 -1, i64 0, i64 127>)
  ret i64 %r
}

define i64 @test_const_umin() {
; CHECK-LABEL: test_const_umin:
; CHECK:       # %bb.0:
; CHECK-NEXT:    li a0, 0
; CHECK-NEXT:    ret
  %r = call i64 @llvm.vector.reduce.umin.v4i64(<4 x i64> <i64 -128, i64 -1, i64 0, i64 127>)
  ret i64 %r
}

define i64 @test_const_smax_i8() {
; CHECK-LABEL: test_const_smax_i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    li a0, 127
; CHECK-NEXT:    ret
  %r = call i8 @llvm.vector.reduce.smax.v4i8(<4 x i8> <i8 -128, i8 -1, i8 0, i8 127>)
  %ext = sext i8 %r to i64
  ret i64 %ext
}

define i64 @test_const_smin_i8() {
; CHECK-LABEL: test_const_smin_i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    li a0, -128
; CHECK-NEXT:    ret
  %r = call i8 @llvm.vector.reduce.smin.v4i8(<4 x i8> <i8 -128, i8 -1, i8 0, i8 127>)
  %ext = sext i8 %r to i64
  ret i64 %ext
}

define i64 @test_const_umax_i8() {
; CHECK-LABEL: test_const_umax_i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    li a0, 255
; CHECK-NEXT:    ret
  %r = call i8 @llvm.vector.reduce.umax.v4i8(<4 x i8> <i8 -128, i8 -1, i8 0, i8 127>)
  %ext = zext i8 %r to i64
  ret i64 %ext
}

define i64 @test_const_umin_i8() {
; CHECK-LABEL: test_const_umin_i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    li a0, 0
; CHECK-NEXT:    ret
  %r = call i8 @llvm.vector.reduce.umin.v4i8(<4 x i8> <i8 -128, i8 -1, i8 0, i8 127>)
  %ext = zext i8 %r to i64
  ret i64 %ext
}

define i64 @test_nonconst(<4 x i64> %v) {
; CHECK-LABEL: test_nonconst:
; CHECK:         vredmax.vs
; CHECK:         ret
  %r = call i64 @llvm.vector.reduce.smax.v4i64(<4 x i64> %v)
  ret i64 %r
}

define i64 @test_poison() {
; CHECK-LABEL: test_poison:
; CHECK-NOT:     vredmax.vs
; CHECK:         ret
  %r = call i64 @llvm.vector.reduce.smax.v4i64(<4 x i64> <i64 -128, i64 -1, i64 poison, i64 127>)
  ret i64 %r
}
