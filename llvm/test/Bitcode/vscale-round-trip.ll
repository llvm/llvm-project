; RUN: llvm-as < %s | llvm-dis | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

@important_val = extern_weak dso_local global i32, align 4

; CHECK-LABEL: define <vscale x 4 x i32> @const_shufflevector(
; CHECK: <vscale x 4 x i32> zeroinitializer

define <vscale x 4 x i32> @const_shufflevector() {
  ret <vscale x 4 x i32> shufflevector (<vscale x 4 x i32> zeroinitializer,
                                        <vscale x 4 x i32> undef,
                                        <vscale x 4 x i32> zeroinitializer)
}

; CHECK-LABEL: define <vscale x 4 x i32> @const_shufflevector_ex()
; CHECK: <vscale x 4 x i32> zeroinitializer

define <vscale x 4 x i32> @const_shufflevector_ex() {
  ret <vscale x 4 x i32> shufflevector (<vscale x 2 x i32> zeroinitializer,
                                        <vscale x 2 x i32> undef,
                                        <vscale x 4 x i32> zeroinitializer)
}

; CHECK-LABEL: define <vscale x 4 x i32> @non_const_shufflevector(
; CHECK: %res = shufflevector <vscale x 4 x i32>

define <vscale x 4 x i32> @non_const_shufflevector(<vscale x 4 x i32> %lhs,
                                                   <vscale x 4 x i32> %rhs) {
  %res = shufflevector <vscale x 4 x i32> %lhs,
                       <vscale x 4 x i32> %rhs,
                       <vscale x 4 x i32> zeroinitializer

  ret <vscale x 4 x i32> %res
}

; CHECK-LABEL: define <vscale x 4 x i32> @const_select()
; CHECK: select <vscale x 4 x i1>

define <vscale x 4 x i32> @const_select() {
  %s = select <vscale x 4 x i1> insertelement
      (<vscale x 4 x i1> undef, i1 true, i32 0),
     <vscale x 4 x i32> zeroinitializer,
     <vscale x 4 x i32> insertelement (<vscale x 4 x i32> undef, i32 1, i32 0)
  ret <vscale x 4 x i32> %s
}
