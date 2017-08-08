; RUN: opt -instcombine -S < %s | FileCheck %s

declare <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %ptrs, i32, <2 x i1> %mask, <2 x double> %src0)
declare void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %val, <2 x double>* %ptrs, i32, <2 x i1> %mask)
declare <2 x double> @llvm.masked.gather.v2f64.v2p0f64(<2 x double*> %ptrs, i32, <2 x i1> %mask, <2 x double> %passthru)
declare void @llvm.masked.scatter.v2f64.v2p0f64(<2 x double> %val, <2 x double*> %ptrs, i32, <2 x i1> %mask)

define <2 x double> @load_zeromask(<2 x double>* %ptr, <2 x double> %passthru)  {
  %res = call <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %ptr, i32 1, <2 x i1> zeroinitializer, <2 x double> %passthru)
  ret <2 x double> %res

; CHECK-LABEL: @load_zeromask(
; CHECK-NEXT:  ret <2 x double> %passthru
}

define <2 x double> @load_onemask(<2 x double>* %ptr, <2 x double> %passthru)  {
  %res = call <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %ptr, i32 2, <2 x i1> <i1 1, i1 1>, <2 x double> %passthru)
  ret <2 x double> %res

; CHECK-LABEL: @load_onemask(
; CHECK-NEXT:  %unmaskedload = load <2 x double>, <2 x double>* %ptr, align 2
; CHECK-NEXT:  ret <2 x double> %unmaskedload
}

define <2 x double> @load_undefmask(<2 x double>* %ptr, <2 x double> %passthru)  {
  %res = call <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %ptr, i32 2, <2 x i1> <i1 1, i1 undef>, <2 x double> %passthru)
  ret <2 x double> %res

; CHECK-LABEL: @load_undefmask(
; CHECK-NEXT:  %unmaskedload = load <2 x double>, <2 x double>* %ptr, align 2
; CHECK-NEXT:  ret <2 x double> %unmaskedload
}

define void @store_zeromask(<2 x double>* %ptr, <2 x double> %val)  {
  call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %val, <2 x double>* %ptr, i32 3, <2 x i1> zeroinitializer)
  ret void

; CHECK-LABEL: @store_zeromask(
; CHECK-NEXT:  ret void
}

define void @store_onemask(<2 x double>* %ptr, <2 x double> %val)  {
  call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %val, <2 x double>* %ptr, i32 4, <2 x i1> <i1 1, i1 1>)
  ret void

; CHECK-LABEL: @store_onemask(
; CHECK-NEXT:  store <2 x double> %val, <2 x double>* %ptr, align 4
; CHECK-NEXT:  ret void
}

define <2 x double> @gather_zeromask(<2 x double*> %ptrs, <2 x double> %passthru)  {
  %res = call <2 x double> @llvm.masked.gather.v2f64.v2p0f64(<2 x double*> %ptrs, i32 5, <2 x i1> zeroinitializer, <2 x double> %passthru)
  ret <2 x double> %res

; CHECK-LABEL: @gather_zeromask(
; CHECK-NEXT:  ret <2 x double> %passthru
}

define void @scatter_zeromask(<2 x double*> %ptrs, <2 x double> %val)  {
  call void @llvm.masked.scatter.v2f64.v2p0f64(<2 x double> %val, <2 x double*> %ptrs, i32 6, <2 x i1> zeroinitializer)
  ret void

; CHECK-LABEL: @scatter_zeromask(
; CHECK-NEXT:  ret void
}

