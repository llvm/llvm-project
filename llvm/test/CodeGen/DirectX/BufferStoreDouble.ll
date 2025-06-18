; RUN: opt -S -dxil-intrinsic-expansion %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

define void @storef64(double %0) {
  ; CHECK: [[B:%.*]] = tail call target("dx.TypedBuffer", double, 1, 0, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f64_1_0_0t(
  ; CHECK-SAME: i32 0, i32 0, i32 1, i32 0, i1 false, ptr null)
  %buffer = tail call target("dx.TypedBuffer", double, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f64_1_0_0t(
          i32 0, i32 0, i32 1, i32 0, i1 false, ptr null)

  ; check we split the double and store the lo and hi bits
  ; CHECK: [[SD:%.*]] = call { i32, i32 } @llvm.dx.splitdouble.i32(double %0)
  ; CHECK: [[Lo:%.*]] = extractvalue { i32, i32 } [[SD]], 0
  ; CHECK: [[Hi:%.*]] = extractvalue { i32, i32 } [[SD]], 1
  ; CHECK: [[Vec1:%.*]] = insertelement <2 x i32> poison, i32 [[Lo]], i32 0
  ; CHECK: [[Vec2:%.*]] = insertelement <2 x i32> [[Vec1]], i32 [[Hi]], i32 1
  ; CHECK: call void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_f64_1_0_0t.v2i32(
  ; CHECK-SAME: target("dx.TypedBuffer", double, 1, 0, 0) [[B]], i32 0, <2 x i32> [[Vec2]])
  call void @llvm.dx.resource.store.typedbuffer(
      target("dx.TypedBuffer", double, 1, 0, 0) %buffer, i32 0,
      double %0)
  ret void
}


define void @storev2f64(<2 x double> %0) {
  ; CHECK: [[B:%.*]] = tail call target("dx.TypedBuffer", <2 x double>, 1, 0, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v2f64_1_0_0t(
  ; CHECK-SAME: i32 0, i32 0, i32 1, i32 0, i1 false, ptr null)
  %buffer = tail call target("dx.TypedBuffer", <2 x double>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v2f64_1_0_0t(
          i32 0, i32 0, i32 1, i32 0, i1 false, ptr null)

  ; CHECK: [[SD:%.*]] = call { <2 x i32>, <2 x i32> }
  ; CHECK-SAME: @llvm.dx.splitdouble.v2i32(<2 x double> %0)
  ; CHECK: [[Lo:%.*]] = extractvalue { <2 x i32>, <2 x i32> } [[SD]], 0
  ; CHECK: [[Hi:%.*]] = extractvalue { <2 x i32>, <2 x i32> } [[SD]], 1
  ; CHECK: [[Vec:%.*]] = shufflevector <2 x i32> [[Lo]], <2 x i32> [[Hi]], <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  ; CHECK: call void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_v2f64_1_0_0t.v4i32(
  ; CHECK-SAME: target("dx.TypedBuffer", <2 x double>, 1, 0, 0) [[B]], i32 0, <4 x i32> [[Vec]])
  call void @llvm.dx.resource.store.typedbuffer(
      target("dx.TypedBuffer", <2 x double>, 1, 0, 0) %buffer, i32 0,
      <2 x double> %0)
  ret void
}

define { double, i1 } @loadAndReturnf64() {
; CHECK-LABEL: define { double, i1 } @loadAndReturnf64() {
; CHECK-NEXT:    [[BUFFER:%.*]] = tail call target("dx.TypedBuffer", double, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f64_1_0_0t(i32 0, i32 0, i32 1, i32 0, i1 false, ptr null)
; CHECK-NEXT:    [[TMP1:%.*]] = call { <2 x i32>, i1 } @llvm.dx.resource.load.typedbuffer.v2i32.tdx.TypedBuffer_f64_1_0_0t(target("dx.TypedBuffer", double, 1, 0, 0) [[BUFFER]], i32 0)
; CHECK-NEXT:    [[TMP2:%.*]] = extractvalue { <2 x i32>, i1 } [[TMP1]], 0
; CHECK-NEXT:    [[TMP3:%.*]] = extractelement <2 x i32> [[TMP2]], i32 0
; CHECK-NEXT:    [[TMP4:%.*]] = extractelement <2 x i32> [[TMP2]], i32 1
; CHECK-NEXT:    [[TMP5:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[TMP3]], i32 [[TMP4]])
; CHECK-NEXT:    [[TMP6:%.*]] = insertvalue { double, i1 } poison, double [[TMP5]], 0
; CHECK-NEXT:    [[TMP7:%.*]] = extractvalue { <2 x i32>, i1 } [[TMP1]], 1
; CHECK-NEXT:    [[TMP8:%.*]] = insertvalue { double, i1 } [[TMP6]], i1 [[TMP7]], 1
; CHECK-NEXT:    ret { double, i1 } [[TMP8]]
;
  %buffer = tail call target("dx.TypedBuffer", double, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f64_1_0_0t(i32 0, i32 0, i32 1, i32 0, i1 false, ptr null)
  %result = call { double, i1 } @llvm.dx.resource.load.typedbuffer.tdx.TypedBuffer_f64_1_0_0t(
  target("dx.TypedBuffer", double, 1, 0, 0) %buffer, i32 0)
  ret { double, i1 } %result
}

define { <2 x double>, i1 } @loadAndReturnv2f64() {
; CHECK-LABEL: define { <2 x double>, i1 } @loadAndReturnv2f64() {
; CHECK-NEXT:    [[BUFFER:%.*]] = tail call target("dx.TypedBuffer", <2 x double>, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v2f64_1_0_0t(i32 0, i32 0, i32 1, i32 0, i1 false, ptr null)
; CHECK-NEXT:    [[TMP1:%.*]] = call { <4 x i32>, i1 } @llvm.dx.resource.load.typedbuffer.v4i32.tdx.TypedBuffer_v2f64_1_0_0t(target("dx.TypedBuffer", <2 x double>, 1, 0, 0) [[BUFFER]], i32 0)
; CHECK-NEXT:    [[TMP2:%.*]] = extractvalue { <4 x i32>, i1 } [[TMP1]], 0
; CHECK-NEXT:    [[TMP3:%.*]] = extractelement <4 x i32> [[TMP2]], i32 0
; CHECK-NEXT:    [[TMP4:%.*]] = extractelement <4 x i32> [[TMP2]], i32 1
; CHECK-NEXT:    [[TMP5:%.*]] = extractelement <4 x i32> [[TMP2]], i32 2
; CHECK-NEXT:    [[TMP6:%.*]] = extractelement <4 x i32> [[TMP2]], i32 3
; CHECK-NEXT:    [[TMP7:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[TMP3]], i32 [[TMP4]])
; CHECK-NEXT:    [[TMP8:%.*]] = insertelement <2 x double> poison, double [[TMP7]], i32 0
; CHECK-NEXT:    [[TMP9:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[TMP5]], i32 [[TMP6]])
; CHECK-NEXT:    [[TMP10:%.*]] = insertelement <2 x double> [[TMP8]], double [[TMP9]], i32 1
; CHECK-NEXT:    [[TMP11:%.*]] = insertvalue { <2 x double>, i1 } poison, <2 x double> [[TMP10]], 0
; CHECK-NEXT:    [[TMP12:%.*]] = extractvalue { <4 x i32>, i1 } [[TMP1]], 1
; CHECK-NEXT:    [[TMP13:%.*]] = insertvalue { <2 x double>, i1 } [[TMP11]], i1 [[TMP12]], 1
; CHECK-NEXT:    ret { <2 x double>, i1 } [[TMP13]]
;
  %buffer = tail call target("dx.TypedBuffer", <2 x double>, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v2f64_1_0_0t(i32 0, i32 0, i32 1, i32 0, i1 false, ptr null)
  %result = call { <2 x double>, i1 } @llvm.dx.resource.load.typedbuffer.tdx.TypedBuffer_v2f64_1_0_0t(
  target("dx.TypedBuffer", <2 x double>, 1, 0, 0) %buffer, i32 0)
  ret { <2 x double>, i1 } %result
}
