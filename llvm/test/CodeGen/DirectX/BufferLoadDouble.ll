; RUN: opt -S -dxil-intrinsic-expansion %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

define void @loadf64() {
  ; check the handle from binding is unchanged
  ; CHECK: [[B:%.*]] = call target("dx.TypedBuffer", double, 1, 0, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f64_1_0_0t(
  ; CHECK-SAME: i32 0, i32 1, i32 1, i32 0, ptr null)
  %buffer = call target("dx.TypedBuffer", double, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f64_1_0_0t(
          i32 0, i32 1, i32 1, i32 0, ptr null)

  ; check we load an <2 x i32> instead of a double
  ; CHECK-NOT: call {double, i1} @llvm.dx.resource.load.typedbuffer
  ; CHECK: [[L0:%.*]] = call { <2 x i32>, i1 }
  ; CHECK-SAME: @llvm.dx.resource.load.typedbuffer.v2i32.tdx.TypedBuffer_f64_1_0_0t(
  ; CHECK-SAME: target("dx.TypedBuffer", double, 1, 0, 0) [[B]], i32 0)	
  %load0 = call {double, i1} @llvm.dx.resource.load.typedbuffer(
      target("dx.TypedBuffer", double, 1, 0, 0) %buffer, i32 0)

  ; check we extract the two i32 and construct a double
  ; CHECK: [[D0:%.*]] = extractvalue { <2 x i32>, i1 } [[L0]], 0
  ; CHECK: [[Lo:%.*]] = extractelement <2 x i32> [[D0]], i32 0
  ; CHECK: [[Hi:%.*]] = extractelement <2 x i32> [[D0]], i32 1
  ; CHECK: [[DBL:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo]], i32 [[Hi]])
  ; CHECK-NOT: extractvalue { double, i1 }
  %data0 = extractvalue {double, i1} %load0, 0
  ret void
}

define void @loadv2f64() {
  ; check the handle from binding is unchanged
  ; CHECK: [[B:%.*]] = call target("dx.TypedBuffer", <2 x double>, 1, 0, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v2f64_1_0_0t(
  ; CHECK-SAME: i32 0, i32 1, i32 1, i32 0, ptr null)
  %buffer = call target("dx.TypedBuffer", <2 x double>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v2f64_1_0_0t(
          i32 0, i32 1, i32 1, i32 0, ptr null)

  ; check we load an <4 x i32> instead of a double2
  ; CHECK: [[L0:%.*]] = call { <4 x i32>, i1 }
  ; CHECK-SAME: @llvm.dx.resource.load.typedbuffer.v4i32.tdx.TypedBuffer_v2f64_1_0_0t(
  ; CHECK-SAME: target("dx.TypedBuffer", <2 x double>, 1, 0, 0) [[B]], i32 0)
  %load0 = call { <2 x double>, i1 } @llvm.dx.resource.load.typedbuffer(
      target("dx.TypedBuffer", <2 x double>, 1, 0, 0) %buffer, i32 0)

  ; check we extract the 4 i32 and construct a <2 x double>
  ; CHECK: [[D0:%.*]] = extractvalue { <4 x i32>, i1 } [[L0]], 0
  ; CHECK: [[Lo1:%.*]] = extractelement <4 x i32> [[D0]], i32 0
  ; CHECK: [[Hi1:%.*]] = extractelement <4 x i32> [[D0]], i32 1
  ; CHECK: [[Lo2:%.*]] = extractelement <4 x i32> [[D0]], i32 2
  ; CHECK: [[Hi2:%.*]] = extractelement <4 x i32> [[D0]], i32 3
  ; CHECK: [[Dbl1:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo1]], i32 [[Hi1]])
  ; CHECK: [[Vec:%.*]] = insertelement <2 x double> poison, double [[Dbl1]], i32 0
  ; CHECK: [[Dbl2:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo2]], i32 [[Hi2]])
  ; CHECK: [[Vec2:%.*]] = insertelement <2 x double> [[Vec]], double [[Dbl2]], i32 1
  ; CHECK-NOT: extractvalue { <2 x double>, i1 }
  %data0 = extractvalue { <2 x double>, i1 } %load0, 0
  ret void
}

; show we properly handle extracting the check bit
define void @loadf64WithCheckBit() {
  ; check the handle from binding is unchanged
  ; CHECK: [[B:%.*]] = call target("dx.TypedBuffer", double, 1, 0, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f64_1_0_0t(
  ; CHECK-SAME: i32 0, i32 1, i32 1, i32 0, ptr null)
  %buffer = call target("dx.TypedBuffer", double, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f64_1_0_0t(
          i32 0, i32 1, i32 1, i32 0, ptr null)

  ; check we load an <2 x i32> instead of a double
  ; CHECK-NOT: call {double, i1} @llvm.dx.resource.load.typedbuffer
  ; CHECK: [[L0:%.*]] = call { <2 x i32>, i1 }
  ; CHECK-SAME: @llvm.dx.resource.load.typedbuffer.v2i32.tdx.TypedBuffer_f64_1_0_0t(
  ; CHECK-SAME: target("dx.TypedBuffer", double, 1, 0, 0) [[B]], i32 0)	
  %load0 = call {double, i1} @llvm.dx.resource.load.typedbuffer(
      target("dx.TypedBuffer", double, 1, 0, 0) %buffer, i32 0)

  ; check we extract the two i32 and construct a double
  ; CHECK: [[D0:%.*]] = extractvalue { <2 x i32>, i1 } [[L0]], 0
  ; CHECK: [[Lo:%.*]] = extractelement <2 x i32> [[D0]], i32 0
  ; CHECK: [[Hi:%.*]] = extractelement <2 x i32> [[D0]], i32 1
  ; CHECK: [[DBL:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo]], i32 [[Hi]])
  %data0 = extractvalue {double, i1} %load0, 0
  ; CHECK: extractvalue { <2 x i32>, i1 } [[L0]], 1
  ; CHECK-NOT: extractvalue { double, i1 }
  %cb = extractvalue {double, i1} %load0, 1
  ret void
}
