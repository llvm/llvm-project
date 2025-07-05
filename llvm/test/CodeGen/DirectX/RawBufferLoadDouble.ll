; RUN: opt -S -dxil-intrinsic-expansion %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.2-compute"

define void @loadf64(i32 %index) {
  ; check the handle from binding is unchanged
  ; CHECK: [[B:%.*]] = call target("dx.Rawbuffer", double, 0, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_f64_0_0t(
  ; CHECK-SAME: i32 0, i32 1, i32 1, i32 0, i1 false, ptr null)
  %buffer = call target("dx.Rawbuffer", double, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_f64_1_0_0t(
          i32 0, i32 1, i32 1, i32 0, i1 false, ptr null)

  ; check we load an <2 x i32> instead of a double
  ; CHECK-NOT: call {double, i1} @llvm.dx.resource.load.rawbuffer
  ; CHECK: [[L0:%.*]] = call { <2 x i32>, i1 }
  ; CHECK-SAME: @llvm.dx.resource.load.rawbuffer.v2i32.tdx.Rawbuffer_f64_0_0t(
  ; CHECK-SAME: target("dx.Rawbuffer", double, 0, 0) [[B]], i32 %index, i32 0)	
  %load0 = call {double, i1} @llvm.dx.resource.load.rawbuffer(
      target("dx.Rawbuffer", double, 0, 0) %buffer, i32 %index, i32 0)

  ; check we extract the two i32 and construct a double
  ; CHECK: [[D0:%.*]] = extractvalue { <2 x i32>, i1 } [[L0]], 0
  ; CHECK: [[Lo:%.*]] = extractelement <2 x i32> [[D0]], i32 0
  ; CHECK: [[Hi:%.*]] = extractelement <2 x i32> [[D0]], i32 1
  ; CHECK: [[DBL:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo]], i32 [[Hi]])
  ; CHECK-NOT: extractvalue { double, i1 }
  %data0 = extractvalue {double, i1} %load0, 0
  ret void
}

define void @loadv2f64(i32 %index) {
  ; check the handle from binding is unchanged
  ; CHECK: [[B:%.*]] = call target("dx.Rawbuffer", <2 x double>, 0, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_v2f64_0_0t(
  ; CHECK-SAME: i32 0, i32 1, i32 1, i32 0, i1 false, ptr null)
  %buffer = call target("dx.Rawbuffer", <2 x double>, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_v2f64_1_0_0t(
          i32 0, i32 1, i32 1, i32 0, i1 false, ptr null)

  ; check we load an <4 x i32> instead of a double2
  ; CHECK: [[L0:%.*]] = call { <4 x i32>, i1 }
  ; CHECK-SAME: @llvm.dx.resource.load.rawbuffer.v4i32.tdx.Rawbuffer_v2f64_0_0t(
  ; CHECK-SAME: target("dx.Rawbuffer", <2 x double>, 0, 0) [[B]], i32 %index, i32 0)
  %load0 = call { <2 x double>, i1 } @llvm.dx.resource.load.rawbuffer(
      target("dx.Rawbuffer", <2 x double>, 0, 0) %buffer, i32 %index, i32 0)

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
define void @loadf64WithCheckBit(i32 %index) {
  ; check the handle from binding is unchanged
  ; CHECK: [[B:%.*]] = call target("dx.Rawbuffer", double, 0, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_f64_0_0t(
  ; CHECK-SAME: i32 0, i32 1, i32 1, i32 0, i1 false, ptr null)
  %buffer = call target("dx.Rawbuffer", double, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_f64_1_0_0t(
          i32 0, i32 1, i32 1, i32 0, i1 false, ptr null)

  ; check we load an <2 x i32> instead of a double
  ; CHECK-NOT: call {double, i1} @llvm.dx.resource.load.rawbuffer
  ; CHECK: [[L0:%.*]] = call { <2 x i32>, i1 }
  ; CHECK-SAME: @llvm.dx.resource.load.rawbuffer.v2i32.tdx.Rawbuffer_f64_0_0t(
  ; CHECK-SAME: target("dx.Rawbuffer", double, 0, 0) [[B]], i32 %index, i32 0)	
  %load0 = call {double, i1} @llvm.dx.resource.load.rawbuffer(
      target("dx.Rawbuffer", double, 0, 0) %buffer, i32 %index, i32 0)

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

; Raw Buffer Load allows for double3 and double4 to be loaded
; In SM6.2 and below, two loads will be performed.
; Show we and the checkbits together
  
define void @loadv3f64(i32 %index) {
  ; check the handle from binding is unchanged
  ; CHECK: [[B:%.*]] = call target("dx.Rawbuffer", <3 x double>, 0, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_v3f64_0_0t(
  ; CHECK-SAME: i32 0, i32 0, i32 1, i32 0, i1 false, ptr null)
  %buffer = call target("dx.Rawbuffer", <3 x double>, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_v3f64_0_0t(
          i32 0, i32 0, i32 1, i32 0, i1 false, ptr null)

  ; check we perform two loads
  ; and do 6 extracts and construct 3 doubles
  ; CHECK-NOT: call {<3 x double>, i1} @llvm.dx.resource.load.rawbuffer
  ; CHECK: [[L0:%.*]] = call { <4 x i32>, i1 }
  ; CHECK-SAME: @llvm.dx.resource.load.rawbuffer.v4i32.tdx.Rawbuffer_v3f64_0_0t(
  ; CHECK-SAME: target("dx.Rawbuffer", <3 x double>, 0, 0) [[B]], i32 %index, i32 0)

  ; CHECK: [[D0:%.*]] = extractvalue { <4 x i32>, i1 } [[L0]], 0
  ; CHECK: [[Lo1:%.*]] = extractelement <4 x i32> [[D0]], i32 0
  ; CHECK: [[Hi1:%.*]] = extractelement <4 x i32> [[D0]], i32 1
  ; CHECK: [[Lo2:%.*]] = extractelement <4 x i32> [[D0]], i32 2
  ; CHECK: [[Hi2:%.*]] = extractelement <4 x i32> [[D0]], i32 3
  ; CHECK: [[DBL1:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo1]], i32 [[Hi1]])
  ; CHECK: [[Vec1:%.*]] = insertelement <3 x double> poison, double [[DBL1]], i32 0
  ; CHECK: [[DBL2:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo2]], i32 [[Hi2]])
  ; CHECK: [[Vec2:%.*]] = insertelement <3 x double> [[Vec1]], double [[DBL2]], i32 1

  ; 2nd load
  ; CHECK: [[L2:%.*]] = call { <2 x i32>, i1 }
  ; CHECK-SAME: @llvm.dx.resource.load.rawbuffer.v2i32.tdx.Rawbuffer_v3f64_0_0t(
  ; CHECK-SAME: target("dx.Rawbuffer", <3 x double>, 0, 0) [[B]], i32 %index, i32 16)

  ; CHECK: [[D2:%.*]] = extractvalue { <2 x i32>, i1 } [[L2]], 0
  ; CHECK: [[Lo3:%.*]] = extractelement <2 x i32> [[D2]], i32 0
  ; CHECK: [[Hi3:%.*]] = extractelement <2 x i32> [[D2]], i32 1
  ; CHECK: [[DBL3:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo3]], i32 [[Hi3]])
  ; CHECK: [[Vec3:%.*]] = insertelement <3 x double> [[Vec2]], double [[DBL3]], i32 2
  %load0 = call {<3 x double>, i1} @llvm.dx.resource.load.rawbuffer(
      target("dx.Rawbuffer", <3 x double>, 0, 0) %buffer, i32 %index, i32 0)


  ; CHECK-NOT: extractvalue {<3 x double>, i1 }
  %data0 = extractvalue {<3 x double>, i1} %load0, 0
  ; check we extract checkbit from both loads and and them together
  ; CHECK: [[B1:%.*]] = extractvalue { <4 x i32>, i1 } [[L0]], 1
  ; CHECK: [[B2:%.*]] = extractvalue { <2 x i32>, i1 } [[L2]], 1
  ; CHECK: and i1 [[B1]], [[B2]]
  %cb = extractvalue {<3 x double>, i1} %load0, 1
  ret void
}

define void @loadv4f64(i32 %index) {
  ; check the handle from binding is unchanged
  ; CHECK: [[B:%.*]] = call target("dx.Rawbuffer", <4 x double>, 0, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_v4f64_0_0t(
  ; CHECK-SAME: i32 0, i32 0, i32 1, i32 0, i1 false, ptr null)
  %buffer = call target("dx.Rawbuffer", <4 x double>, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_v4f64_0_0t(
          i32 0, i32 0, i32 1, i32 0, i1 false, ptr null)

  ; check we perform two loads
  ; and do 8 extracts and construct 4 doubles
  ; CHECK-NOT: call {<4 x double>, i1} @llvm.dx.resource.load.rawbuffer
  ; CHECK: [[L0:%.*]] = call { <4 x i32>, i1 }
  ; CHECK-SAME: @llvm.dx.resource.load.rawbuffer.v4i32.tdx.Rawbuffer_v4f64_0_0t(
  ; CHECK-SAME: target("dx.Rawbuffer", <4 x double>, 0, 0) [[B]], i32 %index, i32 0)

  ; CHECK: [[D0:%.*]] = extractvalue { <4 x i32>, i1 } [[L0]], 0
  ; CHECK: [[Lo1:%.*]] = extractelement <4 x i32> [[D0]], i32 0
  ; CHECK: [[Hi1:%.*]] = extractelement <4 x i32> [[D0]], i32 1
  ; CHECK: [[Lo2:%.*]] = extractelement <4 x i32> [[D0]], i32 2
  ; CHECK: [[Hi2:%.*]] = extractelement <4 x i32> [[D0]], i32 3
  ; CHECK: [[DBL1:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo1]], i32 [[Hi1]])
  ; CHECK: [[Vec1:%.*]] = insertelement <4 x double> poison, double [[DBL1]], i32 0
  ; CHECK: [[DBL2:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo2]], i32 [[Hi2]])
  ; CHECK: [[Vec2:%.*]] = insertelement <4 x double> [[Vec1]], double [[DBL2]], i32 1

  ; 2nd load
  ; CHECK: [[L2:%.*]] = call { <4 x i32>, i1 }
  ; CHECK-SAME: @llvm.dx.resource.load.rawbuffer.v4i32.tdx.Rawbuffer_v4f64_0_0t(
  ; CHECK-SAME: target("dx.Rawbuffer", <4 x double>, 0, 0) [[B]], i32 %index, i32 16)

  ; CHECK: [[D2:%.*]] = extractvalue { <4 x i32>, i1 } [[L2]], 0
  ; CHECK: [[Lo3:%.*]] = extractelement <4 x i32> [[D2]], i32 0
  ; CHECK: [[Hi3:%.*]] = extractelement <4 x i32> [[D2]], i32 1
  ; CHECK: [[Lo4:%.*]] = extractelement <4 x i32> [[D2]], i32 2
  ; CHECK: [[Hi4:%.*]] = extractelement <4 x i32> [[D2]], i32 3
  ; CHECK: [[DBL3:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo3]], i32 [[Hi3]])
  ; CHECK: [[Vec3:%.*]] = insertelement <4 x double> [[Vec2]], double [[DBL3]], i32 2
  ; CHECK: [[DBL4:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo4]], i32 [[Hi4]])
  ; CHECK: [[Vec4:%.*]] = insertelement <4 x double> [[Vec3]], double [[DBL4]], i32 3
  %load0 = call {<4 x double>, i1} @llvm.dx.resource.load.rawbuffer(
      target("dx.Rawbuffer", <4 x double>, 0, 0) %buffer, i32 %index, i32 0)


  ; CHECK-NOT: extractvalue {<4 x double>, i1 }
  %data0 = extractvalue {<4 x double>, i1} %load0, 0
  ; check we extract checkbit from both loads and and them together
  ; CHECK: [[B1:%.*]] = extractvalue { <4 x i32>, i1 } [[L0]], 1
  ; CHECK: [[B2:%.*]] = extractvalue { <4 x i32>, i1 } [[L2]], 1
  ; CHECK: and i1 [[B1]], [[B2]]
  %cb = extractvalue {<4 x double>, i1} %load0, 1
  ret void
}
