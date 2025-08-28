; RUN: opt -mtriple=dxil-pc-shadermodel6.2-compute -S -dxil-intrinsic-expansion %s | FileCheck %s --check-prefixes=CHECK,CHECK62
; RUN: opt -mtriple=dxil-pc-shadermodel6.3-compute -S -dxil-intrinsic-expansion %s | FileCheck %s --check-prefixes=CHECK,CHECK63

define void @loadf64(i32 %index) {
  ; check the handle from binding is unchanged
  ; CHECK: [[B:%.*]] = call target("dx.Rawbuffer", double, 0, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_f64_0_0t(
  ; CHECK-SAME: i32 0, i32 1, i32 1, i32 0, ptr null)
  %buffer = call target("dx.Rawbuffer", double, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_f64_1_0_0t(
          i32 0, i32 1, i32 1, i32 0, ptr null)

  ; check we don't modify the code in sm6.3 or later
  ; CHECK63: [[L0:%.*]] = call { double, i1 } @llvm.dx.resource.load.rawbuffer
  ; CHECK63-SAME: target("dx.Rawbuffer", double, 0, 0) [[B]], i32 %index, i32 0)

  ; check we load an <2 x i32> instead of a double
  ; CHECK62-NOT: call {double, i1} @llvm.dx.resource.load.rawbuffer
  ; CHECK62: [[L0:%.*]] = call { <2 x i32>, i1 }
  ; CHECK62-SAME: @llvm.dx.resource.load.rawbuffer.v2i32.tdx.Rawbuffer_f64_0_0t(
  ; CHECK62-SAME: target("dx.Rawbuffer", double, 0, 0) [[B]], i32 %index, i32 0)	
  %load0 = call {double, i1} @llvm.dx.resource.load.rawbuffer(
      target("dx.Rawbuffer", double, 0, 0) %buffer, i32 %index, i32 0)

  ; CHECK63: extractvalue { double, i1 } [[L0]], 0

  ; check we extract the two i32 and construct a double
  ; CHECK62: [[D0:%.*]] = extractvalue { <2 x i32>, i1 } [[L0]], 0
  ; CHECK62: [[Lo:%.*]] = extractelement <2 x i32> [[D0]], i32 0
  ; CHECK62: [[Hi:%.*]] = extractelement <2 x i32> [[D0]], i32 1
  ; CHECK62: [[DBL:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo]], i32 [[Hi]])
  ; CHECK62-NOT: extractvalue { double, i1 }
  %data0 = extractvalue {double, i1} %load0, 0
  ret void
}

define void @loadv2f64(i32 %index) {
  ; check the handle from binding is unchanged
  ; CHECK: [[B:%.*]] = call target("dx.Rawbuffer", <2 x double>, 0, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_v2f64_0_0t(
  ; CHECK-SAME: i32 0, i32 1, i32 1, i32 0, ptr null)
  %buffer = call target("dx.Rawbuffer", <2 x double>, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_v2f64_1_0_0t(
          i32 0, i32 1, i32 1, i32 0, ptr null)

  ; check we don't modify the code in sm6.3 or later
  ; CHECK63: [[L0:%.*]] = call { <2 x double>, i1 } @llvm.dx.resource.load.rawbuffer
  ; CHECK63-SAME: target("dx.Rawbuffer", <2 x double>, 0, 0) [[B]], i32 %index, i32 0)

  ; check we load an <4 x i32> instead of a double2
  ; CHECK62: [[L0:%.*]] = call { <4 x i32>, i1 }
  ; CHECK62-SAME: @llvm.dx.resource.load.rawbuffer.v4i32.tdx.Rawbuffer_v2f64_0_0t(
  ; CHECK62-SAME: target("dx.Rawbuffer", <2 x double>, 0, 0) [[B]], i32 %index, i32 0)
  %load0 = call { <2 x double>, i1 } @llvm.dx.resource.load.rawbuffer(
      target("dx.Rawbuffer", <2 x double>, 0, 0) %buffer, i32 %index, i32 0)

  ; CHECK63: extractvalue { <2 x double>, i1 } [[L0]], 0

  ; check we extract the 4 i32 and construct a <2 x double>
  ; CHECK62: [[D0:%.*]] = extractvalue { <4 x i32>, i1 } [[L0]], 0
  ; CHECK62: [[Lo1:%.*]] = extractelement <4 x i32> [[D0]], i32 0
  ; CHECK62: [[Hi1:%.*]] = extractelement <4 x i32> [[D0]], i32 1
  ; CHECK62: [[Lo2:%.*]] = extractelement <4 x i32> [[D0]], i32 2
  ; CHECK62: [[Hi2:%.*]] = extractelement <4 x i32> [[D0]], i32 3
  ; CHECK62: [[Dbl1:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo1]], i32 [[Hi1]])
  ; CHECK62: [[Vec:%.*]] = insertelement <2 x double> poison, double [[Dbl1]], i32 0
  ; CHECK62: [[Dbl2:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo2]], i32 [[Hi2]])
  ; CHECK62: [[Vec2:%.*]] = insertelement <2 x double> [[Vec]], double [[Dbl2]], i32 1
  ; CHECK62-NOT: extractvalue { <2 x double>, i1 }
  %data0 = extractvalue { <2 x double>, i1 } %load0, 0
  ret void
}

; show we properly handle extracting the check bit
define void @loadf64WithCheckBit(i32 %index) {
  ; check the handle from binding is unchanged
  ; CHECK: [[B:%.*]] = call target("dx.Rawbuffer", double, 0, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_f64_0_0t(
  ; CHECK-SAME: i32 0, i32 1, i32 1, i32 0, ptr null)
  %buffer = call target("dx.Rawbuffer", double, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_f64_1_0_0t(
          i32 0, i32 1, i32 1, i32 0, ptr null)

  ; check we don't modify the code in sm6.3 or later
  ; CHECK63: [[L0:%.*]] = call { double, i1 } @llvm.dx.resource.load.rawbuffer
  ; CHECK63-SAME: target("dx.Rawbuffer", double, 0, 0) [[B]], i32 %index, i32 0)

  ; check we load an <2 x i32> instead of a double
  ; CHECK62-NOT: call {double, i1} @llvm.dx.resource.load.rawbuffer
  ; CHECK62: [[L0:%.*]] = call { <2 x i32>, i1 }
  ; CHECK62-SAME: @llvm.dx.resource.load.rawbuffer.v2i32.tdx.Rawbuffer_f64_0_0t(
  ; CHECK62-SAME: target("dx.Rawbuffer", double, 0, 0) [[B]], i32 %index, i32 0)	
  %load0 = call {double, i1} @llvm.dx.resource.load.rawbuffer(
      target("dx.Rawbuffer", double, 0, 0) %buffer, i32 %index, i32 0)

  ; CHECK63: extractvalue { double, i1 } [[L0]], 0
  ; CHECK63: extractvalue { double, i1 } [[L0]], 1

  ; check we extract the two i32 and construct a double
  ; CHECK62: [[D0:%.*]] = extractvalue { <2 x i32>, i1 } [[L0]], 0
  ; CHECK62: [[Lo:%.*]] = extractelement <2 x i32> [[D0]], i32 0
  ; CHECK62: [[Hi:%.*]] = extractelement <2 x i32> [[D0]], i32 1
  ; CHECK62: [[DBL:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo]], i32 [[Hi]])
  %data0 = extractvalue {double, i1} %load0, 0
  ; CHECK62: extractvalue { <2 x i32>, i1 } [[L0]], 1
  ; CHECK62-NOT: extractvalue { double, i1 }
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
  ; CHECK-SAME: i32 0, i32 0, i32 1, i32 0, ptr null)
  %buffer = call target("dx.Rawbuffer", <3 x double>, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_v3f64_0_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; check we don't modify the code in sm6.3 or later
  ; CHECK63: [[L0:%.*]] = call { <3 x double>, i1 } @llvm.dx.resource.load.rawbuffer
  ; CHECK63-SAME: target("dx.Rawbuffer", <3 x double>, 0, 0) [[B]], i32 %index, i32 0)

  ; check we perform two loads
  ; and do 6 extracts and construct 3 doubles
  ; CHECK62-NOT: call {<3 x double>, i1} @llvm.dx.resource.load.rawbuffer
  ; CHECK62: [[L0:%.*]] = call { <4 x i32>, i1 }
  ; CHECK62-SAME: @llvm.dx.resource.load.rawbuffer.v4i32.tdx.Rawbuffer_v3f64_0_0t(
  ; CHECK62-SAME: target("dx.Rawbuffer", <3 x double>, 0, 0) [[B]], i32 %index, i32 0)

  ; CHECK62: [[D0:%.*]] = extractvalue { <4 x i32>, i1 } [[L0]], 0
  ; CHECK62: [[Lo1:%.*]] = extractelement <4 x i32> [[D0]], i32 0
  ; CHECK62: [[Hi1:%.*]] = extractelement <4 x i32> [[D0]], i32 1
  ; CHECK62: [[Lo2:%.*]] = extractelement <4 x i32> [[D0]], i32 2
  ; CHECK62: [[Hi2:%.*]] = extractelement <4 x i32> [[D0]], i32 3
  ; CHECK62: [[DBL1:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo1]], i32 [[Hi1]])
  ; CHECK62: [[Vec1:%.*]] = insertelement <3 x double> poison, double [[DBL1]], i32 0
  ; CHECK62: [[DBL2:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo2]], i32 [[Hi2]])
  ; CHECK62: [[Vec2:%.*]] = insertelement <3 x double> [[Vec1]], double [[DBL2]], i32 1

  ; 2nd load
  ; CHECK62: [[L2:%.*]] = call { <2 x i32>, i1 }
  ; CHECK62-SAME: @llvm.dx.resource.load.rawbuffer.v2i32.tdx.Rawbuffer_v3f64_0_0t(
  ; CHECK62-SAME: target("dx.Rawbuffer", <3 x double>, 0, 0) [[B]], i32 %index, i32 16)

  ; CHECK62: [[D2:%.*]] = extractvalue { <2 x i32>, i1 } [[L2]], 0
  ; CHECK62: [[Lo3:%.*]] = extractelement <2 x i32> [[D2]], i32 0
  ; CHECK62: [[Hi3:%.*]] = extractelement <2 x i32> [[D2]], i32 1
  ; CHECK62: [[DBL3:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo3]], i32 [[Hi3]])
  ; CHECK62: [[Vec3:%.*]] = insertelement <3 x double> [[Vec2]], double [[DBL3]], i32 2
  %load0 = call {<3 x double>, i1} @llvm.dx.resource.load.rawbuffer(
      target("dx.Rawbuffer", <3 x double>, 0, 0) %buffer, i32 %index, i32 0)

  ; CHECK63: extractvalue { <3 x double>, i1 } [[L0]], 0
  ; CHECK63: extractvalue { <3 x double>, i1 } [[L0]], 1

  ; CHECK62-NOT: extractvalue {<3 x double>, i1 }
  %data0 = extractvalue {<3 x double>, i1} %load0, 0
  ; check we extract checkbit from both loads and and them together
  ; CHECK62: [[B1:%.*]] = extractvalue { <4 x i32>, i1 } [[L0]], 1
  ; CHECK62: [[B2:%.*]] = extractvalue { <2 x i32>, i1 } [[L2]], 1
  ; CHECK62: and i1 [[B1]], [[B2]]
  %cb = extractvalue {<3 x double>, i1} %load0, 1
  ret void
}

define void @loadv4f64(i32 %index) {
  ; check the handle from binding is unchanged
  ; CHECK: [[B:%.*]] = call target("dx.Rawbuffer", <4 x double>, 0, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_v4f64_0_0t(
  ; CHECK-SAME: i32 0, i32 0, i32 1, i32 0, ptr null)
  %buffer = call target("dx.Rawbuffer", <4 x double>, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_v4f64_0_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; check we don't modify the code in sm6.3 or later
  ; CHECK63: [[L0:%.*]] = call { <4 x double>, i1 } @llvm.dx.resource.load.rawbuffer
  ; CHECK63-SAME: target("dx.Rawbuffer", <4 x double>, 0, 0) [[B]], i32 %index, i32 0)

  ; check we perform two loads
  ; and do 8 extracts and construct 4 doubles
  ; CHECK62-NOT: call {<4 x double>, i1} @llvm.dx.resource.load.rawbuffer
  ; CHECK62: [[L0:%.*]] = call { <4 x i32>, i1 }
  ; CHECK62-SAME: @llvm.dx.resource.load.rawbuffer.v4i32.tdx.Rawbuffer_v4f64_0_0t(
  ; CHECK62-SAME: target("dx.Rawbuffer", <4 x double>, 0, 0) [[B]], i32 %index, i32 0)

  ; CHECK62: [[D0:%.*]] = extractvalue { <4 x i32>, i1 } [[L0]], 0
  ; CHECK62: [[Lo1:%.*]] = extractelement <4 x i32> [[D0]], i32 0
  ; CHECK62: [[Hi1:%.*]] = extractelement <4 x i32> [[D0]], i32 1
  ; CHECK62: [[Lo2:%.*]] = extractelement <4 x i32> [[D0]], i32 2
  ; CHECK62: [[Hi2:%.*]] = extractelement <4 x i32> [[D0]], i32 3
  ; CHECK62: [[DBL1:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo1]], i32 [[Hi1]])
  ; CHECK62: [[Vec1:%.*]] = insertelement <4 x double> poison, double [[DBL1]], i32 0
  ; CHECK62: [[DBL2:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo2]], i32 [[Hi2]])
  ; CHECK62: [[Vec2:%.*]] = insertelement <4 x double> [[Vec1]], double [[DBL2]], i32 1

  ; 2nd load
  ; CHECK62: [[L2:%.*]] = call { <4 x i32>, i1 }
  ; CHECK62-SAME: @llvm.dx.resource.load.rawbuffer.v4i32.tdx.Rawbuffer_v4f64_0_0t(
  ; CHECK62-SAME: target("dx.Rawbuffer", <4 x double>, 0, 0) [[B]], i32 %index, i32 16)

  ; CHECK62: [[D2:%.*]] = extractvalue { <4 x i32>, i1 } [[L2]], 0
  ; CHECK62: [[Lo3:%.*]] = extractelement <4 x i32> [[D2]], i32 0
  ; CHECK62: [[Hi3:%.*]] = extractelement <4 x i32> [[D2]], i32 1
  ; CHECK62: [[Lo4:%.*]] = extractelement <4 x i32> [[D2]], i32 2
  ; CHECK62: [[Hi4:%.*]] = extractelement <4 x i32> [[D2]], i32 3
  ; CHECK62: [[DBL3:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo3]], i32 [[Hi3]])
  ; CHECK62: [[Vec3:%.*]] = insertelement <4 x double> [[Vec2]], double [[DBL3]], i32 2
  ; CHECK62: [[DBL4:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo4]], i32 [[Hi4]])
  ; CHECK62: [[Vec4:%.*]] = insertelement <4 x double> [[Vec3]], double [[DBL4]], i32 3
  %load0 = call {<4 x double>, i1} @llvm.dx.resource.load.rawbuffer(
      target("dx.Rawbuffer", <4 x double>, 0, 0) %buffer, i32 %index, i32 0)

  ; CHECK63: extractvalue { <4 x double>, i1 } [[L0]], 0
  ; CHECK63: extractvalue { <4 x double>, i1 } [[L0]], 1

  ; CHECK62-NOT: extractvalue {<4 x double>, i1 }
  %data0 = extractvalue {<4 x double>, i1} %load0, 0
  ; check we extract checkbit from both loads and and them together
  ; CHECK62: [[B1:%.*]] = extractvalue { <4 x i32>, i1 } [[L0]], 1
  ; CHECK62: [[B2:%.*]] = extractvalue { <4 x i32>, i1 } [[L2]], 1
  ; CHECK62: and i1 [[B1]], [[B2]]
  %cb = extractvalue {<4 x double>, i1} %load0, 1
  ret void
}
