; RUN: opt -mtriple=dxil-pc-shadermodel6.2-compute -S -dxil-intrinsic-expansion %s | FileCheck %s --check-prefixes=CHECK,CHECK62
; RUN: opt -mtriple=dxil-pc-shadermodel6.3-compute -S -dxil-intrinsic-expansion %s | FileCheck %s --check-prefixes=CHECK,CHECK63

define void @loadi64(i32 %index) {
  ; check the handle from binding is unchanged
  ; CHECK: [[B:%.*]] = call target("dx.Rawbuffer", i64, 1, 0, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_i64_1_0_0t(
  ; CHECK-SAME: i32 0, i32 1, i32 1, i32 0, ptr null)
  %buffer = call target("dx.Rawbuffer", i64, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_i64_1_0_0t(
          i32 0, i32 1, i32 1, i32 0, ptr null)

  ; check we don't modify the code in sm6.3 or later
  ; CHECK63: [[L0:%.*]] = call { i64, i1 } @llvm.dx.resource.load.rawbuffer
  ; CHECK63-SAME: target("dx.Rawbuffer", i64, 1, 0, 0) [[B]], i32 %index, i32 0)

  ; check we load an <2 x i32> instead of a i64
  ; CHECK62-NOT: call {i64, i1} @llvm.dx.resource.load.rawbuffer
  ; CHECK62: [[L0:%.*]] = call { <2 x i32>, i1 }
  ; CHECK62-SAME: @llvm.dx.resource.load.rawbuffer.v2i32.tdx.Rawbuffer_i64_1_0_0t(
  ; CHECK62-SAME: target("dx.Rawbuffer", i64, 1, 0, 0) [[B]], i32 %index, i32 0)	
  %load0 = call {i64, i1} @llvm.dx.resource.load.rawbuffer(
      target("dx.Rawbuffer", i64, 1, 0, 0) %buffer, i32 %index, i32 0)

  ; CHECK63: extractvalue { i64, i1 } [[L0]], 0

  ; check we extract the two i32 and construct a i64
  ; CHECK62: [[D0:%.*]] = extractvalue { <2 x i32>, i1 } [[L0]], 0
  ; CHECK62: [[Lo:%.*]] = extractelement <2 x i32> [[D0]], i32 0
  ; CHECK62: [[Hi:%.*]] = extractelement <2 x i32> [[D0]], i32 1
  ; CHECK62: [[ZLo1:%.*]] = zext i32 [[Lo]] to i64
  ; CHECK62: [[ZHi1:%.*]] = zext i32 [[Hi]] to i64
  ; CHECK62: [[A:%.*]] = shl i64 [[ZHi1]], 32
  ; CHECK62: [[B:%.*]] = or i64 [[ZLo1]], [[A]]
  ; CHECK62-NOT: extractvalue { i64, i1 }
  %data0 = extractvalue {i64, i1} %load0, 0
  ret void
}

define void @loadv2i64(i32 %index) {
  ; check the handle from binding is unchanged
  ; CHECK: [[B:%.*]] = call target("dx.Rawbuffer", <2 x i64>, 1, 0, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_v2i64_1_0_0t(
  ; CHECK-SAME: i32 0, i32 1, i32 1, i32 0, ptr null)
  %buffer = call target("dx.Rawbuffer", <2 x i64>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_v2i64_1_0_0t(
          i32 0, i32 1, i32 1, i32 0, ptr null)

  ; check we don't modify the code in sm6.3 or later
  ; CHECK63: [[L0:%.*]] = call { <2 x i64>, i1 } @llvm.dx.resource.load.rawbuffer
  ; CHECK63-SAME: target("dx.Rawbuffer", <2 x i64>, 1, 0, 0) [[B]], i32 %index, i32 0)

  ; check we load an <4 x i32> instead of a i642
  ; CHECK62: [[L0:%.*]] = call { <4 x i32>, i1 }
  ; CHECK62-SAME: @llvm.dx.resource.load.rawbuffer.v4i32.tdx.Rawbuffer_v2i64_1_0_0t(
  ; CHECK62-SAME: target("dx.Rawbuffer", <2 x i64>, 1, 0, 0) [[B]], i32 %index, i32 0)
  %load0 = call { <2 x i64>, i1 } @llvm.dx.resource.load.rawbuffer(
      target("dx.Rawbuffer", <2 x i64>, 1, 0, 0) %buffer, i32 %index, i32 0)

  ; CHECK63: extractvalue { <2 x i64>, i1 } [[L0]], 0

  ; check we extract the 4 i32 and construct a <2 x i64>
  ; CHECK62: [[D0:%.*]] = extractvalue { <4 x i32>, i1 } [[L0]], 0
  ; CHECK62: [[Lo1:%.*]] = extractelement <4 x i32> [[D0]], i32 0
  ; CHECK62: [[Hi1:%.*]] = extractelement <4 x i32> [[D0]], i32 1
  ; CHECK62: [[Lo2:%.*]] = extractelement <4 x i32> [[D0]], i32 2
  ; CHECK62: [[Hi2:%.*]] = extractelement <4 x i32> [[D0]], i32 3
  ; CHECK62: [[ZLo1:%.*]] = zext i32 [[Lo1]] to i64
  ; CHECK62: [[ZHi1:%.*]] = zext i32 [[Hi1]] to i64
  ; CHECK62: [[A:%.*]] = shl i64 [[ZHi1]], 32
  ; CHECK62: [[B:%.*]] = or i64 [[ZLo1]], [[A]]
  ; CHECK62: [[Vec:%.*]] = insertelement <2 x i64> poison, i64 [[B]], i32 0
  ; CHECK62: [[ZLo2:%.*]] = zext i32 [[Lo2]] to i64
  ; CHECK62: [[ZHi2:%.*]] = zext i32 [[Hi2]] to i64
  ; CHECK62: [[A:%.*]] = shl i64 [[ZHi2]], 32
  ; CHECK62: [[B:%.*]] = or i64 [[ZLo2]], [[A]]
  ; CHECK62: [[Vec2:%.*]] = insertelement <2 x i64> [[Vec]], i64 [[B]], i32 1
  ; CHECK62-NOT: extractvalue { <2 x i64>, i1 }
  %data0 = extractvalue { <2 x i64>, i1 } %load0, 0
  ret void
}

; show we properly handle extracting the check bit
define void @loadi64WithCheckBit(i32 %index) {
  ; check the handle from binding is unchanged
  ; CHECK: [[B:%.*]] = call target("dx.Rawbuffer", i64, 1, 0, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_i64_1_0_0t(
  ; CHECK-SAME: i32 0, i32 1, i32 1, i32 0, ptr null)
  %buffer = call target("dx.Rawbuffer", i64, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_i64_1_0_0t(
          i32 0, i32 1, i32 1, i32 0, ptr null)

  ; check we don't modify the code in sm6.3 or later
  ; CHECK63: [[L0:%.*]] = call { i64, i1 } @llvm.dx.resource.load.rawbuffer
  ; CHECK63-SAME: target("dx.Rawbuffer", i64, 1, 0, 0) [[B]], i32 %index, i32 0)

  ; check we load an <2 x i32> instead of a i64
  ; CHECK62-NOT: call {i64, i1} @llvm.dx.resource.load.rawbuffer
  ; CHECK62: [[L0:%.*]] = call { <2 x i32>, i1 }
  ; CHECK62-SAME: @llvm.dx.resource.load.rawbuffer.v2i32.tdx.Rawbuffer_i64_1_0_0t(
  ; CHECK62-SAME: target("dx.Rawbuffer", i64, 1, 0, 0) [[B]], i32 %index, i32 0)	
  %load0 = call {i64, i1} @llvm.dx.resource.load.rawbuffer(
      target("dx.Rawbuffer", i64, 1, 0, 0) %buffer, i32 %index, i32 0)

  ; CHECK63: extractvalue { i64, i1 } [[L0]], 0
  ; CHECK63: extractvalue { i64, i1 } [[L0]], 1

  ; check we extract the two i32 and construct a i64
  ; CHECK62: [[D0:%.*]] = extractvalue { <2 x i32>, i1 } [[L0]], 0
  ; CHECK62: [[Lo:%.*]] = extractelement <2 x i32> [[D0]], i32 0
  ; CHECK62: [[Hi:%.*]] = extractelement <2 x i32> [[D0]], i32 1
  ; CHECK62: [[ZLo1:%.*]] = zext i32 [[Lo]] to i64
  ; CHECK62: [[ZHi1:%.*]] = zext i32 [[Hi]] to i64
  ; CHECK62: [[A:%.*]] = shl i64 [[ZHi1]], 32
  ; CHECK62: [[B:%.*]] = or i64 [[ZLo1]], [[A]]
  %data0 = extractvalue {i64, i1} %load0, 0
  ; CHECK62: extractvalue { <2 x i32>, i1 } [[L0]], 1
  ; CHECK62-NOT: extractvalue { i64, i1 }
  %cb = extractvalue {i64, i1} %load0, 1
  ret void
}

; Raw Buffer Load allows for i64_t3 and i64_t4 to be loaded
; In SM6.2 and below, two loads will be performed.
; Show we and the checkbits together

define void @loadv3i64(i32 %index) {
  ; check the handle from binding is unchanged
  ; CHECK: [[Buf:%.*]] = call target("dx.Rawbuffer", <3 x i64>, 0, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_v3i64_0_0t(
  ; CHECK-SAME: i32 0, i32 0, i32 1, i32 0, ptr null)
  %buffer = call target("dx.Rawbuffer", <3 x i64>, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_v3i64_0_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; check we don't modify the code in sm6.3 or later
  ; CHECK63: [[L0:%.*]] = call { <3 x i64>, i1 } @llvm.dx.resource.load.rawbuffer
  ; CHECK63-SAME: target("dx.Rawbuffer", <3 x i64>, 0, 0) [[B]], i32 %index, i32 0)

  ; check we perform two loads
  ; and do 6 extracts and construct 3 i64s
  ; CHECK62-NOT: call {<3 x i64>, i1} @llvm.dx.resource.load.rawbuffer
  ; CHECK62: [[L0:%.*]] = call { <4 x i32>, i1 }
  ; CHECK62-SAME: @llvm.dx.resource.load.rawbuffer.v4i32.tdx.Rawbuffer_v3i64_0_0t(
  ; CHECK62-SAME: target("dx.Rawbuffer", <3 x i64>, 0, 0) [[Buf]], i32 %index, i32 0)

  ; CHECK62: [[D0:%.*]] = extractvalue { <4 x i32>, i1 } [[L0]], 0
  ; CHECK62: [[Lo1:%.*]] = extractelement <4 x i32> [[D0]], i32 0
  ; CHECK62: [[Hi1:%.*]] = extractelement <4 x i32> [[D0]], i32 1
  ; CHECK62: [[Lo2:%.*]] = extractelement <4 x i32> [[D0]], i32 2
  ; CHECK62: [[Hi2:%.*]] = extractelement <4 x i32> [[D0]], i32 3
  ; CHECK62: [[ZLo1:%.*]] = zext i32 [[Lo1]] to i64
  ; CHECK62: [[ZHi1:%.*]] = zext i32 [[Hi1]] to i64
  ; CHECK62: [[A:%.*]] = shl i64 [[ZHi1]], 32
  ; CHECK62: [[B:%.*]] = or i64 [[ZLo1]], [[A]]
  ; CHECK62: [[Vec1:%.*]] = insertelement <3 x i64> poison, i64 [[B]], i32 0
  ; CHECK62: [[ZLo2:%.*]] = zext i32 [[Lo2]] to i64
  ; CHECK62: [[ZHi2:%.*]] = zext i32 [[Hi2]] to i64
  ; CHECK62: [[A:%.*]] = shl i64 [[ZHi2]], 32
  ; CHECK62: [[B:%.*]] = or i64 [[ZLo2]], [[A]]
  ; CHECK62: [[Vec2:%.*]] = insertelement <3 x i64> [[Vec1]], i64 [[B]], i32 1

  ; 2nd load
  ; CHECK62: [[L2:%.*]] = call { <2 x i32>, i1 }
  ; CHECK62-SAME: @llvm.dx.resource.load.rawbuffer.v2i32.tdx.Rawbuffer_v3i64_0_0t(
  ; CHECK62-SAME: target("dx.Rawbuffer", <3 x i64>, 0, 0) [[Buf]], i32 %index, i32 16)

  ; CHECK62: [[D2:%.*]] = extractvalue { <2 x i32>, i1 } [[L2]], 0
  ; CHECK62: [[Lo3:%.*]] = extractelement <2 x i32> [[D2]], i32 0
  ; CHECK62: [[Hi3:%.*]] = extractelement <2 x i32> [[D2]], i32 1
  ; CHECK62: [[ZLo3:%.*]] = zext i32 [[Lo3]] to i64
  ; CHECK62: [[ZHi3:%.*]] = zext i32 [[Hi3]] to i64
  ; CHECK62: [[A:%.*]] = shl i64 [[ZHi3]], 32
  ; CHECK62: [[B:%.*]] = or i64 [[ZLo3]], [[A]]
  ; CHECK62: [[Vec3:%.*]] = insertelement <3 x i64> [[Vec2]], i64 [[B]], i32 2
  %load0 = call {<3 x i64>, i1} @llvm.dx.resource.load.rawbuffer(
      target("dx.Rawbuffer", <3 x i64>, 0, 0) %buffer, i32 %index, i32 0)

  ; CHECK63: extractvalue { <3 x i64>, i1 } [[L0]], 0
  ; CHECK63: extractvalue { <3 x i64>, i1 } [[L0]], 1

  ; CHECK62-NOT: extractvalue {<3 x i64>, i1 }
  %data0 = extractvalue {<3 x i64>, i1} %load0, 0
  ; check we extract checkbit from both loads and and them together
  ; CHECK62: [[B1:%.*]] = extractvalue { <4 x i32>, i1 } [[L0]], 1
  ; CHECK62: [[B2:%.*]] = extractvalue { <2 x i32>, i1 } [[L2]], 1
  ; CHECK62: and i1 [[B1]], [[B2]]
  %cb = extractvalue {<3 x i64>, i1} %load0, 1
  ret void
}

define void @loadv4i64(i32 %index) {
  ; check the handle from binding is unchanged
  ; CHECK62: [[Buf:%.*]] = call target("dx.Rawbuffer", <4 x i64>, 0, 0)
  ; CHECK62-SAME: @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_v4i64_0_0t(
  ; CHECK62-SAME: i32 0, i32 0, i32 1, i32 0, ptr null)
  %buffer = call target("dx.Rawbuffer", <4 x i64>, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.Rawbuffer_v4i64_0_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; check we don't modify the code in sm6.3 or later
  ; CHECK63: [[L0:%.*]] = call { <4 x i64>, i1 } @llvm.dx.resource.load.rawbuffer
  ; CHECK63-SAME: target("dx.Rawbuffer", <4 x i64>, 0, 0) [[B]], i32 %index, i32 0)

  ; check we perform two loads
  ; and do 8 extracts and construct 4 i64s
  ; CHECK62-NOT: call {<4 x i64>, i1} @llvm.dx.resource.load.rawbuffer
  ; CHECK62: [[L0:%.*]] = call { <4 x i32>, i1 }
  ; CHECK62-SAME: @llvm.dx.resource.load.rawbuffer.v4i32.tdx.Rawbuffer_v4i64_0_0t(
  ; CHECK62-SAME: target("dx.Rawbuffer", <4 x i64>, 0, 0) [[Buf]], i32 %index, i32 0)

  ; CHECK62: [[D0:%.*]] = extractvalue { <4 x i32>, i1 } [[L0]], 0
  ; CHECK62: [[Lo1:%.*]] = extractelement <4 x i32> [[D0]], i32 0
  ; CHECK62: [[Hi1:%.*]] = extractelement <4 x i32> [[D0]], i32 1
  ; CHECK62: [[Lo2:%.*]] = extractelement <4 x i32> [[D0]], i32 2
  ; CHECK62: [[Hi2:%.*]] = extractelement <4 x i32> [[D0]], i32 3
  ; CHECK62: [[ZLo1:%.*]] = zext i32 [[Lo1]] to i64
  ; CHECK62: [[ZHi1:%.*]] = zext i32 [[Hi1]] to i64
  ; CHECK62: [[A:%.*]] = shl i64 [[ZHi1]], 32
  ; CHECK62: [[B:%.*]] = or i64 [[ZLo1]], [[A]]
  ; CHECK62: [[Vec1:%.*]] = insertelement <4 x i64> poison, i64 [[B]], i32 0
  ; CHECK62: [[ZLo2:%.*]] = zext i32 [[Lo2]] to i64
  ; CHECK62: [[ZHi2:%.*]] = zext i32 [[Hi2]] to i64
  ; CHECK62: [[A:%.*]] = shl i64 [[ZHi2]], 32
  ; CHECK62: [[B:%.*]] = or i64 [[ZLo2]], [[A]]
  ; CHECK62: [[Vec2:%.*]] = insertelement <4 x i64> [[Vec1]], i64 [[B]], i32 1

  ; 2nd load
  ; CHECK62: [[L2:%.*]] = call { <4 x i32>, i1 }
  ; CHECK62-SAME: @llvm.dx.resource.load.rawbuffer.v4i32.tdx.Rawbuffer_v4i64_0_0t(
  ; CHECK62-SAME: target("dx.Rawbuffer", <4 x i64>, 0, 0) [[Buf]], i32 %index, i32 16)

  ; CHECK62: [[D2:%.*]] = extractvalue { <4 x i32>, i1 } [[L2]], 0
  ; CHECK62: [[Lo3:%.*]] = extractelement <4 x i32> [[D2]], i32 0
  ; CHECK62: [[Hi3:%.*]] = extractelement <4 x i32> [[D2]], i32 1
  ; CHECK62: [[Lo4:%.*]] = extractelement <4 x i32> [[D2]], i32 2
  ; CHECK62: [[Hi4:%.*]] = extractelement <4 x i32> [[D2]], i32 3
  ; CHECK62: [[ZLo3:%.*]] = zext i32 [[Lo3]] to i64
  ; CHECK62: [[ZHi3:%.*]] = zext i32 [[Hi3]] to i64
  ; CHECK62: [[A:%.*]] = shl i64 [[ZHi3]], 32
  ; CHECK62: [[B:%.*]] = or i64 [[ZLo3]], [[A]]
  ; CHECK62: [[Vec3:%.*]] = insertelement <4 x i64> [[Vec2]], i64 [[B]], i32 2
  ; CHECK62: [[ZLo4:%.*]] = zext i32 [[Lo4]] to i64
  ; CHECK62: [[ZHi4:%.*]] = zext i32 [[Hi4]] to i64
  ; CHECK62: [[A:%.*]] = shl i64 [[ZHi4]], 32
  ; CHECK62: [[B:%.*]] = or i64 [[ZLo4]], [[A]]
  ; CHECK62: [[Vec4:%.*]] = insertelement <4 x i64> [[Vec3]], i64 [[B]], i32 3
  %load0 = call {<4 x i64>, i1} @llvm.dx.resource.load.rawbuffer(
      target("dx.Rawbuffer", <4 x i64>, 0, 0) %buffer, i32 %index, i32 0)

  ; CHECK63: extractvalue { <4 x i64>, i1 } [[L0]], 0
  ; CHECK63: extractvalue { <4 x i64>, i1 } [[L0]], 1

  ; CHECK62-NOT: extractvalue {<4 x i64>, i1 }
  %data0 = extractvalue {<4 x i64>, i1} %load0, 0
  ; check we extract checkbit from both loads and and them together
  ; CHECK62: [[B1:%.*]] = extractvalue { <4 x i32>, i1 } [[L0]], 1
  ; CHECK62: [[B2:%.*]] = extractvalue { <4 x i32>, i1 } [[L2]], 1
  ; CHECK62: and i1 [[B1]], [[B2]]
  %cb = extractvalue {<4 x i64>, i1} %load0, 1
  ret void
}
