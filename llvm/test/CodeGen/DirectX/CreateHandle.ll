; RUN: opt -S -passes=dxil-translate-metadata,dxil-op-lower %s | FileCheck %s
; RUN: opt -S -passes=dxil-pretty-printer %s 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PRETTY %s

; CHECK-PRETTY:       Type  Format         Dim      ID      HLSL Bind     Count
; CHECK-PRETTY: ---------- ------- ----------- ------- -------------- ---------
; CHECK-PRETTY:        SRV     f32         buf      T0      t7        unbounded
; CHECK-PRETTY:        SRV    byte         r/o      T1      t8,space1         1
; CHECK-PRETTY:        SRV  struct         r/o      T2      t2,space4         1
; CHECK-PRETTY:        SRV     u32         buf      T3      t3,space5        24
; CHECK-PRETTY:        UAV     i32         buf      U0      u7,space2         1
; CHECK-PRETTY:        UAV     f32         buf      U1      u5,space3         1

target triple = "dxil-pc-shadermodel6.0-compute"

declare i32 @some_val();

define void @test_buffers() {
  ; RWBuffer<float4> Buf : register(u5, space3)
  %typed0 = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
              @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v4f32_1_0_0(
                  i32 3, i32 5, i32 1, i32 0, i1 false)
  ; CHECK: call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 1, i32 5, i1 false) #[[#ATTR:]]
  ; CHECK-NOT: @llvm.dx.cast.handle

  ; RWBuffer<int> Buf : register(u7, space2)
  %typed1 = call target("dx.TypedBuffer", i32, 1, 0, 1)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0_1t(
          i32 2, i32 7, i32 1, i32 0, i1 false)
  ; CHECK: call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 0, i32 7, i1 false) #[[#ATTR]]

  ; Buffer<uint4> Buf[24] : register(t3, space5)
  ; Buffer<uint4> typed2 = Buf[4]
  ; Note that the index below is 3 + 4 = 7
  %typed2 = call target("dx.TypedBuffer", <4 x i32>, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_0_0_0t(
          i32 5, i32 3, i32 24, i32 4, i1 false)
  ; CHECK: call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 3, i32 7, i1 false) #[[#ATTR]]

  ; struct S { float4 a; uint4 b; };
  ; StructuredBuffer<S> Buf : register(t2, space4)
  %struct0 = call target("dx.RawBuffer", {<4 x float>, <4 x i32>}, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_sl_v4f32v4i32s_0_0t(
          i32 4, i32 2, i32 1, i32 0, i1 true)
  ; CHECK: call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 2, i32 2, i1 true) #[[#ATTR]]

  ; ByteAddressBuffer Buf : register(t8, space1)
  %byteaddr0 = call target("dx.RawBuffer", i8, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0t(
          i32 1, i32 8, i32 1, i32 0, i1 false)
  ; CHECK: call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 1, i32 8, i1 false) #[[#ATTR]]

  ; Buffer<float4> Buf[] : register(t7)
  ; Buffer<float4> typed3 = Buf[ix]
  %typed3_ix = call i32 @some_val()
  %typed3 = call target("dx.TypedBuffer", <4 x float>, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v4f32_0_0_0t(
          i32 0, i32 7, i32 -1, i32 %typed3_ix, i1 false)
  ; CHECK: %[[IX:.*]] = add i32 %typed3_ix, 7
  ; CHECK: call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 0, i32 %[[IX]], i1 false) #[[#ATTR]]

  ret void
}

; CHECK: attributes #[[#ATTR]] = {{{.*}} memory(read) {{.*}}}

; Just check that we have the right types and number of metadata nodes, the
; contents of the metadata are tested elsewhere.
;
; CHECK: !dx.resources = !{[[RESMD:![0-9]+]]}
; CHECK: [[RESMD]] = !{[[SRVMD:![0-9]+]], [[UAVMD:![0-9]+]], null, null}
; CHECK-DAG: [[SRVMD]] = !{!{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}}
; CHECK-DAG: [[UAVMD]] = !{!{{[0-9]+}}, !{{[0-9]+}}}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }
