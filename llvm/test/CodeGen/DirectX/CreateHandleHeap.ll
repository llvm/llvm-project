; RUN: opt -S -passes=dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

declare i32 @some_val();

define void @test() {
  ; RWBuffer<float4> Buf = ResourceDescriptorHeap[3]
  %typed = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
              @llvm.dx.resource.handlefromheap.tdx.TypedBuffer_v4f32_1_0_0(i32 3, i1 false)
  ; CHECK: [[HANDLE0:%.*]] = call %dx.types.Handle @dx.op.createHandleFromHeap(i32 218, i32 3, i1 false, i1 false)
  ; CHECK: [[ANNOT_HANDLE0:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[HANDLE0]], %dx.types.ResourceProperties { i32 4106, i32 1033 })

  ; int val = some_val();
  ; RWBuffer<float4> Buf = ResourceDescriptorHeap[val]
  %val = call i32 @some_val()
  %typed1 = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
              @llvm.dx.resource.handlefromheap.tdx.TypedBuffer_v4f32_1_0_0(i32 %val, i1 false)
  ; CHECK: [[HANDLE1:%.*]] = call %dx.types.Handle @dx.op.createHandleFromHeap(i32 218, i32 %val, i1 false, i1 false)
  ; CHECK: [[ANNOT_HANDLE1:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[HANDLE1]], %dx.types.ResourceProperties { i32 4106, i32 1033 })

  ; SamplerState Samp = SamplerDescriptorHeap[val];
  %samp = call target("dx.Sampler", 0)
              @llvm.dx.resource.handlefromheap.tdx.Sampler_0(i32 %val, i1 true)
  ; CHECK: [[HANDLE2:%.*]] = call %dx.types.Handle @dx.op.createHandleFromHeap(i32 218, i32 %val, i1 true, i1 false)
  ; CHECK: [[ANNOT_HANDLE2:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[HANDLE2]], %dx.types.ResourceProperties { i32 14, i32 0 })

  ret void
}

; CHECK: declare %dx.types.Handle @dx.op.createHandleFromHeap(i32, i32, i1, i1)
; CHECK: declare %dx.types.Handle @dx.op.annotateHandle(i32, %dx.types.Handle, %dx.types.ResourceProperties)
