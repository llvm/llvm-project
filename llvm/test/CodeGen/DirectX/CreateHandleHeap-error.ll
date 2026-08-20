; RUN: not opt -S -passes=verify %s -disable-output 2>&1 | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK: immarg operand has non-immediate parameter
; CHECK: error: input module is broken!

declare i1 @some_val();

define void @test() {
  %val = call i1 @some_val()
  %handle = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
              @llvm.dx.resource.handlefromheap.tdx.TypedBuffer_v4f32_1_0_0(i32 3, i1 %val)

  ret void
}
