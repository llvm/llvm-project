; RUN: not opt -S -dxil-forward-handle-accesses -mtriple=dxil--shadermodel6.3-library %s 2>&1 | FileCheck %s

; CHECK: error: Load of "buf" is not a global resource handle

%"class.hlsl::RWStructuredBuffer" = type { target("dx.RawBuffer", <4 x float>, 1, 0) }
@Buf = internal global %"class.hlsl::RWStructuredBuffer" poison, align 4

define float @f() {
entry:
  %buf = alloca target("dx.RawBuffer", <4 x float>, 1, 0), align 4
  %h = call target("dx.RawBuffer", <4 x float>, 1, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, i1 false)
  store target("dx.RawBuffer", <4 x float>, 1, 0) %h, ptr %buf, align 4

  %b = load target("dx.RawBuffer", <4 x float>, 1, 0), ptr %buf, align 4
  %l = call { <4 x float>, i1 } @llvm.dx.resource.load.rawbuffer(target("dx.RawBuffer", <4 x float>, 1, 0) %b, i32 0, i32 0)
  %x = extractvalue { <4 x float>, i1 } %l, 0
  %v = extractelement <4 x float> %x, i32 0

  ret float %v
}
