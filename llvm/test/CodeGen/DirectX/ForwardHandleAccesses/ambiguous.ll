; RUN: not opt -S -dxil-forward-handle-accesses -mtriple=dxil--shadermodel6.3-library %s 2>&1 | FileCheck %s

; CHECK: error: Handle at "h2" overwrites handle at "h1"

%"class.hlsl::RWStructuredBuffer" = type { target("dx.RawBuffer", <4 x float>, 1, 0) }
@Buf = internal global %"class.hlsl::RWStructuredBuffer" poison, align 4

define float @f() {
entry:
  %h1 = call target("dx.RawBuffer", <4 x float>, 1, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  store target("dx.RawBuffer", <4 x float>, 1, 0) %h1, ptr @Buf, align 4
  %h2 = call target("dx.RawBuffer", <4 x float>, 1, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 1, i32 1, i32 0, ptr null)
  store target("dx.RawBuffer", <4 x float>, 1, 0) %h2, ptr @Buf, align 4

  %b = load target("dx.RawBuffer", <4 x float>, 1, 0), ptr @Buf, align 4
  %l = call { <4 x float>, i1 } @llvm.dx.resource.load.rawbuffer(target("dx.RawBuffer", <4 x float>, 1, 0) %b, i32 0, i32 0)
  %x = extractvalue { <4 x float>, i1 } %l, 0
  %v = extractelement <4 x float> %x, i32 0

  ret float %v
}
