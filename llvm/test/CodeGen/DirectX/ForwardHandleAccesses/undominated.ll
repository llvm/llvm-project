; RUN: not opt -S -dxil-forward-handle-accesses -mtriple=dxil--shadermodel6.3-library %s 2>&1 | FileCheck %s

; CHECK: error: Load at "b" is not dominated by handle creation at "h1"

%"class.hlsl::RWStructuredBuffer" = type { target("dx.RawBuffer", <4 x float>, 1, 0) }
@Buf = internal global %"class.hlsl::RWStructuredBuffer" poison, align 4

define void @f() {
entry:
  %b = load target("dx.RawBuffer", <4 x float>, 1, 0), ptr @Buf, align 4

  %h1 = call target("dx.RawBuffer", <4 x float>, 1, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  store target("dx.RawBuffer", <4 x float>, 1, 0) %h1, ptr @Buf, align 4

  ret void
}
