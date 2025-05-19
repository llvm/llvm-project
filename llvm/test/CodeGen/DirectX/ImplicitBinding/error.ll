; RUN: not opt -S -dxil-resource-implicit-binding %s 2>&1 | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

@A.str = private unnamed_addr constant [2 x i8] c"A\00", align 1
@B.str = private unnamed_addr constant [2 x i8] c"B\00", align 1
@C.str = private unnamed_addr constant [2 x i8] c"C\00", align 1

define void @test_simple_binding() {

; StructuredBuffer<float> A[] : register(t1);
  %bufA = call target("dx.RawBuffer", float, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 1, i32 -1, i32 0, i1 false, ptr @A.str)

; StructuredBuffer<float> B[2]; // does not fit in space0
  %bufB = call target("dx.RawBuffer", float, 0, 0)
      @llvm.dx.resource.handlefromimplicitbinding(i32 100, i32 0, i32 2, i32 0, i1 false, ptr @B.str)

; StructuredBuffer<float> C; // ok
  %bufC = call target("dx.RawBuffer", float, 0, 0)
      @llvm.dx.resource.handlefromimplicitbinding(i32 200, i32 0, i32 1, i32 0, i1 false, ptr @C.str)

; CHECK: error:{{.*}} resource B cannot be allocated

  ret void
}
