; RUN: opt -S -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

define i32 @test_getdimensions_no_mips() {
  ; CHECK: %[[HANDLE:.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, 
  ; CHECK-NEXT: %[[ANNOT_HANDLE:.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %[[HANDLE]]
  %handle = call target("dx.TypedBuffer", <4 x float>, 0, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NEXT: %[[RETVAL:.*]] = call %dx.types.Dimensions @dx.op.getDimensions(i32 72, %dx.types.Handle %[[ANNOT_HANDLE]], i32 undef)
  ; CHECK-NEXT: %[[DIM:.*]] = extractvalue %dx.types.Dimensions %[[RETVAL]], 0
  %1 = call i32 @llvm.dx.resource.getdimensions.x(target("dx.TypedBuffer", <4 x float>, 0, 0, 0) %handle)
  
  ; CHECK-NEXT: ret i32 %[[DIM]]
  ret i32 %1
}
