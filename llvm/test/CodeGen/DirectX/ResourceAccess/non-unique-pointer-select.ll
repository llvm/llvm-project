; RUN: not opt -S -dxil-resource-access -mtriple=dxil--shadermodel6.3-library %s 2>&1 | FileCheck %s

; Regression test for llvm/llvm-project#224422. Once the pass diagnoses the
; non-unique resource access, compilation must terminate before pointer
; lowering encounters the select.

; CHECK: note: At resource access:  %value = load float, ptr %ptr, align 4
; CHECK-DAG: note: Uses resource handle:  %handle0 = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 0, i32 0, i32 1, i32 0, ptr @BufferA.str)
; CHECK-DAG: note: Uses resource handle:  %handle1 = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 0, i32 1, i32 1, i32 0, ptr @BufferB.str)
; CHECK: LLVM ERROR: Resource access is not guaranteed to map to a unique global resource
; CHECK-NOT: Unhandled instruction - pointer escaped

@BufferA.str = internal unnamed_addr constant [8 x i8] c"BufferA\00"
@BufferB.str = internal unnamed_addr constant [8 x i8] c"BufferB\00"

define float @pointer_select(i1 %cond) {
entry:
  %handle0 = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 0, i32 0, i32 1, i32 0, ptr @BufferA.str)
  %handle1 = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 0, i32 1, i32 1, i32 0, ptr @BufferB.str)
  %ptr0 = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %handle0, i32 0)
  %ptr1 = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %handle1, i32 1)
  %ptr = select i1 %cond, ptr %ptr0, ptr %ptr1
  %value = load float, ptr %ptr, align 4
  ret float %value
}
