; RUN: not opt -S -dxil-resource-access -mtriple=dxil--shadermodel6.3-library %s 2>&1 | FileCheck %s

; CHECK: note: At resource access:  %count = call i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %handle, i8 1)
; CHECK-DAG: note: Uses resource handle:  %handle0 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str.2)
; CHECK-DAG: note: Uses resource handle:  %handle1 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 2, i32 1, i32 0, ptr nonnull @.str.4)
; CHECK: LLVM ERROR: Resource access is not guaranteed to map to a unique global resource

@.str.2 = internal unnamed_addr constant [5 x i8] c"Out0\00", align 1
@.str.4 = internal unnamed_addr constant [5 x i8] c"Out1\00", align 1

define i32 @updatecounter_select(i1 %cond) {
entry:
  %handle0 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str.2)
  %handle1 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 2, i32 1, i32 0, ptr nonnull @.str.4)
  %handle = select i1 %cond, target("dx.RawBuffer", i32, 1, 0) %handle0, target("dx.RawBuffer", i32, 1, 0) %handle1
  %count = call i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %handle, i8 1)
  ret i32 %count
}
