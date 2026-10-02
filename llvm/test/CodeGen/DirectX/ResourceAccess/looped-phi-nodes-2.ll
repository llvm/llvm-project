; RUN: not opt -S -dxil-resource-type -dxil-resource-access -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

@.str = private unnamed_addr constant [5 x i8] c"bufA\00", align 1
@.str.2 = private unnamed_addr constant [5 x i8] c"bufB\00", align 1

; Ensure that a cyclic loop of resource ptrs reports a fatal error and exits
; compilation, rather than hanging or crashing later on the illegal access.

; CHECK: note: At resource access:  %count = call i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %src, i8 1)
; CHECK-DAG: note: Uses resource handle:  %handle0 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
; CHECK-DAG: note: Uses resource handle:  %handle1 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str.2)
; CHECK: LLVM ERROR: Resource access is not guaranteed to map to a unique global resource

define i32 @updatecounter_loop(i32 %n) {
entry:
  %handle0 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %handle1 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str.2)
  br label %loop

loop:
  %dst = phi target("dx.RawBuffer", i32, 1, 0) [ %handle1, %entry ], [ %src, %loop ]
  %src = phi target("dx.RawBuffer", i32, 1, 0) [ %handle0, %entry ], [ %dst, %loop ]
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %count = call i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %src, i8 1)
  %inc = add nuw i32 %i, 1
  %exit = icmp eq i32 %inc, %n
  br i1 %exit, label %end, label %loop

end:
  ret i32 %count
}
