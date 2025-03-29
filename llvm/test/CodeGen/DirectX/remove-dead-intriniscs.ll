
; RUN: llc %s -mtriple=dxil-pc-shadermodel6.3-library --filetype=asm -o - | FileCheck %s

declare void @llvm.lifetime.start.p0(i64, ptr) #1
declare void @llvm.lifetime.end.p0(i64, ptr) #1
declare i32 @llvm.dx.udot.v4i32(<4 x i32>, <4 x i32>) #2
declare void @llvm.memset.p0.i32(ptr, i8, i32, i1) #3

; CHECK-NOT: declare void @llvm.lifetime.start.p0(i64, ptr)
; CHECK-NOT: declare void @llvm.lifetime.end.p0(i64, ptr)
; CHECK-NOT: declare i32 @llvm.dx.udot.v4i32(<4 x i32>, <4 x i32>)
; CHECK-NOT: declare void @llvm.memset.p0.i32(ptr, i8, i32, i1)

; CHECK-LABEL: empty_fn
define void @empty_fn () local_unnamed_addr #0 {
    ret void
 } 

attributes #0 = { convergent norecurse nounwind "hlsl.export"}
attributes #1 = { nounwind memory(argmem: readwrite) }
attributes #2 = { nounwind memory(none) }
attributes #3 = { nounwind memory(argmem: write) }
