; RUN: llvm-as < %s | llvm-dis | FileCheck %s

declare amdgpu_cs_chain void @callee() nounwind
declare void @llvm.amdgcn.cs.chain.p0.i64.i32.i32(ptr, i64, i32, i32, i32 immarg, ...)
declare void @llvm.amdgcn.unreachable()
declare void @llvm.dbg.value(metadata, metadata, metadata)

; @llvm.amdgcn.unreachable is legal after @llvm.amdgcn.cs.chain
; CHECK: define amdgpu_cs_chain void @test_cc_chain_unreachable(i32 %val)
define amdgpu_cs_chain void @test_cc_chain_unreachable(i32 %val) {
tail.block:
  %.cond = icmp ne i32 %val, 0
  br i1 %.cond, label %chain.block, label %UnifiedReturnBlock

chain.block:
  call void (ptr, i64, i32, i32, i32, ...) @llvm.amdgcn.cs.chain.p0.i64.i32.i32(ptr @callee, i64 -1, i32 inreg 1, i32 2, i32 1, i32 inreg 32, i32 inreg -1, ptr @callee)
  call void @llvm.amdgcn.unreachable()
  br label %UnifiedReturnBlock

UnifiedReturnBlock:
  ret void
}

; debug instructions should be ignored
; CHECK: define amdgpu_cs_chain void @test_cc_chain_unreachable_debug(i32 %val)
define amdgpu_cs_chain void @test_cc_chain_unreachable_debug(i32 %val) {
init:
  call void (ptr, i64, i32, i32, i32, ...) @llvm.amdgcn.cs.chain.p0.i64.i32.i32(ptr @callee, i64 -1, i32 inreg 1, i32 2, i32 1, i32 inreg 32, i32 inreg -1, ptr @callee)
  call void @llvm.dbg.value(metadata i32 0, metadata !{}, metadata !DIExpression())
  call void @llvm.amdgcn.unreachable()
  ret void
}
