; RUN: not opt -S %s 2>&1 | FileCheck %s

declare void @llvm.amdgcn.kill(i1)

; CHECK: Callbr amdgcn_kill only supports one indirect dest
define void @test_callbr_intrinsic_indirect0(i1 %c) {
  callbr void @llvm.amdgcn.kill(i1 %c) to label %cont []
kill:
  unreachable
cont:
  ret void
}

; CHECK-NEXT: Callbr amdgcn_kill only supports one indirect dest
define void @test_callbr_intrinsic_indirect2(i1 %c) {
  callbr void @llvm.amdgcn.kill(i1 %c) to label %cont [label %kill1, label %kill2]
kill1:
  unreachable
kill2:
  unreachable
cont:
  ret void
}

; CHECK-NEXT: Callbr amdgcn_kill indirect dest needs to be unreachable
define void @test_callbr_intrinsic_no_unreachable(i1 %c) {
  callbr void @llvm.amdgcn.kill(i1 %c) to label %cont [label %kill]
kill:
  ret void
cont:
  ret void
}

; CHECK-NEXT: Callbr currently only supports asm-goto and selected intrinsics
declare i32 @llvm.amdgcn.workitem.id.x()
define void @test_callbr_intrinsic_unsupported() {
  callbr i32 @llvm.amdgcn.workitem.id.x() to label %cont []
cont:
  ret void
}

; CHECK-NEXT: Callbr: indirect function / invalid signature
define void @test_callbr_intrinsic_wrong_signature(ptr %ptr) {
  %func = load ptr, ptr %ptr, align 8
  callbr void %func() to label %cont []
cont:
  ret void
}

; CHECK-NEXT: Callbr for intrinsics currently doesn't support operand bundles
define void @test_callbr_intrinsic_no_operand_bundles(i1 %c) {
  callbr void @llvm.amdgcn.kill(i1 %c) [ "foo"(i1 %c) ] to label %cont [label %kill]
kill:
  unreachable
cont:
  ret void
}
