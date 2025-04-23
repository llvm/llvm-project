; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

define void @test_ret_popless_not_musttail() {
; CHECK: llvm.ret.popless call must be musttail
  call void @llvm.ret.popless()
  ret void
}

define i64 @test_ret_popless_not_returned(i64 %a) {
; CHECK: musttail intrinsic call must precede a ret
  musttail call void @llvm.ret.popless()
  %res = bitcast i64 %a to i64
  ret i64 %res
}
