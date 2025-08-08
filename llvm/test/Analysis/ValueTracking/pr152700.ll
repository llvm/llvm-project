; Check that we do not crash (see PR #152700)
; RUN: opt < %s -passes=instcombine

declare noundef i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
declare i32 @llvm.umin.i32(i32, i32)
define i32 @foo(i1 %c, i32 %arg) {
entry:
  %i = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  br i1 %c, label %bb.1, label %bb.2
bb.1:
  br label %bb.2
bb.2:
  %phi = phi i32 [ %i, %entry ], [ 0, %bb.1 ]
  %res = call i32 @llvm.umin.i32(i32 %phi, i32 %arg)
  ret i32 %res
}
