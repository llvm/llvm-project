; RUN: llc < %s -march=nvptx64 | FileCheck %s
declare i32 @get_register()

define i1 @test1() {
entry:
  %call = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !0
  %cmp = icmp eq i32 %call, 1
  ret i1 %cmp
}

; CHECK-LABEL: test1(
; CHECK: setp.eq.s32  %p1, %r1, 1;
; CHECK: selp.u32     %[[R:.+]], 1, 0, %p1;
; CHECK: st.param.b32 [func_retval0+0], %[[R]];

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!0 = !{ i32 0, i32 3 }
