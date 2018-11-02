; RUN: llc -mtriple=mips-mti-linux-gnu -mcpu=mips32r2 -mattr=+micromips -stop-after=expand-isel-pseudos < %s | FileCheck %s

; CHECK: JAL_MM
; CHECK-NOT: JALR16_MM

define dso_local void @foo(i32* nocapture %ar) local_unnamed_addr {
entry:
  %0 = bitcast i32* %ar to i8*
  tail call void @llvm.memset.p0i8.i32(i8* align 4 %0, i8 0, i32 100, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i32(i8* nocapture writeonly, i8, i32, i1)
