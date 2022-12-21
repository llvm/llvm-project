; RUN: llc -mtriple=mips-mti-linux-gnu -mcpu=mips32r2 -mattr=+micromips \
; RUN:     -stop-after=finalize-isel < %s | FileCheck --check-prefix=MM2 %s
; RUN: llc -mtriple=mips-mti-linux-gnu -mcpu=mips32r6 -mattr=+micromips \
; RUN:     -stop-after=finalize-isel < %s | FileCheck --check-prefix=MM6 %s

; MM2: JAL_MM @bar
; MM2: JAL_MM &memset
; MM2-NOT: JALR16_MM

; MM6: JAL_MMR6 @bar
; MM6: JAL_MMR6 &memset
; MM6-NOT: JALRC16_MMR6

define dso_local void @foo(ptr nocapture %ar) local_unnamed_addr {
entry:
  call void @bar()
  tail call void @llvm.memset.p0.i32(ptr align 4 %ar, i8 0, i32 100, i1 false)
  ret void
}

declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1)
declare void @bar()
