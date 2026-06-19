; RUN: not opt -passes=verify -S < %s 2>&1 | FileCheck %s

; AMX exposes 8 physical tile registers (TMM0-TMM7), so a tile-register index
; operand must be in the range [0, 8). This is a structural constraint on the
; immediate operand and needs no subtarget information.

declare void @llvm.x86.tilezero(i8 immarg)
declare void @llvm.x86.tileloadd64(i8 immarg, ptr, i64)
declare void @llvm.x86.tdpbssd(i8 immarg, i8 immarg, i8 immarg)

define void @bad_tilezero() {
; CHECK: AMX tile register index must be in the range [0, 8)
  call void @llvm.x86.tilezero(i8 8)
  ret void
}

; Only one of the three tile operands is out of range.
define void @bad_tdpbssd() {
; CHECK: AMX tile register index must be in the range [0, 8)
  call void @llvm.x86.tdpbssd(i8 0, i8 1, i8 9)
  ret void
}

; A valid tile index (0) with a large *constant* stride. The stride is operand 2,
; not a tile-register index, so it must not be diagnosed.
; CHECK-NOT: AMX tile register index
define void @good_tileloadd64(ptr %p) {
  call void @llvm.x86.tileloadd64(i8 0, ptr %p, i64 64)
  ret void
}
