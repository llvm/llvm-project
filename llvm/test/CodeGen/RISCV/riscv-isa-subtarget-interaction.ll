; RUN: llc < %s -mtriple=riscv32 -mattr=+zca,+zcb,+zcmp,+zcmt -filetype=obj --filetype=obj -o /dev/null

target triple = "riscv32-unknown-elf"

define void @f() {
  ret void
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"target-abi", !"ilp32d"}
!1 = !{i32 6, !"riscv-isa", !2}
!2 = !{!"rv32i2p1_f2p2_d2p2_zca1p0_zcb1p0_zcf1p0_zcmp1p0_zcmt1p0"}
