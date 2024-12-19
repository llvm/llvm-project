; RUN: llc -mtriple=riscv32 -filetype=obj < %s | llvm-readelf -h - | FileCheck -check-prefixes=FLAGS %s

; FLAGS: Flags: 0x11, RVC, TSO

define i32 @addi(i32 %a) {
  %1 = add i32 %a, 1
  ret i32 %1
}

!llvm.module.flags = !{!0}

!0 = !{i32 6, !"riscv-isa", !1}
!1 = !{!"rv64i2p1_c2p0_ztso1p0"}
