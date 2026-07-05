; RUN: llc -mtriple=riscv64 < %s -o - | FileCheck --check-prefixes=CHECK-ATTRIBUTES %s
; RUN: llc -mtriple=riscv64 < %s -filetype=obj | llvm-readelf -h - \
; RUN:     | FileCheck --check-prefixes=CHECK-EFLAGS %s

; CHECK-ATTRIBUTES: .attribute      5, "rv64i2p1"
; CHECK-EFLAGS: Flags: 0x0
define void @test() {
  tail call void asm ".option arch, +c", ""()
  ret void
}
