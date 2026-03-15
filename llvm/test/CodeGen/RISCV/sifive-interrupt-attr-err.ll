; RUN: not llc -mtriple riscv32-unknown-elf -mattr=+experimental-xsfmclic -o - %s 2>&1 \
; RUN:   | FileCheck %s
; RUN: not llc -mtriple riscv64-unknown-elf -mattr=+experimental-xsfmclic -o - %s 2>&1 \
; RUN:   | FileCheck %s

;; Test that these report fatal errors.

; CHECK: LLVM ERROR: 'SiFive-CLIC-preemptible' interrupt kinds cannot have a frame pointer

define void @preemptible() "interrupt"="SiFive-CLIC-preemptible" "frame-pointer"="all" {
  ret void
}
