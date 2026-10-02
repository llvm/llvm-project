; RUN: llc < %s -mtriple=riscv32 -mattr=+experimental-xsfmclic \
; RUN:   -verify-machineinstrs -stop-after=prolog-epilog -o - \
; RUN:   | FileCheck %s
; RUN: llc < %s -mtriple=riscv64 -mattr=+experimental-xsfmclic \
; RUN:   -verify-machineinstrs -stop-after=prolog-epilog -o - \
; RUN:   | FileCheck %s

define void @preemptible_stack_swap() "interrupt"="SiFive-CLIC-preemptible-stack-swap" {
; CHECK-LABEL: name: preemptible_stack_swap
; CHECK: body:             |
; CHECK-NEXT:   bb.0
; CHECK-NEXT:     liveins: $x5
; CHECK:          $x2 = frame-setup CSRRW 840, killed $x2
; CHECK-NEXT:     $x2 = frame-setup ADDI $x2, -{{16|32}}
; CHECK-NEXT:     frame-setup CFI_INSTRUCTION def_cfa_offset {{16|32}}
; CHECK-NEXT:     frame-setup {{SW|SD}} killed $x5
; CHECK-NEXT:     frame-setup CFI_INSTRUCTION offset $x5, -{{12|24}}
; CHECK-NEXT:     $x5 = frame-setup CSRRS 834, $x0
; CHECK-NEXT:     frame-setup {{SW|SD}} killed $x5
; CHECK-NEXT:     $x5 = frame-setup CSRRS 833, $x0
; CHECK-NEXT:     $x0 = frame-setup CSRRSI 768, 8
; CHECK-NEXT:     frame-setup {{SW|SD}} killed $x5
; CHECK-NEXT:     $x5 = frame-destroy {{LW|LD}}
; CHECK-NEXT:     $x0 = frame-destroy CSRRCI 768, 8
; CHECK-NEXT:     $x0 = frame-destroy CSRRW 833, killed $x5
; CHECK-NEXT:     $x5 = frame-destroy {{LW|LD}}
; CHECK-NEXT:     $x0 = frame-destroy CSRRW 834, killed $x5
; CHECK-NEXT:     $x5 = frame-destroy {{LW|LD}}
; CHECK-NEXT:     frame-destroy CFI_INSTRUCTION restore $x5
; CHECK-NEXT:     $x2 = frame-destroy ADDI $x2, {{16|32}}
; CHECK-NEXT:     frame-destroy CFI_INSTRUCTION def_cfa_offset 0
; CHECK-NEXT:     $x2 = frame-destroy CSRRW 840, killed $x2
; CHECK-NEXT:     MRET
  ret void
}
