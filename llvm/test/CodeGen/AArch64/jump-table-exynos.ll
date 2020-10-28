; RUN: llc -o - %s -mtriple=aarch64-none-linux-gnu -mattr=+force-32bit-jump-tables -aarch64-enable-atomic-cfg-tidy=0 | FileCheck %s
; RUN: llc -o - %s -mtriple=aarch64-none-linux-gnu -mcpu=exynos-m3 -aarch64-enable-atomic-cfg-tidy=0 | FileCheck %s

; Exynos doesn't want jump tables to be compressed for now.

define i32 @test_jumptable(i32 %in)  {
  switch i32 %in, label %def [
    i32 0, label %lbl1
    i32 1, label %lbl2
    i32 2, label %lbl3
    i32 4, label %lbl4
  ]
; CHECK-LABEL: test_jumptable:
; CHECK:     adrp [[JTPAGE:x[0-9]+]], .LJTI0_0
; CHECK:     add x[[JT:[0-9]+]], [[JTPAGE]], {{#?}}:lo12:.LJTI0_0
; CHECK:  [[PCREL_LBL:.Ltmp.*]]:
; CHECK-NEXT: adr [[PCBASE:x[0-9]+]], [[PCREL_LBL]]
; CHECK:     ldrsw x[[OFFSET:[0-9]+]], [x[[JT]], {{x[0-9]+}}, lsl #2]
; CHECK:     add [[DEST:x[0-9]+]], [[PCBASE]], x[[OFFSET]]
; CHECK:     br [[DEST]]


; CHECK: .LJTI0_0:
; CHECK-NEXT:     .word .LBB{{.*}}-[[PCREL_LBL]]

def:
  ret i32 0

lbl1:
  ret i32 1

lbl2:
  ret i32 2

lbl3:
  ret i32 4

lbl4:
  ret i32 8

}

define i32 @test_jumptable_minsize(i32 %in) minsize {
  switch i32 %in, label %def [
    i32 0, label %lbl1
    i32 1, label %lbl2
    i32 2, label %lbl3
    i32 4, label %lbl4
  ]
; CHECK-LABEL: test_jumptable_minsize:
; CHECK:     adrp [[JTPAGE:x[0-9]+]], .LJTI1_0
; CHECK:     add x[[JT:[0-9]+]], [[JTPAGE]], {{#?}}:lo12:.LJTI1_0
; CHECK:     adr [[PCBASE:x[0-9]+]], [[JTBASE:.LBB[0-9]+_[0-9]+]]
; CHECK:     ldrb w[[OFFSET:[0-9]+]], [x[[JT]], {{x[0-9]+}}]
; CHECK:     add [[DEST:x[0-9]+]], [[PCBASE]], x[[OFFSET]], lsl #2
; CHECK:     br [[DEST]]



def:
  ret i32 0

lbl1:
  ret i32 1

lbl2:
  ret i32 2

lbl3:
  ret i32 4

lbl4:
  ret i32 8

}
