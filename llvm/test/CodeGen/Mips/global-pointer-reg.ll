; DISABLED: llc < %s -march=mipsel -mips-fix-global-base-reg=false | FileCheck %s 
; RUN: false
; XFAIL: *

@g0 = external global i32
@g1 = external global i32
@g2 = external global i32

define void @foo1() nounwind {
entry:
; CHECK-NOT:    .cpload
; CHECK-NOT:    .cprestore
; CHECK: lui    $[[R0:[0-9]+]], %hi(_gp_disp)
; CHECK: addiu  $[[R1:[0-9]+]], $[[R0]], %lo(_gp_disp)
; CHECK: addu   $[[GP:[0-9]+]], $[[R1]], $25
; CHECK: lw     ${{[0-9]+}}, %call16(foo2)($[[GP]])

  tail call void @foo2(ptr @g0) nounwind
  tail call void @foo2(ptr @g1) nounwind
  tail call void @foo2(ptr @g2) nounwind
  ret void
}

declare void @foo2(ptr)
