; RUN: llc -mtriple=hexagon -mv73 -mattr=+hvxv73,+hvx-length128b -verify-machineinstrs=true < %s | FileCheck %s

; Test that plain M-register usage by HVX vgather pseudo-instructions
; (V6_vgathermh_pseudo etc., which take a plain ModRegs:$Mu modifier) does not
; clash with an M register that is allocated as part of the CS/M CircRegs pair
; used by the new-style circular load intrinsics. Both a circular load and a
; vgather are live in the same loop; the allocator must keep the CS-paired M
; register (feeding the circular load) and the plain M register (feeding
; vgather) independent.

; CHECK-LABEL: test_circ_and_gather
; CHECK: m1 = r3
; CHECK: cs0 = r1
; CHECK: memub(r0++#4:circ(m0))
; CHECK: vgather(r{{[0-9]+}},m1,v{{[0-9]+}}.h).h

declare {i32, ptr} @llvm.hexagon.L2.loadrub.pci(ptr, i32, i32, ptr)
declare void @llvm.hexagon.V6.vgathermh.128B(ptr, i32, i32, <32 x i32>)

@g = global i32 0

define void @test_circ_and_gather(ptr %pc, ptr %bc, ptr %gdst, i32 %gmod, <32 x i32> %gaddr, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%inext, %loop]
  %ppc = phi ptr [%pc, %entry], [%nppc, %loop]
  %rc = call {i32, ptr} @llvm.hexagon.L2.loadrub.pci(ptr %ppc, i32 4, i32 100, ptr %bc)
  %vc = extractvalue {i32, ptr} %rc, 0
  %nppc = extractvalue {i32, ptr} %rc, 1
  call void @llvm.hexagon.V6.vgathermh.128B(ptr %gdst, i32 ptrtoint (ptr @g to i32), i32 %gmod, <32 x i32> %gaddr)
  %inext = add i32 %i, 1
  %cmp = icmp slt i32 %inext, %n
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}
