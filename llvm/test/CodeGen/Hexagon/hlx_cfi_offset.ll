; RUN: llc -mtriple=hexagon -mcpu=hexagonv85 -mattr=+hlx -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,NOVSP
; RUN: llc -mtriple=hexagon -mcpu=hexagonv85 -mattr=+hlx -verify-machineinstrs -hexagon-save-parent-vsp < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,VSP
; RUN: llc -mtriple=hexagon -mcpu=hexagonv85 -mattr=+hlx -verify-machineinstrs -hexagon-save-parent-vsp -filetype=obj < %s \
; RUN:   | llvm-dwarfdump --debug-frame - | FileCheck %s --check-prefix=DWARF

; The HLX callee-saved registers L12-L15 are spilled to the HLX vector stack,
; so their unwind rules are expressions relative to VSP. With
; -hexagon-save-parent-vsp the caller's VSP is also saved on the scalar stack.

; DWARF:      DW_CFA_expression: L12 DW_OP_bregx VSP+6144
; DWARF-NEXT: DW_CFA_expression: L13 DW_OP_bregx VSP+4096
; DWARF-NEXT: DW_CFA_expression: L14 DW_OP_bregx VSP+2048
; DWARF-NEXT: DW_CFA_expression: L15 DW_OP_bregx VSP+0
; DWARF:      DW_CFA_same_value: L12

; CHECK-LABEL: foo:
; CHECK:      .cfi_offset r16, -16
; CHECK-NEXT: .cfi_escape 0x10, 0x9a, 0x02, 0x04, 0x92, 0x60, 0x80, 0x30 // L12 at vsp + 6144
; CHECK-NEXT: .cfi_escape 0x10, 0x9b, 0x02, 0x04, 0x92, 0x60, 0x80, 0x20 // L13 at vsp + 4096
; CHECK-NEXT: .cfi_escape 0x10, 0x9c, 0x02, 0x04, 0x92, 0x60, 0x80, 0x10 // L14 at vsp + 2048
; CHECK-NEXT: .cfi_escape 0x10, 0x9d, 0x02, 0x03, 0x92, 0x60, 0x00 // L15 at vsp + 0
; VSP-NEXT:   .cfi_offset vsp, -20
; NOVSP-NOT:  .cfi_offset vsp
; VSP:        r7 = vsp
; VSP-NEXT:   allocframe(r29,#16):raw
; VSP:        vallocframe(#4)
; VSP:        memw(r29+#4) = r7
define <1024 x i16> @foo(ptr %in, i32 %n) {
entry:
  %call = tail call i32 @bar(i32 %n)
  %r = tail call <1024 x i16> @llvm.hexagon.LX.assign.r(i32 %call)
  %v = load <1024 x i16>, ptr %in, align 2048
  %add = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %v, <1024 x i16> %r)
  ret <1024 x i16> %add
}

; CHECK-LABEL: no_vtcm:
; CHECK-NOT:  vsp
; CHECK:      .cfi_endproc
define i32 @no_vtcm(i32 %n) {
  %call = tail call i32 @bar(i32 %n)
  ret i32 %call
}

; L12-L15 are not spilled, so they keep their values.
; CHECK-LABEL: leaf:
; CHECK:      .cfi_offset r30, -8
; CHECK-NEXT: .cfi_same_value L12
; CHECK-NEXT: .cfi_same_value L13
; CHECK-NEXT: .cfi_same_value L14
; CHECK-NEXT: .cfi_same_value L15
; VSP-NEXT:   .cfi_offset vsp, -12
define void @leaf(<1024 x i16> %v) {
  %a = alloca <1024 x i16>, align 2048
  store volatile <1024 x i16> %v, ptr %a, align 2048
  ret void
}

; The caller's VSP is copied through R7, which does not hold an argument.
; CHECK-LABEL: live_args:
; VSP:        r7 = vsp
; VSP-NEXT:   allocframe(r29,#8):raw
; VSP:        .cfi_offset vsp, -12
; VSP:        memw(r29+#4) = r7
define void @live_args(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, <1024 x i16> %v) {
  %a = alloca <1024 x i16>, align 2048
  store volatile <1024 x i16> %v, ptr %a, align 2048
  call void @use6(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5)
  ret void
}

; HLX code is not shrink-wrapped, so the prologue stays in the entry block.
; CHECK-LABEL: shrink_wrap:
; VSP:        r7 = vsp
; VSP-NEXT:   cmp.eq
; CHECK:      allocframe
; CHECK:      %work
define <1024 x i16> @shrink_wrap(i32 %c, <1024 x i16> %v) {
entry:
  %t = icmp eq i32 %c, 0
  br i1 %t, label %exit, label %work
work:
  %call = tail call i32 @bar(i32 %c)
  %r = tail call <1024 x i16> @llvm.hexagon.LX.assign.r(i32 %call)
  %add = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %v, <1024 x i16> %r)
  br label %exit
exit:
  %p = phi <1024 x i16> [ %v, %entry ], [ %add, %work ]
  ret <1024 x i16> %p
}

declare i32 @bar(i32)
declare <1024 x i16> @llvm.hexagon.LX.assign.r(i32)
declare <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16>, <1024 x i16>)
declare void @use6(i32, i32, i32, i32, i32, i32)
