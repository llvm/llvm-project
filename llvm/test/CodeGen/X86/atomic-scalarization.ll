; RUN: llc %s --mtriple=x86_64 -print-after=unpack-mi-bundles -disable-verify 2>&1 | FileCheck %s

define i32 @atomic_scalar() {
; CHECK: # *** IR Dump After Unpack machine instruction bundles (unpack-mi-bundles) ***:
; CHECK-NEXT: # Machine code for function atomic_scalar: NoPHIs, TracksLiveness, NoVRegs, TiedOpsRewritten, TracksDebugUserValues
; CHECK-NEXT: Frame Objects:
; CHECK-NEXT:   fi#0: size=4, align=4, at location [SP-4]
; CHECK:      bb.0 (%ir-block.0):
; CHECK-NEXT:   renamable $eax = MOV32rm $rsp, 1, $noreg, -4, $noreg :: (dereferenceable load acquire (s32) from %ir.1)
; CHECK-NEXT:   RET64 $eax
; CHECK:      # End machine code for function atomic_scalar.
  %1 = alloca <1 x i32>
  %2 = load atomic <1 x i32>, ptr %1 acquire, align 4
  %3 = extractelement <1 x i32> %2, i32 0
  ret i32 %3
}
