; RUN: llc %s --mtriple=x86_64 -print-after=finalize-isel -disable-verify 2>&1 | FileCheck %s

define i32 @atomic_scalar() {
; CHECK: # *** IR Dump After Finalize ISel and expand pseudo-instructions (finalize-isel) ***:
; CHECK-NEXT: # Machine code for function atomic_scalar: IsSSA, TracksLiveness
; CHECK-NEXT: Frame Objects:
; CHECK-NEXT:   fi#0: size=4, align=4, at location [SP+8]
; CHECK:      bb.0 (%ir-block.0):
; CHECK-NEXT:   %0:gr32 = MOV32rm %stack.0, 1, $noreg, 0, $noreg :: (dereferenceable load acquire (s32) from %ir.1)
; CHECK-NEXT:   $eax = COPY %0:gr32
; CHECK-NEXT:   RET 0, $eax
; CHECK:      # End machine code for function atomic_scalar.
  %1 = alloca <1 x i32>
  %2 = load atomic <1 x i32>, ptr %1 acquire, align 4
  %3 = extractelement <1 x i32> %2, i32 0
  ret i32 %3
}
