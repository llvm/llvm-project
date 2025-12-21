; RUN: not llc -mcpu=gfx1100 -mtriple=amdgcn-amd-amdhsa -stress-regalloc=4 -amdgpu-enable-rewrite-partial-reg-uses=0 -filetype=null -verify-machineinstrs %s 2>&1 | FileCheck %s

; CHECK: error: <unknown>:0:0: ran out of registers during register allocation in function 'f'
; CHECK-NOT: Bad machine code

define <16 x half> @f(i1 %LGV2, <16 x half> %0) {
BB:
  br i1 %LGV2, label %SW_C3, label %SW_C

SW_C:                                             ; preds = %BB
  %B1 = fmul <16 x half> %0, zeroinitializer
  ret <16 x half> %B1

SW_C3:                                            ; preds = %BB
  ret <16 x half> <half 0xH0000, half poison, half poison, half poison, half poison, half poison, half poison, half poison, half poison, half poison, half poison, half poison, half poison, half poison, half poison, half poison>
}
