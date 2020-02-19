; RUN: llc -mtriple=x86_64-none-linux < %s | FileCheck %s

; Check that all current floating-point types are correctly emitted to assembly
; on a little-endian target.

@var128 = global fp128 0xL00000000000000008000000000000000, align 16
@varppc128 = global ppc_fp128 0xM80000000000000000000000000000000, align 16
@var80 = global x86_fp80 0xK80000000000000000000, align 16
@var64 = global double -0.0, align 8
@var32 = global float -0.0, align 4
@var16 = global half -0.0, align 2
@var4f32 = global <4 x float> <float -0.0, float 0.0, float 1.0, float 2.0>
@var4f16 = global <4 x half> <half -0.0, half 0.0, half 1.0, half 2.0>

; CHECK: var128:
; CHECK-NEXT: .quad 0x0000000000000000      # fp128 -0
; CHECK-NEXT: .quad 0x8000000000000000
; CHECK-NEXT: .size

; CHECK: varppc128:
; For ppc_fp128, the high double always comes first.
; CHECK-NEXT: .quad 0x8000000000000000      # ppc_fp128 -0
; CHECK-NEXT: .quad 0x0000000000000000
; CHECK-NEXT: .size

; CHECK: var80:
; CHECK-NEXT: .quad  0x0000000000000000     # x86_fp80 -0
; CHECK-NEXT: .short 0x8000
; CHECK-NEXT: .zero 6
; CHECK-NEXT: .size

; CHECK: var64:
; CHECK-NEXT: .quad 0x8000000000000000      # double -0
; CHECK-NEXT: .size

; CHECK: var32:
; CHECK-NEXT: .long 0x80000000                # float -0
; CHECK-NEXT: .size

; CHECK: var16:
; CHECK-NEXT: .short 0x8000                   # half -0
; CHECK-NEXT: .size

; CHECK: var4f32:
; CHECK-NEXT: .long 0x80000000               # float -0
; CHECK-NEXT: .long 0x00000000               # float 0
; CHECK-NEXT: .long 0x3f800000               # float 1
; CHECK-NEXT: .long 0x40000000               # float 2
; CHECK-NEXT: .size

; CHECK: var4f16:
; CHECK-NEXT: .short 0x8000                   # half -0
; CHECK-NEXT: .short 0x0000                   # half 0
; CHECK-NEXT: .short 0x3c00                   # half 1
; CHECK-NEXT: .short 0x4000                   # half 2
; CHECK-NEXT: .size
