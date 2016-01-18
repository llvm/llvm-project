; RUN: llc -mtriple=x86_64-none-linux < %s | FileCheck %s

; Check that all current floating-point types are correctly emitted to assembly
; on a little-endian target.

@var128 = global fp128 0xL00000000000000008000000000000000, align 16
@varppc128 = global ppc_fp128 0xM80000000000000000000000000000000, align 16
@var80 = global x86_fp80 0xK80000000000000000000, align 16
@var64 = global double -0.0, align 8
@var32 = global float -0.0, align 4
@var16 = global half -0.0, align 2

; CHECK: var128:
; CHECK-NEXT: .quad 0                         # fp128 -0
; CHECK-NEXT: .quad -9223372036854775808
; CHECK-NEXT: .size

; CHECK: varppc128:
; For ppc_fp128, the high double always comes first.
; CHECK-NEXT: .quad -9223372036854775808      # ppc_fp128 -0
; CHECK-NEXT: .quad 0
; CHECK-NEXT: .size

; CHECK: var80:
; CHECK-NEXT: .quad 0                         # x86_fp80 -0
; CHECK-NEXT: .short 32768
; CHECK-NEXT: .zero 6
; CHECK-NEXT: .size

; CHECK: var64:
; CHECK-NEXT: .quad -9223372036854775808      # double -0
; CHECK-NEXT: .size

; CHECK: var32:
; CHECK-NEXT: .long 2147483648                # float -0
; CHECK-NEXT: .size

; CHECK: var16:
; CHECK-NEXT: .short 32768                    # half -0
; CHECK-NEXT: .size

