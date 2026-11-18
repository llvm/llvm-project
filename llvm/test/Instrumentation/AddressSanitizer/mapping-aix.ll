; Test shadow memory mapping on AIX

; RUN: opt -passes=asan -mtriple=powerpc64-ibm-aix -S < %s | FileCheck %s -check-prefix=CHECK-64
; RUN: opt -passes=asan -mtriple=powerpc-ibm-aix -S < %s | FileCheck %s -check-prefix=CHECK-32

; CHECK: @test
; On 64-bit AIX, we expect a left shift of 6 (HIGH_BITS) followed by a right shift of 9 (HIGH_BITS 
; + ASAN_SHADOW_SCALE) and an offset of 0x0a01000000000000.
; CHECK-64: shl {{.*}} 6
; CHECK-64-NEXT: lshr {{.*}} 9
; CHECK-64-NEXT: add {{.*}} 720857415355990016 
; On 32-bit AIX, we expect just a right shift of 3 and an offset of 0x40000000.
; CHECK-32: lshr {{.*}} 3
; CHECK-32-NEXT: add {{.*}} 1073741824 
; CHECK: ret

define i32 @test(i32* %a) sanitize_address {
entry:
  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
}
