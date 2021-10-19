; RUN: opt < %s -passes='asan-pipeline' -S -mtriple=x86_64-scei-ps4 | FileCheck %s

define i32 @read_4_bytes(i32* %a) sanitize_address {
entry:
  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
}

; CHECK: @read_4_bytes
; CHECK-NOT: ret
; Check for ASAN's Offset on the PS4 (2^40 or 0x10000000000)
; CHECK: lshr {{.*}} 3
; CHECK-NEXT: {{1099511627776}}
; CHECK: ret
