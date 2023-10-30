; RUN: opt < %s -passes=asan -S -mtriple=aarch64-linux-gnu | FileCheck --check-prefix=CHECK-AARCH64LE %s
; RUN: opt < %s -passes=asan -S -mtriple=aarch64_be-linux-gnu | FileCheck --check-prefix=CHECK-AARCH64BE %s
; REQUIRES: aarch64-registered-target
 
; REQUIRES: aarch64-registered-target

define i32 @read_4_bytes(i32* %a) sanitize_address {
entry:
  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
}
 
; CHECK-AARCH64LE: @read_4_bytes
; CHECK-AARCH64LE-NOT: ret
; Check for ASAN's Offset for AArch64 LE (1 << 36 or 68719476736)
; CHECK-AARCH64LE: lshr {{.*}} 3
; CHECK-AARCH64Le-NEXT: {{68719476736}}
; CHECK-AARCH64LE: ret
 
; CHECK-AARCH64BE: @read_4_bytes
; CHECK-AARCH64BE-NOT: ret
; Check for ASAN's Offset for AArch64 BE (1 << 36 or 68719476736)
; CHECK-AARCH64BE: lshr {{.*}} 3
; CHECK-AARCH64BE-NEXT: {{68719476736}}
; CHECK-AARCH64BE: ret
