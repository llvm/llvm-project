; RUN: opt < %s -passes=asan -S | FileCheck --check-prefixes=CHECK,CHECK-S3 %s
; RUN: opt < %s -passes=asan -asan-mapping-scale=5 -S | FileCheck --check-prefixes=CHECK,CHECK-S5 %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
define i32 @read_4_bytes(ptr %a) sanitize_address {
entry:
  %tmp1 = load i32, ptr %a, align 4
  ret i32 %tmp1
}
; CHECK-LABEL: @read_4_bytes
; CHECK-NOT: ret
; CHECK-S3: lshr {{.*}} 3
; CHECK-S5: lshr {{.*}} 5
; Check for ASAN's Offset for 64-bit (7fff8000|7ffe0000)
; CHECK-S3-NEXT: add{{.*}}2147450880
; CHECK-S5-NEXT: add{{.*}}2147352576
; CHECK: ret

define void @example_atomicrmw(ptr %ptr) nounwind uwtable sanitize_address {
entry:
  %0 = atomicrmw add ptr %ptr, i64 1 seq_cst
  ret void
}

; CHECK-LABEL: @example_atomicrmw
; CHECK-S3: lshr {{.*}} 3
; CHECK-S5: lshr {{.*}} 5
; CHECK: __asan_report_store8
; CHECK-NOT: __asan_report
; CHECK: atomicrmw
; CHECK: ret

define void @example_cmpxchg(ptr %ptr, i64 %compare_to, i64 %new_value) nounwind uwtable sanitize_address {
entry:
  %0 = cmpxchg ptr %ptr, i64 %compare_to, i64 %new_value seq_cst seq_cst
  ret void
}

; CHECK-LABEL: @example_cmpxchg
; CHECK-S3: lshr {{.*}} 3
; CHECK-S5: lshr {{.*}} 5
; CHECK: __asan_report_store8
; CHECK-NOT: __asan_report
; CHECK: cmpxchg
; CHECK: ret
