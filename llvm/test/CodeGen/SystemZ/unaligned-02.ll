; Check that an unaligned i128 access get the correct alignment added.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -stop-after=pre-isel-intrinsic-lowering \
; RUN:   | FileCheck %s

define void @f1(ptr %ptr) {
; CHECK:       define void @f1(ptr %ptr) {
; CHECK-NEXT:    store i128 0, ptr %ptr, align 8
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }
  store i128 0, ptr %ptr
  ret void
}
