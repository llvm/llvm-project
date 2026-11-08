; RUN: opt -passes=attributor -verify-each -S < %s | FileCheck %s
; RUN: opt -passes=attributor-cgscc -verify-each -S < %s | FileCheck %s

target datalayout = "p:128:128-p1:128:128:128:32-p2:64:64-i128:128"

@bytes = global [16 x i8] zeroinitializer
@narrow = addrspace(1) global [16 x i8] zeroinitializer
@ordinary = addrspace(2) global [16 x i8] zeroinitializer

; Original reproducer from InstSimplify/gep-wide-index.ll.
define ptr @index() {
; CHECK-LABEL: define noundef ptr @index()
  %r = getelementptr i8, ptr null, i128 18446744073709551621
  ret ptr %r
}

; Looking at only the low 64 bits would place both addresses at @bytes + 5.
define ptr @wide_positive() {
; CHECK-LABEL: define noundef ptr @wide_positive()
  %r = getelementptr i8, ptr @bytes, i128 18446744073709551621
  ret ptr %r
}

define ptr @wide_negative() {
; CHECK-LABEL: define noundef ptr @wide_negative()
  %r = getelementptr i8, ptr @bytes, i128 -18446744073709551611
  ret ptr %r
}

; This positive displacement fits in uint64_t but exceeds INT64_MAX.
define ptr @just_past_signed_max() {
; CHECK-LABEL: define noundef ptr @just_past_signed_max()
  %r = getelementptr i8, ptr @bytes, i128 9223372036854775808
  ret ptr %r
}

define ptr @small_offset() {
; CHECK-LABEL: define noundef dereferenceable_or_null(11) ptr @small_offset()
  %r = getelementptr i8, ptr @bytes, i128 5
  ret ptr %r
}

; Check the accumulated displacement: the two large indices cancel to five.
define ptr @cancel() {
; CHECK-LABEL: define noundef dereferenceable_or_null(11) ptr @cancel()
  %a = getelementptr i8, ptr @bytes, i128 18446744073709551621
  %r = getelementptr i8, ptr %a, i128 -18446744073709551616
  ret ptr %r
}

; The address space determines which index bits participate in the GEP.
define ptr addrspace(1) @index32() {
; CHECK-LABEL: define noundef dereferenceable_or_null(11) ptr addrspace(1) @index32()
  %r = getelementptr i8, ptr addrspace(1) @narrow, i128 4294967301
  ret ptr addrspace(1) %r
}

define ptr addrspace(2) @index64() {
; CHECK-LABEL: define noundef dereferenceable_or_null(11) ptr addrspace(2) @index64()
  %r = getelementptr i8, ptr addrspace(2) @ordinary, i128 18446744073709551621
  ret ptr addrspace(2) %r
}

; The store proves one byte at %q. It gives no dereferenceable bytes at %p.
define ptr @access_before_base(ptr %p) {
; CHECK-LABEL: define noundef nonnull dereferenceable(1) ptr @access_before_base(
; CHECK-NOT: dereferenceable
; CHECK-SAME: %p)
  %q = getelementptr inbounds i8, ptr %p, i128 -18446744073709551611
  store i8 0, ptr %q
  ret ptr %q
}
