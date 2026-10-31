; RUN: opt -S -mtriple=s390x-linux-gnu \
; RUN:   -passes='require<libcall-lowering-info>,atomic-expand' %s | FileCheck %s

; Test that the memory ordering arguments of the generic atomic libcalls, which
; are C ints, get the extension attribute required by the SystemZ ABI.

define i128 @load(ptr %src) {
; CHECK-LABEL: @load(
; CHECK: call void @__atomic_load(i64 16, ptr %src, ptr %{{.*}}, i32 signext 5)
  %val = load atomic i128, ptr %src seq_cst, align 8
  ret i128 %val
}

define void @store(ptr %dst, i128 %val) {
; CHECK-LABEL: @store(
; CHECK: call void @__atomic_store(i64 16, ptr %dst, ptr %{{.*}}, i32 signext 5)
  store atomic i128 %val, ptr %dst seq_cst, align 8
  ret void
}

define i128 @exchange(ptr %dst, i128 %val) {
; CHECK-LABEL: @exchange(
; CHECK: call void @__atomic_exchange(i64 16, ptr %dst, ptr %{{.*}}, ptr %{{.*}}, i32 signext 5)
  %res = atomicrmw xchg ptr %dst, i128 %val seq_cst, align 8
  ret i128 %res
}

define i1 @cmpxchg(ptr %dst, i128 %cmp, i128 %new) {
; CHECK-LABEL: @cmpxchg(
; CHECK: call zeroext i1 @__atomic_compare_exchange(i64 16, ptr %dst, ptr %{{.*}}, ptr %{{.*}}, i32 signext 5, i32 signext 5)
  %pair = cmpxchg ptr %dst, i128 %cmp, i128 %new seq_cst seq_cst, align 8
  %success = extractvalue { i128, i1 } %pair, 1
  ret i1 %success
}

; The attribute is on the declarations as well.
; CHECK: declare void @__atomic_load(i64, ptr, ptr, i32 signext)
; CHECK: declare void @__atomic_store(i64, ptr, ptr, i32 signext)
; CHECK: declare void @__atomic_exchange(i64, ptr, ptr, ptr, i32 signext)
; CHECK: declare zeroext i1 @__atomic_compare_exchange(i64, ptr, ptr, ptr, i32 signext, i32 signext)
