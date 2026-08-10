; RUN: opt -passes=assume-builder --enable-knowledge-retention -S %s | FileCheck %s

define void @test_atomic_accesses(ptr %p, ptr %q) {
; CHECK-LABEL: define {{[^@]+}}@test_atomic_accesses
; CHECK-SAME: (ptr [[P:%.*]], ptr [[Q:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.assume(i1 true) [ "dereferenceable"(ptr [[P]], i64 4), "nonnull"(ptr [[P]]), "align"(ptr [[P]], i64 8) ]
; CHECK-NEXT:    [[OLD:%.*]] = atomicrmw add ptr [[P]], i32 1 monotonic, align 8
; CHECK-NEXT:    call void @llvm.assume(i1 true) [ "dereferenceable"(ptr [[Q]], i64 8), "nonnull"(ptr [[Q]]), "align"(ptr [[Q]], i64 16) ]
; CHECK-NEXT:    [[PAIR:%.*]] = cmpxchg ptr [[Q]], i64 0, i64 1 monotonic monotonic, align 16
; CHECK-NEXT:    ret void
;
entry:
  %old = atomicrmw add ptr %p, i32 1 monotonic, align 8
  %pair = cmpxchg ptr %q, i64 0, i64 1 monotonic monotonic, align 16
  ret void
}
