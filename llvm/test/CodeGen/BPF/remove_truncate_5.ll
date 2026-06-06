; RUN: llc < %s -mtriple=bpfel | FileCheck -check-prefixes=CHECK %s

; Source code:
; struct test_t {
;       int a;
;       char b;
;       int c;
;       char d;
; };
; void foo(ptr);
; void test() {
;       struct test_t t = {.a = 5};
;       foo(&t);
; }

%struct.test_t = type { i32, i8, i32, i8 }

@test.t = private unnamed_addr constant %struct.test_t { i32 5, i8 0, i32 0, i8 0 }, align 4

; Function Attrs: nounwind
define dso_local void @test() local_unnamed_addr #0 {
; CHECK-LABEL: test:
  %1 = alloca %struct.test_t, align 4
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %1) #3
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 4 %1, ptr align 4 @test.t, i64 16, i1 false)
; CHECK: r1 = 0
; CHECK: *(u64 *)(r10 - 8) = r1
; CHECK: r1 = 5
; CHECK: *(u64 *)(r10 - 16) = r1
; CHECK: r1 = r10
; CHECK: r1 += -16
  call void @foo(ptr nonnull %1) #3
; CHECK: call foo
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %1) #3
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1) #1

declare dso_local void @foo(ptr) local_unnamed_addr

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
attributes #3 = { nounwind }
