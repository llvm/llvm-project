; RUN: llc < %s -march=bpfel | FileCheck -check-prefixes=CHECK %s

; Source code:
; struct test_t {
;       int a;
;       char b;
;       int c;
;       char d;
; };
; void foo(void *);
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
  %2 = bitcast %struct.test_t* %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %2) #3
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 4 %2, i8* align 4 bitcast (%struct.test_t* @test.t to i8*), i64 16, i1 false)
; CHECK: r1 = 0
; CHECK: r1 <<= 32
; CHECK: r2 = r1
; CHECK: r2 |= 0
; CHECK: *(u64 *)(r10 - 8) = r2
; CHECK: r1 |= 5
; CHECK: *(u64 *)(r10 - 16) = r1
  call void @foo(i8* nonnull %2) #3
; CHECK: call foo
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %2) #3
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

declare dso_local void @foo(i8*) local_unnamed_addr

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
attributes #3 = { nounwind }
