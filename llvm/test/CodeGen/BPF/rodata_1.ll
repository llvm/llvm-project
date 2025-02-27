; RUN: llc < %s -mtriple=bpfel -mcpu=v1 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=bpfeb -mcpu=v1 -verify-machineinstrs | FileCheck %s

; Source code:
; struct test_t1 {
;   char a, b, c;
; };
; struct test_t2 {
;   int a, b, c, d, e;
; };
;
; struct test_t1 g1;
; struct test_t2 g2;
; int test()
; {
;   struct test_t1 t1 = {.c = 1};
;   struct test_t2 t2 = {.c = 1};
;   g1 = t1;
;   g2 = t2;
;   return 0;
; }

%struct.test_t1 = type { i8, i8, i8 }
%struct.test_t2 = type { i32, i32, i32, i32, i32 }

@test.t1 = private unnamed_addr constant %struct.test_t1 { i8 0, i8 0, i8 1 }, align 1
@test.t2 = private unnamed_addr constant %struct.test_t2 { i32 0, i32 0, i32 1, i32 0, i32 0 }, align 4
@g1 = common local_unnamed_addr global %struct.test_t1 zeroinitializer, align 1
@g2 = common local_unnamed_addr global %struct.test_t2 zeroinitializer, align 4

; Function Attrs: nounwind
define i32 @test() local_unnamed_addr #0 {
; CHECK-LABEL: test:

entry:
    tail call void @llvm.memcpy.p0.p0.i64(ptr @g1, ptr @test.t1, i64 3, i1 false)
    tail call void @llvm.memcpy.p0.p0.i64(ptr align 4 @g2, ptr align 4 @test.t2, i64 20, i1 false)
; CHECK:  r1 = g1
; CHECK:  r2 = 0
; CHECK:  *(u8 *)(r1 + 1) = r2
; CHECK:  r3 = 1
; CHECK:  *(u8 *)(r1 + 2) = r3
; CHECK:  r1 = g2
; CHECK:  *(u32 *)(r1 + 8) = r3
    ret i32 0
}
; CHECK: .section  .rodata,"a",@progbits

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
