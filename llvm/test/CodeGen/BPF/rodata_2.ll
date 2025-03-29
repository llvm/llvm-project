; RUN: llc < %s -mtriple=bpfel -mcpu=v1 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=bpfeb -mcpu=v1 -verify-machineinstrs | FileCheck %s

; Source code:
; struct test_t1 {
;   char a;
;   int  b;
; };
; struct test_t2 {
;   char a, b;
;   struct test_t1 c[2];
;   int d[2];
;   int e;
; };
; struct test_t2 g;
; int test()
; {
;    struct test_t2 t2 = {.c = {{}, {.b = 1}}, .d = {2, 3}};
;    g = t2;
;    return 0;
; }

%struct.test_t2 = type { i8, i8, [2 x %struct.test_t1], [2 x i32], i32 }
%struct.test_t1 = type { i8, i32 }

@test.t2 = private unnamed_addr constant %struct.test_t2 { i8 0, i8 0, [2 x %struct.test_t1] [%struct.test_t1 zeroinitializer, %struct.test_t1 { i8 0, i32 1 }], [2 x i32] [i32 2, i32 3], i32 0 }, align 4
@g = common local_unnamed_addr global %struct.test_t2 zeroinitializer, align 4

; Function Attrs: nounwind
define i32 @test() local_unnamed_addr #0 {
; CHECK-LABEL: test:

entry:
    tail call void @llvm.memcpy.p0.p0.i64(ptr align 4 @g, ptr align 4 @test.t2, i64 32, i1 false)
; CHECK: r1 = g ll
; CHECK: r2 = 3
; CHECK: *(u32 *)(r1 + 24) = r2
; CHECK: r2 = 2
; CHECK: *(u32 *)(r1 + 20) = r2
; CHECK: r2 = 1
; CHECK: *(u32 *)(r1 + 16) = r2
; CHECK: r2 = 0
; CHECK: *(u32 *)(r1 + 28) = r2
; CHECK: *(u32 *)(r1 + 8) = r2
; CHECK: *(u32 *)(r1 + 4) = r2
; CHECK: *(u32 *)(r1 + 0) = r2
      ret i32 0
}
; CHECK: .section  .rodata.cst32,"aM",@progbits,32

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
