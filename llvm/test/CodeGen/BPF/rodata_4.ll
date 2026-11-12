; RUN: llc < %s -mtriple=bpfel -mcpu=v1 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=bpfeb -mcpu=v1 -verify-machineinstrs | FileCheck %s

; Source code:
; struct test_t1
; {
;   short a;
;   short b;
;   char c;
; };
;
; struct test_t1 g;
; int test()
; {
;   struct test_t1 t1[] = {{50, 500, 5}, {60, 600, 6}, {70, 700, 7}, {80, 800, 8} };
;
;   g = t1[1];
;   return 0;
; }

%struct.test_t1 = type { i16, i16, i8 }

@test.t1 = private unnamed_addr constant [4 x %struct.test_t1] [%struct.test_t1 { i16 50, i16 500, i8 5 }, %struct.test_t1 { i16 60, i16 600, i8 6 }, %struct.test_t1 { i16 70, i16 700, i8 7 }, %struct.test_t1 { i16 80, i16 800, i8 8 }], align 2
@g = common local_unnamed_addr global %struct.test_t1 zeroinitializer, align 2

; Function Attrs: nounwind
define i32 @test() local_unnamed_addr #0 {
; CHECK-LABEL: test:
entry:
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 2 @g, ptr align 2 getelementptr inbounds ([4 x %struct.test_t1], ptr @test.t1, i64 0, i64 1), i64 6, i1 false)
; CHECK:  r2 = 600
; CHECK:  *(u16 *)(r1 + 2) = r2
; CHECK:  r2 = 60
; CHECK:  *(u16 *)(r1 + 0) = r2
  ret i32 0
}
; CHECK:  .section  .rodata,"a",@progbits

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
