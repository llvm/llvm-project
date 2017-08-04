; REQUIRES: x86_64-linux
; RUN: llc < %s -march=bpfel -verify-machineinstrs | FileCheck --check-prefix=CHECK-EL %s
; RUN: llc < %s -march=bpfeb -verify-machineinstrs | FileCheck --check-prefix=CHECK-EB %s
;
; This test requires little-endian host, so we specific x86_64-linux here.
; Source code:
; struct test_t1 {
;   char a;
;   int b, c, d;
; };
;
; struct test_t1 g;
; int test()
; {
;   struct test_t1 t1 = {.a = 1};
;   g = t1;
;   return 0;
; }

%struct.test_t1 = type { i8, i32, i32, i32 }

@test.t1 = private unnamed_addr constant %struct.test_t1 { i8 1, i32 0, i32 0, i32 0 }, align 4
@g = common local_unnamed_addr global %struct.test_t1 zeroinitializer, align 4

; Function Attrs: nounwind
define i32 @test() local_unnamed_addr #0 {
entry:
    tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* getelementptr inbounds (%struct.test_t1, %struct.test_t1* @g, i64 0, i32 0), i8* getelementptr inbounds (%struct.test_t1, %struct.test_t1* @test.t1, i64 0, i32 0), i64 16, i32 4, i1 false)
; CHECK-EL:  r2 = 1
; CHECK-EL:  *(u32 *)(r1 + 0) = r2
; CHECK-EB:  r2 = 16777216
; CHECK-EB:  *(u32 *)(r1 + 0) = r2
    ret i32 0
}
; CHECK-EL:  .section .rodata.cst16,"aM",@progbits,16
; CHECK-EB:  .section .rodata.cst16,"aM",@progbits,16

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
