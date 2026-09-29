; RUN: llc -mtriple=bpf -mcpu=v3 < %s | FileCheck %s
;
; Source code:
;   void foo(int, int, int, long, int);
;   int test(int a, int b, int c, long d, int e) {
;     foo(a, b, c, d, e);
;     __asm__ __volatile__ ("":::"r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "memory");
;     foo(a, b, c, d, e);
;     return 0;
;   }
; Compilation flag:
;   clang -target bpf -S -emit-llvm -O2 -mcpu=v3 t.c

; Function Attrs: nounwind
define dso_local i32 @test(i32 %a, i32 %b, i32 %c, i64 %d, i32 %e) local_unnamed_addr #0 {
entry:
  tail call void @foo(i32 %a, i32 %b, i32 %c, i64 %d, i32 %e) #2
  tail call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{memory}"() #2

; CHECK:        *(u32 *)(r10 - 8) = w5
; CHECK:        *(u64 *)(r10 - 16) = r4
; CHECK:        *(u32 *)(r10 - 24) = w3
; CHECK:        *(u32 *)(r10 - 32) = w2
; CHECK:        *(u32 *)(r10 - 40) = w1
; CHECK:        call foo

  tail call void @foo(i32 %a, i32 %b, i32 %c, i64 %d, i32 %e) #2
  ret i32 0
}

declare dso_local void @foo(i32, i32, i32, i64, i32) local_unnamed_addr #1

attributes #0 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="v3" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="v3" }
attributes #2 = { nounwind }
