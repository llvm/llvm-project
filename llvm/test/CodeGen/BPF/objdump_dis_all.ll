; RUN: llc -march=bpfel -filetype=obj -o - %s | llvm-objdump -D - | FileCheck %s
;
; Source:
;   /* *(u64 *)(r10 - 16) = r1 */
;   unsigned long long g = 0x00000000fff01a7bULL;
;   /* *(u64 *)(r15 - 16) = r1 */
;   unsigned long long h = 0x00000000fff01f7bULL;
;   int test() {
;     return 0;
;   }
; Compilation flag:
;  clang -target bpf -O2 -S -emit-llvm t.c

@g = dso_local local_unnamed_addr global i64 4293925499, align 8
@h = dso_local local_unnamed_addr global i64 4293926779, align 8

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @test() local_unnamed_addr {
entry:
  ret i32 0
}
; CHECK-LABEL: section .data
; CHECK-LABEL: g
; CHECK:       *(u64 *)(r10 - 16) = r1
; CHECK-LABEL: h
; CHECK:       <unknown>
