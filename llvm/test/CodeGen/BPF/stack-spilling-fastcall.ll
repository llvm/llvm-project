; RUN: llc -mtriple=bpfel -mcpu=v3 -bpf-stack-spilling -verify-machineinstrs < %s | FileCheck %s

; Caller-saved spills inserted after frame layout must follow the same layout
; as ordinary spills, and must not overwrite incoming arguments or locals.
declare i32 @ext() "bpf_fastcall"

define i32 @caller(i32 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %x) {
; CHECK-LABEL: caller:
; CHECK: *(u64 *)(r10 - 4088) = r{{[0-9]+}}
; CHECK: *(u64 *)(r10 - 4032) = r1
; CHECK-NEXT: call ext
; CHECK-NEXT: r1 = *(u64 *)(r10 - 4032)
; CHECK: w0 = w1
; CHECK: exit
  %local = alloca i64, align 8
  store volatile i64 42, ptr %local
  %r = call i32 @ext() "bpf_fastcall"
  ret i32 %a
}
