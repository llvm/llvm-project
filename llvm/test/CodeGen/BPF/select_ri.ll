; RUN: llc < %s -march=bpf -verify-machineinstrs | FileCheck %s
;
; Source file:
; int b, c;
; int test() {
;   int a = b;
;   if (a)
;     a = c;
;   return a;
; }
@b = common local_unnamed_addr global i32 0, align 4
@c = common local_unnamed_addr global i32 0, align 4

; Function Attrs: norecurse nounwind readonly
define i32 @test() local_unnamed_addr #0 {
entry:
  %0 = load i32, i32* @b, align 4
  %tobool = icmp eq i32 %0, 0
  %1 = load i32, i32* @c, align 4
  %. = select i1 %tobool, i32 0, i32 %1
; CHECK:  r1 = b
; CHECK:  r1 = *(u32 *)(r1 + 0)
; CHECK:  if r1 == 0 goto
  ret i32 %.
}

attributes #0 = { norecurse nounwind readonly }
