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

; test immediate out of 32-bit range
; Source file:

; unsigned long long
; load_word(void *buf, unsigned long long off)
; asm("llvm.bpf.load.word");
;
; int
; foo(void *buf)
; {
;  unsigned long long sum = 0;
;
;  sum += load_word(buf, 100);
;  sum += load_word(buf, 104);
;
;  if (sum != 0x1ffffffffULL)
;    return ~0U;
;
;  return 0;
;}

; Function Attrs: nounwind readonly
define i32 @foo(i8*) local_unnamed_addr #0 {
  %2 = tail call i64 @llvm.bpf.load.word(i8* %0, i64 100)
  %3 = tail call i64 @llvm.bpf.load.word(i8* %0, i64 104)
  %4 = add i64 %3, %2
  %5 = icmp ne i64 %4, 8589934591
; CHECK:  r{{[0-9]+}} = 8589934591 ll
  %6 = sext i1 %5 to i32
  ret i32 %6
}

; Function Attrs: nounwind readonly
declare i64 @llvm.bpf.load.word(i8*, i64) #1
