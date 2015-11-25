; RUN: llc < %s -march=x86-64 | FileCheck %s

; With optimization at O2 we actually get the legalized function optimized
; away through legalization and stack coloring, but check that we do all of
; that here and don't crash during legalization.

; Original program:
; typedef enum { A, B, C, D } P;
; struct { P x[2]; } a;

; void fn2();
; void fn1() {
;   int b;
;   unsigned c;
;   for (;; c++) {
;     fn2();
;     unsigned n;
;     for (; c; c++) {
;       b = a.x[c] == A || a.x[c] == B || a.x[c] == D;
;       if (b) n++;
;     }
;     if (n)
;	for (;;)
;	  ;
;   }
; }

define void @fn1() {
; CHECK-LABEL: fn1
; CHECK: movb	$0, {{.*}}(%rsp)
; CHECK: cmpq	$8, %rax
for.cond:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %for.cond
  %x42 = bitcast <4 x i4> zeroinitializer to i16
  %x43 = icmp ne i16 %x42, 0
  %x44 = select i1 %x43, i32 undef, i32 0
  %x72 = bitcast <4 x i1> zeroinitializer to i4
  %x73 = icmp ne i4 %x72, 0
  %x74 = select i1 %x73, i32 %x44, i32 undef
  %x84 = select i1 undef, i32 undef, i32 %x74
  %x88 = icmp eq i64 undef, 8
  br i1 %x88, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %0 = select i1 undef, i32 undef, i32 %x84
  ret void
}
