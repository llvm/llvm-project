;; Test callsite context graph generation for simple call graph with
;; two memprof contexts and no inlining, where one callsite required for
;; cloning is missing (e.g. unmatched). Use this to test aggressive hinting
;; threshold.
;;
;; Original code looks like:
;;
;; char *foo() {
;;   return new char[10];
;; }
;;
;; int main(int argc, char **argv) {
;;   char *x = foo();
;;   char *y = foo();
;;   memset(x, 0, 10);
;;   memset(y, 0, 10);
;;   delete[] x;
;;   sleep(200);
;;   delete[] y;
;;   return 0;
;; }

; RUN: opt -thinlto-bc -memprof-report-hinted-sizes %s >%t.o
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:	-supports-hot-cold-new \
; RUN:	-r=%t.o,main,plx \
; RUN:	-r=%t.o,_Znam, \
; RUN:	-memprof-report-hinted-sizes \
; RUN:	-pass-remarks=memprof-context-disambiguation -save-temps \
; RUN:	-o %t.out 2>&1 | FileCheck %s --implicit-check-not "call in clone _Z3foov" \
; RUN:  --check-prefix=SIZESUNHINTED
; RUN: llvm-dis %t.out.1.4.opt.bc -o - | FileCheck %s --implicit-check-not "\"memprof\"=\"cold\""

;; Check that we do hint with a sufficient -memprof-cloning-cold-threshold.
; RUN: opt -thinlto-bc -memprof-report-hinted-sizes %s >%t.o
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:	-supports-hot-cold-new \
; RUN:	-r=%t.o,main,plx \
; RUN:	-r=%t.o,_Znam, \
; RUN:	-memprof-report-hinted-sizes -memprof-cloning-cold-threshold=80 \
; RUN:	-pass-remarks=memprof-context-disambiguation -save-temps \
; RUN:	-o %t.out 2>&1 | FileCheck %s --check-prefix=REMARKSHINTED \
; RUN:  --check-prefix=SIZESHINTED
; RUN: llvm-dis %t.out.1.4.opt.bc -o - | FileCheck %s --check-prefix=IRHINTED

;; Check again that we hint with a sufficient -memprof-cloning-cold-threshold,
;; even if we don't specify -memprof-report-hinted-sizes.
; RUN: opt -thinlto-bc -memprof-report-hinted-sizes %s >%t.o
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:	-supports-hot-cold-new \
; RUN:	-r=%t.o,main,plx \
; RUN:	-r=%t.o,_Znam, \
; RUN:	-memprof-cloning-cold-threshold=80 \
; RUN:	-pass-remarks=memprof-context-disambiguation -save-temps \
; RUN:	-o %t.out 2>&1 | FileCheck %s --check-prefix=REMARKSHINTED
; RUN: llvm-dis %t.out.1.4.opt.bc -o - | FileCheck %s --check-prefix=IRHINTED

source_filename = "memprof-missing-callsite.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() #0 {
entry:
  ;; Missing callsite metadata blocks cloning
  %call = call ptr @_Z3foov()
  %call1 = call ptr @_Z3foov()
  ret i32 0
}

define internal ptr @_Z3foov() #0 {
entry:
  %call = call ptr @_Znam(i64 0), !memprof !2, !callsite !7
  ret ptr null
}

declare ptr @_Znam(i64)

; uselistorder directives
uselistorder ptr @_Z3foov, { 1, 0 }

attributes #0 = { noinline optnone }

!2 = !{!3, !5}
!3 = !{!4, !"notcold", !10}
!4 = !{i64 9086428284934609951, i64 8632435727821051414}
!5 = !{!6, !"cold", !11, !12}
!6 = !{i64 9086428284934609951, i64 -3421689549917153178}
!7 = !{i64 9086428284934609951}
!10 = !{i64 123, i64 100}
!11 = !{i64 456, i64 200}
!12 = !{i64 789, i64 300}

; SIZESUNHINTED: NotCold full allocation context 123 with total size 100 is NotColdCold after cloning
; SIZESUNHINTED: Cold full allocation context 456 with total size 200 is NotColdCold after cloning
; SIZESUNHINTED: Cold full allocation context 789 with total size 300 is NotColdCold after cloning

; SIZESHINTED: NotCold full allocation context 123 with total size 100 is NotColdCold after cloning marked Cold due to cold byte percent
; SIZESHINTED: Cold full allocation context 456 with total size 200 is NotColdCold after cloning marked Cold due to cold byte percent
; SIZESHINTED: Cold full allocation context 789 with total size 300 is NotColdCold after cloning marked Cold due to cold byte percent

; REMARKSHINTED: call in clone _Z3foov marked with memprof allocation attribute cold

; IRHINTED: "memprof"="cold"
