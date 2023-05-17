;; Test callsite context graph generation for call graph with two memprof
;; contexts and multiple levels of inlining, requiring generation of new
;; fused nodes to represent the inlined sequence while matching callsite
;; nodes onto the graph. In particular this tests the case where a function
;; has inlined a callee containing an inlined callee.
;;
;; Original code looks like:
;;
;; char *bar() __attribute__((noinline)) {
;;   return new char[10];
;; }
;;
;; char *baz() {
;;   return bar();
;; }
;;
;; char *foo() {
;;   return baz();
;; }
;;
;; int main(int argc, char **argv) {
;;   char *x = foo();
;;   char *y = foo();
;;   memset(x, 0, 10);
;;   memset(y, 0, 10);
;;   delete[] x;
;;   sleep(10);
;;   delete[] y;
;;   return 0;
;; }
;;
;; Code compiled with -mllvm -memprof-ave-lifetime-cold-threshold=5 so that the
;; memory freed after sleep(10) results in cold lifetimes.
;;
;; Both foo and baz are inlined into main, at both foo callsites.
;; We should update the graph for new fused nodes for both of those inlined
;; callsites to bar.
;;
;; Note that baz and bar are both dead due to the inlining, but have been left
;; in the input IR to ensure that the MIB call chain is matched to the longer
;; inline sequences from main.
;;
;; The IR was then reduced using llvm-reduce with the expected FileCheck input.

; RUN: opt -thinlto-bc %s >%t.o
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_Z3barv,plx \
; RUN:  -r=%t.o,_Z3bazv,plx \
; RUN:  -r=%t.o,_Z3foov,plx \
; RUN:  -r=%t.o,_ZdaPv, \
; RUN:  -r=%t.o,sleep, \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes -memprof-dump-ccg \
; RUN:  -o %t.out 2>&1 | FileCheck %s --check-prefix=DUMP


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @_Z3barv() {
entry:
  %call = call ptr @_Znam(i64 0), !memprof !0, !callsite !5
  ret ptr null
}

declare ptr @_Znam(i64)

declare ptr @_Z3bazv()

declare ptr @_Z3foov()

define i32 @main() {
delete.end5:
  %call.i.i = call ptr @_Z3barv(), !callsite !6
  %call.i.i8 = call ptr @_Z3barv(), !callsite !7
  ret i32 0
}

declare void @_ZdaPv()

declare i32 @sleep()

!0 = !{!1, !3}
!1 = !{!2, !"notcold"}
!2 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 8632435727821051414}
!3 = !{!4, !"cold"}
!4 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 -3421689549917153178}
!5 = !{i64 9086428284934609951}
!6 = !{i64 -5964873800580613432, i64 2732490490862098848, i64 8632435727821051414}
!7 = !{i64 -5964873800580613432, i64 2732490490862098848, i64 -3421689549917153178}


; DUMP: CCG before cloning:
; DUMP: Callsite Context Graph:
; DUMP: Node [[BAR:0x[a-z0-9]+]]
; DUMP: 	Versions: 1 MIB:
; DUMP: 		AllocType 1 StackIds: 0, 1, 2
; DUMP: 		AllocType 2 StackIds: 0, 1, 3
; DUMP: 	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN1:0x[a-z0-9]+]] AllocTypes: NotCold ContextIds: 1
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN2:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 2

;; This is the node synthesized for the first inlined call chain of main->foo->baz
; DUMP: Node [[MAIN1]]
; DUMP: 	Callee: 17377440600225628772 (_Z3barv) Clones: 0 StackIds: 0, 1, 2	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 1
; DUMP: 	CallerEdges:

;; This is the node synthesized for the second inlined call chain of main->foo->baz
; DUMP: Node [[MAIN2]]
; DUMP: 	Callee: 17377440600225628772 (_Z3barv) Clones: 0 StackIds: 0, 1, 3	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 2
; DUMP: 	CallerEdges:
