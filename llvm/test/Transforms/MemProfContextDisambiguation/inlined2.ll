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

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:	-memprof-verify-ccg -memprof-verify-nodes -memprof-dump-ccg \
; RUN:	%s -S 2>&1 | FileCheck %s --check-prefix=DUMP


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @_Z3barv() #0 {
entry:
  %call = call noalias noundef nonnull dereferenceable(10) ptr @_Znam(i64 noundef 10) #7, !memprof !7, !callsite !12, !heapallocsite !13
  ret ptr null
}

; Function Attrs: nobuiltin
declare ptr @_Znam(i64) #1

; Function Attrs: mustprogress
declare ptr @_Z3bazv() #2

define i32 @main() #3 {
delete.end5:
  %call.i.i = call noundef ptr @_Z3barv(), !callsite !14
  %call.i.i8 = call noundef ptr @_Z3barv(), !callsite !15
  ret i32 0
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #4

declare void @_ZdaPv() #5

declare i32 @sleep() #6

attributes #0 = { "stack-protector-buffer-size"="8" }
attributes #1 = { nobuiltin }
attributes #2 = { mustprogress }
attributes #3 = { "tune-cpu"="generic" }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #5 = { "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }
attributes #6 = { "disable-tail-calls"="true" }
attributes #7 = { builtin }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}

!0 = !{i32 7, !"Dwarf Version", i32 5}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 8, !"PIC Level", i32 2}
!4 = !{i32 7, !"PIE Level", i32 2}
!5 = !{i32 7, !"uwtable", i32 2}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = !{!8, !10}
!8 = !{!9, !"notcold"}
!9 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 8632435727821051414}
!10 = !{!11, !"cold"}
!11 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 -3421689549917153178}
!12 = !{i64 9086428284934609951}
!13 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!14 = !{i64 -5964873800580613432, i64 2732490490862098848, i64 8632435727821051414}
!15 = !{i64 -5964873800580613432, i64 2732490490862098848, i64 -3421689549917153178}


; DUMP: CCG before cloning:
; DUMP: Callsite Context Graph:
; DUMP: Node [[BAR:0x[a-z0-9]+]]
; DUMP: 	  %call = call noalias noundef nonnull dereferenceable(10) ptr @_Znam(i64 noundef 10) #7, !heapallocsite !7	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN1:0x[a-z0-9]+]] AllocTypes: NotCold ContextIds: 1
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN2:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 2

;; This is the node synthesized for the first inlined call chain of main->foo->baz
; DUMP: Node [[MAIN1]]
; DUMP: 	  %call.i.i = call noundef ptr @_Z3barv()	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 1
; DUMP: 	CallerEdges:

;; This is the node synthesized for the second inlined call chain of main->foo->baz
; DUMP: Node [[MAIN2]]
; DUMP: 	  %call.i.i8 = call noundef ptr @_Z3barv()	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 2
; DUMP: 	CallerEdges:
