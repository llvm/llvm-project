;; Test callsite context graph generation for call graph with with MIBs
;; that have pruned contexts that partially match multiple inlined
;; callsite contexts, requiring duplication of context ids and nodes
;; while matching callsite nodes onto the graph. This test requires more
;; complex duplication due to multiple contexts for different allocations
;; that share some of the same callsite nodes.
;;
;; Original code looks like:
;;
;; char *D(bool Call1) {
;;   if (Call1)
;;     return new char[10];
;;   else
;;     return new char[10];
;; }
;;
;; char *C(bool Call1) {
;;   return D(Call1);
;; }
;;
;; char *B(bool Call1) {
;;   if (Call1)
;;     return C(true);
;;   else
;;     return C(false);
;; }
;;
;; char *A(bool Call1) {
;;   return B(Call1);
;; }
;;
;; char *A1() {
;;   return A(true);
;; }
;;
;; char *A2() {
;;   return A(true);
;; }
;;
;; char *A3() {
;;   return A(false);
;; }
;;
;; char *A4() {
;;   return A(false);
;; }
;;
;; char *E() {
;;   return B(true);
;; }
;;
;; char *F() {
;;   return B(false);
;; }
;;
;; int main(int argc, char **argv) {
;;   char *a1 = A1(); // cold
;;   char *a2 = A2(); // cold
;;   char *e = E(); // default
;;   char *a3 = A3(); // default
;;   char *a4 = A4(); // default
;;   char *f = F(); // cold
;;   memset(a1, 0, 10);
;;   memset(a2, 0, 10);
;;   memset(e, 0, 10);
;;   memset(a3, 0, 10);
;;   memset(a4, 0, 10);
;;   memset(f, 0, 10);
;;   delete[] a3;
;;   delete[] a4;
;;   delete[] e;
;;   sleep(10);
;;   delete[] a1;
;;   delete[] a2;
;;   delete[] f;
;;   return 0;
;; }
;;
;; Code compiled with -mllvm -memprof-ave-lifetime-cold-threshold=5 so that the
;; memory freed after sleep(10) results in cold lifetimes.
;;
;; The code below was created by forcing inlining of A into its callers,
;; without any other inlining or optimizations. Since both allocation contexts
;; via A for each allocation in D have the same allocation type (cold via
;; A1 and A2 for the first new in D, and non-cold via A3 and A4 for the second
;; new in D, the contexts for those respective allocations are pruned above A.
;; The allocations via E and F are to ensure we don't prune above B.
;;
;; The matching onto the inlined A[1234]->A sequences will require duplication
;; of the context id assigned to the context from A for each allocation in D.
;; This test ensures that we do this correctly in the presence of callsites
;; shared by the different duplicated context ids (i.e. callsite in C).
;;
;; The IR was then reduced using llvm-reduce with the expected FileCheck input.

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes -memprof-dump-ccg \
; RUN:  -memprof-export-to-dot -memprof-dot-file-path-prefix=%t. \
; RUN:  %s -S 2>&1 | FileCheck %s --check-prefix=DUMP


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress noinline uwtable
define ptr @_Z1Db(i1 %Call1) #0 {
entry:
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #7, !memprof !0, !callsite !5
  br label %return

if.else:                                          ; No predecessors!
  %call1 = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #7, !memprof !6, !callsite !11
  br label %return

return:                                           ; preds = %if.else, %entry
  ret ptr null
}

; Function Attrs: nobuiltin
declare ptr @_Znam(i64) #1

define ptr @_Z1Cb(i1 %Call1) {
entry:
  %tobool = trunc i8 0 to i1
  %call = call noundef ptr @_Z1Db(i1 noundef zeroext %tobool), !callsite !12
  ret ptr null
}

; Function Attrs: mustprogress noinline uwtable
define ptr @_Z1Bb(i1 %Call1) #0 {
entry:
  %call = call noundef ptr @_Z1Cb(i1 noundef zeroext true), !callsite !13
  br label %return

if.else:                                          ; No predecessors!
  %call1 = call noundef ptr @_Z1Cb(i1 noundef zeroext false), !callsite !14
  br label %return

return:                                           ; preds = %if.else, %entry
  ret ptr null
}

define ptr @_Z1Ab(i1 %tobool) #2 {
entry:
  %call = call noundef ptr @_Z1Bb(i1 noundef zeroext %tobool), !callsite !15
  ret ptr null
}

; Function Attrs: mustprogress noinline uwtable
define ptr @_Z2A1v(i1 %tobool.i) #0 {
entry:
  %call.i = call noundef ptr @_Z1Bb(i1 noundef zeroext %tobool.i), !callsite !16
  ret ptr null
}

; Function Attrs: mustprogress noinline uwtable
define ptr @_Z2A2v(i1 %tobool.i) #0 {
entry:
  %call.i = call noundef ptr @_Z1Bb(i1 noundef zeroext %tobool.i), !callsite !17
  ret ptr null
}

; Function Attrs: mustprogress noinline uwtable
define ptr @_Z2A3v(i1 %tobool.i) #0 {
entry:
  %call.i = call noundef ptr @_Z1Bb(i1 noundef zeroext %tobool.i), !callsite !18
  ret ptr null
}

; Function Attrs: mustprogress noinline uwtable
define ptr @_Z2A4v(i1 %tobool.i) #0 {
entry:
  %call.i = call noundef ptr @_Z1Bb(i1 noundef zeroext %tobool.i), !callsite !19
  ret ptr null
}

; Function Attrs: mustprogress noinline uwtable
define ptr @_Z1Ev() #0 {
entry:
  %call = call noundef ptr @_Z1Bb(i1 noundef zeroext true), !callsite !20
  ret ptr null
}

; Function Attrs: mustprogress noinline uwtable
define ptr @_Z1Fv() #0 {
entry:
  %call = call noundef ptr @_Z1Bb(i1 noundef zeroext false), !callsite !21
  ret ptr null
}

; Function Attrs: noinline
declare i32 @main() #3

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #4

declare void @_ZdaPv() #5

declare i32 @sleep() #6

; uselistorder directives
uselistorder ptr @_Znam, { 1, 0 }

attributes #0 = { mustprogress noinline uwtable "disable-tail-calls"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nobuiltin }
attributes #2 = { "tune-cpu"="generic" }
attributes #3 = { noinline }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #5 = { "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }
attributes #6 = { "disable-tail-calls"="true" }
attributes #7 = { builtin allocsize(0) }

!0 = !{!1, !3}
!1 = !{!2, !"notcold"}
!2 = !{i64 4854880825882961848, i64 -904694911315397047, i64 6532298921261778285, i64 1905834578520680781}
!3 = !{!4, !"cold"}
!4 = !{i64 4854880825882961848, i64 -904694911315397047, i64 6532298921261778285, i64 -6528110295079665978}
!5 = !{i64 4854880825882961848}
!6 = !{!7, !9}
!7 = !{!8, !"notcold"}
!8 = !{i64 -8775068539491628272, i64 -904694911315397047, i64 7859682663773658275, i64 -6528110295079665978}
!9 = !{!10, !"cold"}
!10 = !{i64 -8775068539491628272, i64 -904694911315397047, i64 7859682663773658275, i64 -4903163940066524832}
!11 = !{i64 -8775068539491628272}
!12 = !{i64 -904694911315397047}
!13 = !{i64 6532298921261778285}
!14 = !{i64 7859682663773658275}
!15 = !{i64 -6528110295079665978}
!16 = !{i64 -6528110295079665978, i64 5747919905719679568}
!17 = !{i64 -6528110295079665978, i64 -5753238080028016843}
!18 = !{i64 -6528110295079665978, i64 1794685869326395337}
!19 = !{i64 -6528110295079665978, i64 5462047985461644151}
!20 = !{i64 1905834578520680781}
!21 = !{i64 -4903163940066524832}


;; After adding only the alloc node memprof metadata, we only have 4 contexts (we only
;; match the interesting parts of the pre-update graph here).

; DUMP: CCG before updating call stack chains:
; DUMP: Callsite Context Graph:

; DUMP: Node [[D1:0x[a-z0-9]+]]
; DUMP: 	  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #7	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2

; DUMP: Node [[C:0x[a-z0-9]+]]
; DUMP:         null Call
; DUMP:         AllocTypes: NotColdCold
; DUMP:         ContextIds: 1 2 3 4
; DUMP:         CalleeEdges:
; DUMP:                 Edge from Callee [[D1]] to Caller: [[C]] AllocTypes: NotColdCold ContextIds: 1 2
; DUMP:                 Edge from Callee [[D2:0x[a-z0-9]+]] to Caller: [[C]] AllocTypes: NotColdCold ContextIds: 3 4

; DUMP: Node [[D2]]
; DUMP: 	  %call1 = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #7	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 3 4


;; After updating for callsite metadata, we should have duplicated the context
;; ids coming from node A (2 and 3) 4 times, for the 4 different callers of A,
;; and used those on new nodes for those callers. Note that while in reality
;; we only have cold edges coming from A1 and A2 and noncold from A3 and A4,
;; due to the pruning we have lost this information and thus end up duplicating
;; both of A's contexts to all of the new nodes (which could result in some
;; unnecessary cloning.

; DUMP: CCG before cloning:
; DUMP: Callsite Context Graph:
; DUMP: Node [[D1]]
; DUMP: 	  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #7	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2 5 7 9 11
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[D1]] to Caller: [[C]] AllocTypes: NotColdCold ContextIds: 1 2 5 7 9 11

; DUMP: Node [[C]]
; DUMP: 	  %call = call noundef ptr @_Z1Db(i1 noundef zeroext %tobool)	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2 3 4 5 6 7 8 9 10 11 12
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[D1]] to Caller: [[C]] AllocTypes: NotColdCold ContextIds: 1 2 5 7 9 11
; DUMP: 		Edge from Callee [[D2]] to Caller: [[C]] AllocTypes: NotColdCold ContextIds: 3 4 6 8 10 12
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[C]] to Caller: [[B1:0x[a-z0-9]+]] AllocTypes: NotColdCold ContextIds: 1 2 5 7 9 11
; DUMP: 		Edge from Callee [[C]] to Caller: [[B2:0x[a-z0-9]+]] AllocTypes: NotColdCold ContextIds: 3 4 6 8 10 12

; DUMP: Node [[B1]]
; DUMP: 	  %call = call noundef ptr @_Z1Cb(i1 noundef zeroext true)	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2 5 7 9 11
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[C]] to Caller: [[B1]] AllocTypes: NotColdCold ContextIds: 1 2 5 7 9 11
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[B1]] to Caller: [[E:0x[a-z0-9]+]] AllocTypes: NotCold ContextIds: 1
; DUMP: 		Edge from Callee [[B1]] to Caller: [[A2:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 5
; DUMP: 		Edge from Callee [[B1]] to Caller: [[A3:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 7
; DUMP: 		Edge from Callee [[B1]] to Caller: [[A1:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 9
; DUMP: 		Edge from Callee [[B1]] to Caller: [[A4:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 11
; DUMP: 		Edge from Callee [[B1]] to Caller: [[A:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 2

; DUMP: Node [[E]]
; DUMP: 	  %call = call noundef ptr @_Z1Bb(i1 noundef zeroext true)	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[B1]] to Caller: [[E]] AllocTypes: NotCold ContextIds: 1
; DUMP: 	CallerEdges:

; DUMP: Node [[D2]]
; DUMP: 	  %call1 = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #7	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 3 4 6 8 10 12
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[D2]] to Caller: [[C]] AllocTypes: NotColdCold ContextIds: 3 4 6 8 10 12

; DUMP: Node [[B2]]
; DUMP: 	  %call1 = call noundef ptr @_Z1Cb(i1 noundef zeroext false)	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 3 4 6 8 10 12
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[C]] to Caller: [[B2]] AllocTypes: NotColdCold ContextIds: 3 4 6 8 10 12
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[B2]] to Caller: [[F:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 4
; DUMP: 		Edge from Callee [[B2]] to Caller: [[A2]] AllocTypes: NotCold ContextIds: 6
; DUMP: 		Edge from Callee [[B2]] to Caller: [[A3]] AllocTypes: NotCold ContextIds: 8
; DUMP: 		Edge from Callee [[B2]] to Caller: [[A1]] AllocTypes: NotCold ContextIds: 10
; DUMP: 		Edge from Callee [[B2]] to Caller: [[A4]] AllocTypes: NotCold ContextIds: 12
; DUMP: 		Edge from Callee [[B2]] to Caller: [[A]] AllocTypes: NotCold ContextIds: 3

; DUMP: Node [[F]]
; DUMP: 	  %call = call noundef ptr @_Z1Bb(i1 noundef zeroext false)	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 4
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[B2]] to Caller: [[F]] AllocTypes: Cold ContextIds: 4
; DUMP: 	CallerEdges:

; DUMP: Node [[A2]]
; DUMP: 	  %call = call noundef ptr @_Z1Bb(i1 noundef zeroext %tobool)	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 5 6
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[B1]] to Caller: [[A2]] AllocTypes: Cold ContextIds: 5
; DUMP: 		Edge from Callee [[B2]] to Caller: [[A2]] AllocTypes: NotCold ContextIds: 6
; DUMP: 	CallerEdges:

; DUMP: Node [[A3]]
; DUMP: 	  %call.i = call noundef ptr @_Z1Bb(i1 noundef zeroext %tobool.i)	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 7 8
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[B1]] to Caller: [[A3]] AllocTypes: Cold ContextIds: 7
; DUMP: 		Edge from Callee [[B2]] to Caller: [[A3]] AllocTypes: NotCold ContextIds: 8
; DUMP: 	CallerEdges:

; DUMP: Node [[A1]]
; DUMP: 	  %call.i = call noundef ptr @_Z1Bb(i1 noundef zeroext %tobool.i)	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 9 10
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[B1]] to Caller: [[A1]] AllocTypes: Cold ContextIds: 9
; DUMP: 		Edge from Callee [[B2]] to Caller: [[A1]] AllocTypes: NotCold ContextIds: 10
; DUMP: 	CallerEdges:

; DUMP: Node [[A4]]
; DUMP: 	  %call.i = call noundef ptr @_Z1Bb(i1 noundef zeroext %tobool.i)	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 11 12
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[B1]] to Caller: [[A4]] AllocTypes: Cold ContextIds: 11
; DUMP: 		Edge from Callee [[B2]] to Caller: [[A4]] AllocTypes: NotCold ContextIds: 12
; DUMP: 	CallerEdges:

; DUMP: Node [[A]]
; DUMP: 	  %call.i = call noundef ptr @_Z1Bb(i1 noundef zeroext %tobool.i)	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 2 3
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[B1]] to Caller: [[A]] AllocTypes: Cold ContextIds: 2
; DUMP: 		Edge from Callee [[B2]] to Caller: [[A]] AllocTypes: NotCold ContextIds: 3
; DUMP: 	CallerEdges:
