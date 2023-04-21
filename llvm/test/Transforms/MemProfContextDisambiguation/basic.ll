;; Test callsite context graph generation for simple call graph with
;; two memprof contexts and no inlining.
;;
;; Original code looks like:
;;
;; char *bar() {
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
;; Code compiled with -mllvm -memprof-min-lifetime-cold-threshold=5 so that the
;; memory freed after sleep(10) results in cold lifetimes.
;;
;; The IR was then reduced using llvm-reduce with the expected FileCheck input.

; RUN: opt -passes=memprof-context-disambiguation \
; RUN:	-memprof-verify-ccg -memprof-verify-nodes -memprof-dump-ccg \
; RUN:	-memprof-export-to-dot -memprof-dot-file-path-prefix=%t. \
; RUN:	%s -S 2>&1 | FileCheck %s --check-prefix=DUMP

; RUN:	cat %t.ccg.postbuild.dot | FileCheck %s --check-prefix=DOT

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() #0 {
entry:
  %call = call noundef ptr @_Z3foov(), !callsite !0
  %call1 = call noundef ptr @_Z3foov(), !callsite !1
  ret i32 0
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #1

; Function Attrs: nobuiltin
declare void @_ZdaPv() #2

define internal ptr @_Z3barv() #3 {
entry:
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #6, !memprof !2, !callsite !7
  ret ptr null
}

declare ptr @_Znam(i64)

define internal ptr @_Z3bazv() #4 {
entry:
  %call = call noundef ptr @_Z3barv(), !callsite !8
  ret ptr null
}

; Function Attrs: noinline
define internal ptr @_Z3foov() #5 {
entry:
  %call = call noundef ptr @_Z3bazv(), !callsite !9
  ret ptr null
}

; uselistorder directives
uselistorder ptr @_Z3foov, { 1, 0 }

attributes #0 = { "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #2 = { nobuiltin }
attributes #3 = { "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }
attributes #4 = { "stack-protector-buffer-size"="8" }
attributes #5 = { noinline }
attributes #6 = { builtin }

!0 = !{i64 8632435727821051414}
!1 = !{i64 -3421689549917153178}
!2 = !{!3, !5}
!3 = !{!4, !"notcold"}
!4 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 8632435727821051414}
!5 = !{!6, !"cold"}
!6 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 -3421689549917153178}
!7 = !{i64 9086428284934609951}
!8 = !{i64 -5964873800580613432}
!9 = !{i64 2732490490862098848}


; DUMP: CCG before cloning:
; DUMP: Callsite Context Graph:
; DUMP: Node [[BAR:0x[a-z0-9]+]]
; DUMP: 	  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #6	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[BAZ:0x[a-z0-9]+]] AllocTypes: NotColdCold ContextIds: 1 2

; DUMP: Node [[BAZ]]
; DUMP: 	  %call = call noundef ptr @_Z3barv()	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[BAZ]] AllocTypes: NotColdCold ContextIds: 1 2
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAZ]] to Caller: [[FOO:0x[a-z0-9]+]] AllocTypes: NotColdCold ContextIds: 1 2

; DUMP: Node [[FOO]]
; DUMP: 	  %call = call noundef ptr @_Z3bazv()	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAZ]] to Caller: [[FOO]] AllocTypes: NotColdCold ContextIds: 1 2
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN1:0x[a-z0-9]+]] AllocTypes: NotCold ContextIds: 1
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN2:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 2

; DUMP: Node [[MAIN1]]
; DUMP: 	  %call = call noundef ptr @_Z3foov()	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 1
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN2]]
; DUMP: 	  %call1 = call noundef ptr @_Z3foov()	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 2
; DUMP: 	CallerEdges:


; DOT: digraph "postbuild" {
; DOT: 	label="postbuild";
; DOT: 	Node[[BAR:0x[a-z0-9]+]] [shape=record,tooltip="N[[BAR]] ContextIds: 1 2",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: Alloc0\n_Z3barv -\> _Znam}"];
; DOT: 	Node[[BAZ:0x[a-z0-9]+]] [shape=record,tooltip="N[[BAZ]] ContextIds: 1 2",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: 12481870273128938184\n_Z3bazv -\> _Z3barv}"];
; DOT: 	Node[[BAZ]] -> Node[[BAR]][tooltip="ContextIds: 1 2",fillcolor="mediumorchid1"];
; DOT: 	Node[[FOO:0x[a-z0-9]+]] [shape=record,tooltip="N[[FOO]] ContextIds: 1 2",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: 2732490490862098848\n_Z3foov -\> _Z3bazv}"];
; DOT: 	Node[[FOO]] -> Node[[BAZ]][tooltip="ContextIds: 1 2",fillcolor="mediumorchid1"];
; DOT: 	Node[[MAIN1:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN1]] ContextIds: 1",fillcolor="brown1",style="filled",style="filled",label="{OrigId: 8632435727821051414\nmain -\> _Z3foov}"];
; DOT: 	Node[[MAIN1]] -> Node[[FOO]][tooltip="ContextIds: 1",fillcolor="brown1"];
; DOT: 	Node[[MAIN2:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN2]] ContextIds: 2",fillcolor="cyan",style="filled",style="filled",label="{OrigId: 15025054523792398438\nmain -\> _Z3foov}"];
; DOT: 	Node[[MAIN2]] -> Node[[FOO]][tooltip="ContextIds: 2",fillcolor="cyan"];
; DOT: }
