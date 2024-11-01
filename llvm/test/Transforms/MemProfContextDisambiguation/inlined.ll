;; Test callsite context graph generation for call graph with two memprof
;; contexts and partial inlining, requiring generation of a new fused node to
;; represent the inlined sequence while matching callsite nodes onto the graph.
;; Also tests graph and IR cloning.
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
;; Code compiled with -mllvm -memprof-ave-lifetime-cold-threshold=5 so that the
;; memory freed after sleep(10) results in cold lifetimes.
;;
;; The code below was created by forcing inlining of baz into foo, and
;; bar into baz. Due to the inlining of bar we will initially have two
;; allocation nodes in the graph. This tests that we correctly match
;; foo (with baz inlined) onto the graph nodes first, and generate a new
;; fused node for it. We should then not match baz (with bar inlined) as that
;; is not reached by the MIB contexts (since all calls from main will look
;; like main -> foo(+baz) -> bar after the inlining reflected in this IR).
;;
;; The IR was then reduced using llvm-reduce with the expected FileCheck input.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:	-memprof-verify-ccg -memprof-verify-nodes -memprof-dump-ccg \
; RUN:	-memprof-export-to-dot -memprof-dot-file-path-prefix=%t. \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation \
; RUN:	%s -S 2>&1 | FileCheck %s --check-prefix=DUMP --check-prefix=IR \
; RUN:  --check-prefix=STATS --check-prefix=REMARKS

; RUN:	cat %t.ccg.postbuild.dot | FileCheck %s --check-prefix=DOT
;; We should create clones for foo and bar for the call from main to allocate
;; cold memory.
; RUN:	cat %t.ccg.cloned.dot | FileCheck %s --check-prefix=DOTCLONED


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define internal ptr @_Z3barv() {
entry:
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #7, !memprof !0, !callsite !5
  ret ptr null
}

; Function Attrs: nobuiltin
declare ptr @_Znam(i64) #0

; Function Attrs: mustprogress
define internal ptr @_Z3bazv() #1 {
entry:
  %call.i = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #7, !memprof !0, !callsite !6
  ret ptr null
}

; Function Attrs: noinline
define internal ptr @_Z3foov() #2 {
entry:
  %call.i = call noundef ptr @_Z3barv(), !callsite !7
  ret ptr null
}

define i32 @main() #3 {
entry:
  %call = call noundef ptr @_Z3foov(), !callsite !8
  %call1 = call noundef ptr @_Z3foov(), !callsite !9
  ret i32 0
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #4

; Function Attrs: nounwind
declare void @_ZdaPv() #5

declare i32 @sleep() #6

attributes #0 = { nobuiltin }
attributes #1 = { mustprogress }
attributes #2 = { noinline }
attributes #3 = { "tune-cpu"="generic" }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #5 = { nounwind }
attributes #6 = { "disable-tail-calls"="true" }
attributes #7 = { builtin }

!0 = !{!1, !3}
!1 = !{!2, !"notcold"}
!2 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 8632435727821051414}
!3 = !{!4, !"cold"}
!4 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 -3421689549917153178}
!5 = !{i64 9086428284934609951}
!6 = !{i64 9086428284934609951, i64 -5964873800580613432}
!7 = !{i64 -5964873800580613432, i64 2732490490862098848}
!8 = !{i64 8632435727821051414}
!9 = !{i64 -3421689549917153178}


; DUMP: CCG before cloning:
; DUMP: Callsite Context Graph:
; DUMP: Node [[BAR:0x[a-z0-9]+]]
; DUMP: 	  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #7	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[FOO:0x[a-z0-9]+]] AllocTypes: NotColdCold ContextIds: 1 2

;; This is leftover from the MIB on the alloc inlined into baz. It is not
;; matched with any call, since there is no such node in the IR. Due to the
;; null call it will not participate in any context transformations.
; DUMP: Node [[FOO2:0x[a-z0-9]+]]
; DUMP: 	null Call
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 3 4
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAZ:0x[a-z0-9]+]] to Caller: [[FOO2]] AllocTypes: NotColdCold ContextIds: 3 4
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN1:0x[a-z0-9]+]] AllocTypes: NotCold ContextIds: 3
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN2:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 4

; DUMP: Node [[MAIN1]]
; DUMP: 	  %call = call noundef ptr @_Z3foov()	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1 3
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 3
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 1
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN2]]
; DUMP: 	  %call1 = call noundef ptr @_Z3foov()	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2 4
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 4
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 2
; DUMP: 	CallerEdges:

; DUMP: Node [[BAZ]]
; DUMP: 	  %call.i = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #7	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 3 4
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAZ]] to Caller: [[FOO2]] AllocTypes: NotColdCold ContextIds: 3 4

;; This is the node synthesized for the call to bar in foo that was created
;; by inlining baz into foo.
; DUMP: Node [[FOO]]
; DUMP: 	  %call.i = call noundef ptr @_Z3barv()	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[FOO]] AllocTypes: NotColdCold ContextIds: 1 2
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 1
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 2

; DUMP: CCG after cloning:
; DUMP: Callsite Context Graph:
; DUMP: Node [[BAR]]
; DUMP: 	  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #7	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[FOO]] AllocTypes: NotCold ContextIds: 1
; DUMP:         Clones: [[BAR2:0x[a-z0-9]+]]

; DUMP: Node [[FOO2]]
; DUMP: 	null Call
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 3 4
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAZ]] to Caller: [[FOO2]] AllocTypes: NotColdCold ContextIds: 3 4
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 3
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 4

; DUMP: Node [[MAIN1]]
; DUMP: 	  %call = call noundef ptr @_Z3foov()	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1 3
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 3
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 1
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN2]]
; DUMP: 	  %call1 = call noundef ptr @_Z3foov()	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2 4
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 4
; DUMP: 		Edge from Callee [[FOO3:0x[a-z0-9]+]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 2
; DUMP: 	CallerEdges:

; DUMP: Node [[BAZ]]
; DUMP: 	  %call.i = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #7	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 3 4
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAZ]] to Caller: [[FOO2]] AllocTypes: NotColdCold ContextIds: 3 4

; DUMP: Node [[FOO]]
; DUMP: 	  %call.i = call noundef ptr @_Z3barv()	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[FOO]] AllocTypes: NotCold ContextIds: 1
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 1
; DUMP:         Clones: [[FOO3]]

; DUMP: Node [[FOO3]]
; DUMP: 	  %call.i = call noundef ptr @_Z3barv()	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR2]] to Caller: [[FOO3]] AllocTypes: Cold ContextIds: 2
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO3]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 2
; DUMP:         Clone of [[FOO]]

; DUMP: Node [[BAR2]]
; DUMP: 	  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #7	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR2]] to Caller: [[FOO3]] AllocTypes: Cold ContextIds: 2
; DUMP:         Clone of [[BAR]]


; REMARKS: created clone _Z3barv.memprof.1
; REMARKS: created clone _Z3foov.memprof.1
; REMARKS: call in clone main assigned to call function clone _Z3foov.memprof.1
; REMARKS: call in clone _Z3foov.memprof.1 assigned to call function clone _Z3barv.memprof.1
; REMARKS: call in clone _Z3barv.memprof.1 marked with memprof allocation attribute cold
; REMARKS: call in clone main assigned to call function clone _Z3foov
; REMARKS: call in clone _Z3foov assigned to call function clone _Z3barv
; REMARKS: call in clone _Z3barv marked with memprof allocation attribute notcold
; REMARKS: call in clone _Z3bazv marked with memprof allocation attribute notcold


; IR: define internal {{.*}} @_Z3barv()
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[NOTCOLD:[0-9]+]]
; IR: define internal {{.*}} @_Z3foov()
; IR:   call {{.*}} @_Z3barv()
; IR: define {{.*}} @main()
;; The first call to foo does not allocate cold memory. It should call the
;; original functions, which ultimately call the original allocation decorated
;; with a "notcold" attribute.
; IR:   call {{.*}} @_Z3foov()
;; The second call to foo allocates cold memory. It should call cloned functions
;; which ultimately call a cloned allocation decorated with a "cold" attribute.
; IR:   call {{.*}} @_Z3foov.memprof.1()
; IR: define internal {{.*}} @_Z3barv.memprof.1()
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[COLD:[0-9]+]]
; IR: define internal {{.*}} @_Z3foov.memprof.1()
; IR:   call {{.*}} @_Z3barv.memprof.1()
; IR: attributes #[[NOTCOLD]] = { builtin "memprof"="notcold" }
; IR: attributes #[[COLD]] = { builtin "memprof"="cold" }


; STATS: 1 memprof-context-disambiguation - Number of cold static allocations (possibly cloned)
; STATS: 2 memprof-context-disambiguation - Number of not cold static allocations (possibly cloned)
; STATS: 2 memprof-context-disambiguation - Number of function clones created during whole program analysis


; DOT: digraph "postbuild" {
; DOT: 	label="postbuild";
; DOT: 	Node[[BAR:0x[a-z0-9]+]] [shape=record,tooltip="N[[BAR]] ContextIds: 1 2",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: Alloc0\n_Z3barv -\> _Znam}"];
; DOT: 	Node[[FOO:0x[a-z0-9]+]] [shape=record,tooltip="N[[FOO]] ContextIds: 3 4",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: 2732490490862098848\nnull call (external)}"];
; DOT: 	Node[[FOO]] -> Node[[BAZ:0x[a-z0-9]+]][tooltip="ContextIds: 3 4",fillcolor="mediumorchid1"];
; DOT: 	Node[[MAIN1:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN1]] ContextIds: 1 3",fillcolor="brown1",style="filled",style="filled",label="{OrigId: 8632435727821051414\nmain -\> _Z3foov}"];
; DOT: 	Node[[MAIN1]] -> Node[[FOO]][tooltip="ContextIds: 3",fillcolor="brown1"];
; DOT: 	Node[[MAIN1]] -> Node[[FOO2:0x[a-z0-9]+]][tooltip="ContextIds: 1",fillcolor="brown1"];
; DOT: 	Node[[MAIN2:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN2]] ContextIds: 2 4",fillcolor="cyan",style="filled",style="filled",label="{OrigId: 15025054523792398438\nmain -\> _Z3foov}"];
; DOT: 	Node[[MAIN2]] -> Node[[FOO]][tooltip="ContextIds: 4",fillcolor="cyan"];
; DOT: 	Node[[MAIN2]] -> Node[[FOO2]][tooltip="ContextIds: 2",fillcolor="cyan"];
; DOT: 	Node[[BAZ]] [shape=record,tooltip="N[[BAZ]] ContextIds: 3 4",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: Alloc2\n_Z3bazv -\> _Znam}"];
; DOT: 	Node[[FOO2]] [shape=record,tooltip="N[[FOO2]] ContextIds: 1 2",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: 0\n_Z3foov -\> _Z3barv}"];
; DOT: 	Node[[FOO2]] -> Node[[BAR]][tooltip="ContextIds: 1 2",fillcolor="mediumorchid1"];
; DOT: }


; DOTCLONED: digraph "cloned" {
; DOTCLONED: 	label="cloned";
; DOTCLONED: 	Node[[BAR:0x[a-z0-9]+]] [shape=record,tooltip="N[[BAR]] ContextIds: 1",fillcolor="brown1",style="filled",style="filled",label="{OrigId: Alloc0\n_Z3barv -\> _Znam}"];
; DOTCLONED: 	Node[[FOO2:0x[a-z0-9]+]] [shape=record,tooltip="N[[FOO2]] ContextIds: 3 4",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: 2732490490862098848\nnull call (external)}"];
; DOTCLONED: 	Node[[FOO2]] -> Node[[BAZ:0x[a-z0-9]+]][tooltip="ContextIds: 3 4",fillcolor="mediumorchid1"];
; DOTCLONED: 	Node[[MAIN1:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN1]] ContextIds: 1 3",fillcolor="brown1",style="filled",style="filled",label="{OrigId: 8632435727821051414\nmain -\> _Z3foov}"];
; DOTCLONED: 	Node[[MAIN1]] -> Node[[FOO2]][tooltip="ContextIds: 3",fillcolor="brown1"];
; DOTCLONED: 	Node[[MAIN1]] -> Node[[FOO:0x[a-z0-9]+]][tooltip="ContextIds: 1",fillcolor="brown1"];
; DOTCLONED: 	Node[[MAIN2:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN2]] ContextIds: 2 4",fillcolor="cyan",style="filled",style="filled",label="{OrigId: 15025054523792398438\nmain -\> _Z3foov}"];
; DOTCLONED: 	Node[[MAIN2]] -> Node[[FOO2]][tooltip="ContextIds: 4",fillcolor="cyan"];
; DOTCLONED: 	Node[[MAIN2]] -> Node[[FOO3:0x[a-z0-9]+]][tooltip="ContextIds: 2",fillcolor="cyan"];
; DOTCLONED: 	Node[[BAZ]] [shape=record,tooltip="N[[BAZ]] ContextIds: 3 4",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: Alloc2\n_Z3bazv -\> _Znam}"];
; DOTCLONED: 	Node[[FOO]] [shape=record,tooltip="N[[FOO]] ContextIds: 1",fillcolor="brown1",style="filled",style="filled",label="{OrigId: 0\n_Z3foov -\> _Z3barv}"];
; DOTCLONED: 	Node[[FOO]] -> Node[[BAR]][tooltip="ContextIds: 1",fillcolor="brown1"];
; DOTCLONED: 	Node[[FOO3]] [shape=record,tooltip="N[[FOO3]] ContextIds: 2",fillcolor="cyan",style="filled",color="blue",style="filled,bold,dashed",label="{OrigId: 0\n_Z3foov -\> _Z3barv}"];
; DOTCLONED: 	Node[[FOO3]] -> Node[[BAR2:0x[a-z0-9]+]][tooltip="ContextIds: 2",fillcolor="cyan"];
; DOTCLONED: 	Node[[BAR2]] [shape=record,tooltip="N[[BAR2]] ContextIds: 2",fillcolor="cyan",style="filled",color="blue",style="filled,bold,dashed",label="{OrigId: Alloc0\n_Z3barv -\> _Znam}"];
; DOTCLONED: }
