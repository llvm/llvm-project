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

; RUN: opt -thinlto-bc %s >%t.o
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:	-r=%t.o,main,plx \
; RUN:	-r=%t.o,_ZdaPv, \
; RUN:	-r=%t.o,sleep, \
; RUN:	-r=%t.o,_Znam, \
; RUN:	-memprof-verify-ccg -memprof-verify-nodes -memprof-dump-ccg \
; RUN:	-memprof-export-to-dot -memprof-dot-file-path-prefix=%t. \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation -save-temps \
; RUN:	-o %t.out 2>&1 | FileCheck %s --check-prefix=DUMP \
; RUN:  --check-prefix=STATS --check-prefix=STATS-BE \
; RUN:  --check-prefix=STATS-INPROCESS-BE --check-prefix=REMARKS

; RUN:	cat %t.ccg.postbuild.dot | FileCheck %s --check-prefix=DOT
;; We should create clones for foo and bar for the call from main to allocate
;; cold memory.
; RUN:	cat %t.ccg.cloned.dot | FileCheck %s --check-prefix=DOTCLONED

; RUN: llvm-dis %t.out.1.4.opt.bc -o - | FileCheck %s --check-prefix=IR


;; Try again but with distributed ThinLTO
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_ZdaPv, \
; RUN:  -r=%t.o,sleep, \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes -memprof-dump-ccg \
; RUN:  -memprof-export-to-dot -memprof-dot-file-path-prefix=%t2. \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation \
; RUN:  -o %t2.out 2>&1 | FileCheck %s --check-prefix=DUMP \
; RUN:  --check-prefix=STATS

; RUN:	cat %t.ccg.postbuild.dot | FileCheck %s --check-prefix=DOT
;; We should create clones for foo and bar for the call from main to allocate
;; cold memory.
; RUN:	cat %t.ccg.cloned.dot | FileCheck %s --check-prefix=DOTCLONED

;; Run ThinLTO backend
; RUN: opt -passes=memprof-context-disambiguation \
; RUN:  -memprof-import-summary=%t.o.thinlto.bc \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation \
; RUN:  %t.o -S 2>&1 | FileCheck %s --check-prefix=IR \
; RUN:  --check-prefix=STATS-BE --check-prefix=STATS-DISTRIB-BE \
; RUN:  --check-prefix=REMARKS

source_filename = "inlined.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define internal ptr @_Z3barv() #0 {
entry:
  %call = call ptr @_Znam(i64 0), !memprof !0, !callsite !5
  ret ptr null
}

declare ptr @_Znam(i64)

define internal ptr @_Z3bazv() #0 {
entry:
  %call.i = call ptr @_Znam(i64 0), !memprof !0, !callsite !6
  ret ptr null
}

define internal ptr @_Z3foov() #0 {
entry:
  %call.i = call ptr @_Z3barv(), !callsite !7
  ret ptr null
}

define i32 @main() #0 {
entry:
  %call = call ptr @_Z3foov(), !callsite !8
  %call1 = call ptr @_Z3foov(), !callsite !9
  ret i32 0
}

declare void @_ZdaPv()

declare i32 @sleep()

attributes #0 = { noinline optnone }

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

; DUMP: Node [[BAZ:0x[a-z0-9]+]]
; DUMP: 	Versions: 1 MIB:
; DUMP: 		AllocType 1 StackIds: 1, 2
; DUMP: 		AllocType 2 StackIds: 1, 3
; DUMP: 	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAZ]] to Caller: [[FOO2:0x[a-z0-9]+]] AllocTypes: NotColdCold ContextIds: 1 2

;; This is leftover from the MIB on the alloc inlined into baz. It is not
;; matched with any call, since there is no such node in the IR. Due to the
;; null call it will not participate in any context transformations.
; DUMP: Node [[FOO2]]
; DUMP: 	null Call
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAZ]] to Caller: [[FOO2]] AllocTypes: NotColdCold ContextIds: 1 2
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN1:0x[a-z0-9]+]] AllocTypes: NotCold ContextIds: 1
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN2:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 2

; DUMP: Node [[MAIN1]]
; DUMP: 	Callee: 2229562716906371625 (_Z3foov) Clones: 0 StackIds: 2	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1 3
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 1
; DUMP: 		Edge from Callee [[FOO:0x[a-z0-9]+]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 3
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN2]]
; DUMP: 	Callee: 2229562716906371625 (_Z3foov) Clones: 0 StackIds: 3	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2 4
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 2
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 4
; DUMP: 	CallerEdges:

; DUMP: Node [[BAR:0x[a-z0-9]+]]
; DUMP: 	Versions: 1 MIB:
; DUMP: 		AllocType 1 StackIds: 0, 1, 2
; DUMP: 		AllocType 2 StackIds: 0, 1, 3
; DUMP: 	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 3 4
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[FOO]] AllocTypes: NotColdCold ContextIds: 3 4

;; This is the node synthesized for the call to bar in foo that was created
;; by inlining baz into foo.
; DUMP: Node [[FOO]]
; DUMP: 	Callee: 16064618363798697104 (_Z3barv) Clones: 0 StackIds: 0, 1	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 3 4
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[FOO]] AllocTypes: NotColdCold ContextIds: 3 4
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 3
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 4

; DUMP: CCG after cloning:
; DUMP: Callsite Context Graph:
; DUMP: Node [[BAZ]]
; DUMP:         Versions: 1 MIB:
; DUMP:                 AllocType 1 StackIds: 1, 2
; DUMP:                 AllocType 2 StackIds: 1, 3
; DUMP:         (clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAZ]] to Caller: [[FOO2]] AllocTypes: NotColdCold ContextIds: 1 2

; DUMP: Node [[FOO2]]
; DUMP: 	null Call
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAZ]] to Caller: [[FOO2]] AllocTypes: NotColdCold ContextIds: 1 2
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 1
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 2

; DUMP: Node [[MAIN1]]
; DUMP:         Callee: 2229562716906371625 (_Z3foov) Clones: 0 StackIds: 2     (clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1 3
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 1
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 3
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN2]]
; DUMP:         Callee: 2229562716906371625 (_Z3foov) Clones: 0 StackIds: 3     (clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2 4
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 2
; DUMP: 		Edge from Callee [[FOO3:0x[a-z0-9]+]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 4
; DUMP: 	CallerEdges:

; DUMP: Node [[BAR]]
; DUMP:         Versions: 1 MIB:
; DUMP:                 AllocType 1 StackIds: 0, 1, 2
; DUMP:                 AllocType 2 StackIds: 0, 1, 3
; DUMP:         (clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 3
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[FOO]] AllocTypes: NotCold ContextIds: 3
; DUMP:         Clones: [[BAR2:0x[a-z0-9]+]]

; DUMP: Node [[FOO]]
; DUMP:         Callee: 16064618363798697104 (_Z3barv) Clones: 0 StackIds: 0, 1 (clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 3
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[FOO]] AllocTypes: NotCold ContextIds: 3
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 3
; DUMP:         Clones: [[FOO3]]

; DUMP: Node [[FOO3]]
; DUMP:         Callee: 16064618363798697104 (_Z3barv) Clones: 0 StackIds: 0, 1 (clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 4
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR2]] to Caller: [[FOO3]] AllocTypes: Cold ContextIds: 4
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO3]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 4
; DUMP:         Clone of [[FOO]]

; DUMP: Node [[BAR2]]
; DUMP:         Versions: 1 MIB:
; DUMP:                 AllocType 1 StackIds: 0, 1, 2
; DUMP:                 AllocType 2 StackIds: 0, 1, 3
; DUMP:         (clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 4
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR2]] to Caller: [[FOO3]] AllocTypes: Cold ContextIds: 4
; DUMP:         Clone of [[BAR]]


; REMARKS: created clone _Z3barv.memprof.1
; REMARKS: call in clone _Z3barv marked with memprof allocation attribute notcold
; REMARKS: call in clone _Z3barv.memprof.1 marked with memprof allocation attribute cold
; REMARKS: created clone _Z3foov.memprof.1
; REMARKS: call in clone _Z3foov.memprof.1 assigned to call function clone _Z3barv.memprof.1
; REMARKS: call in clone main assigned to call function clone _Z3foov.memprof.1


; IR: define internal {{.*}} @_Z3barv()
; IR:   call {{.*}} @_Znam(i64 0) #[[NOTCOLD:[0-9]+]]
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
; IR:   call {{.*}} @_Znam(i64 0) #[[COLD:[0-9]+]]
; IR: define internal {{.*}} @_Z3foov.memprof.1()
; IR:   call {{.*}} @_Z3barv.memprof.1()
; IR: attributes #[[NOTCOLD]] = { "memprof"="notcold" }
; IR: attributes #[[COLD]] = { "memprof"="cold" }


; STATS: 1 memprof-context-disambiguation - Number of cold static allocations (possibly cloned)
; STATS-BE: 1 memprof-context-disambiguation - Number of cold static allocations (possibly cloned) during ThinLTO backend
; STATS: 2 memprof-context-disambiguation - Number of not cold static allocations (possibly cloned)
; STATS-BE: 1 memprof-context-disambiguation - Number of not cold static allocations (possibly cloned) during ThinLTO backend
; STATS-INPROCESS-BE: 2 memprof-context-disambiguation - Number of allocation versions (including clones) during ThinLTO backend
;; The distributed backend hasn't yet eliminated the now-dead baz with
;; the allocation from bar inlined, so it has one more allocation.
; STATS-DISTRIB-BE: 3 memprof-context-disambiguation - Number of allocation versions (including clones) during ThinLTO backend
; STATS: 2 memprof-context-disambiguation - Number of function clones created during whole program analysis
; STATS-BE: 2 memprof-context-disambiguation - Number of function clones created during ThinLTO backend
; STATS-BE: 2 memprof-context-disambiguation - Number of functions that had clones created during ThinLTO backend
; STATS-BE: 2 memprof-context-disambiguation - Maximum number of allocation versions created for an original allocation during ThinLTO backend
; STATS-INPROCESS-BE: 1 memprof-context-disambiguation - Number of original (not cloned) allocations with memprof profiles during ThinLTO backend
;; The distributed backend hasn't yet eliminated the now-dead baz with
;; the allocation from bar inlined, so it has one more allocation.
; STATS-DISTRIB-BE: 2 memprof-context-disambiguation - Number of original (not cloned) allocations with memprof profiles during ThinLTO backend


; DOT: digraph "postbuild" {
; DOT: 	label="postbuild";
; DOT: 	Node[[BAZ:0x[a-z0-9]+]] [shape=record,tooltip="N[[BAZ]] ContextIds: 1 2",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: Alloc0\n_Z3bazv -\> alloc}"];
; DOT: 	Node[[FOO:0x[a-z0-9]+]] [shape=record,tooltip="N[[FOO]] ContextIds: 1 2",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: 2732490490862098848\nnull call (external)}"];
; DOT: 	Node[[FOO]] -> Node[[BAZ]][tooltip="ContextIds: 1 2",fillcolor="mediumorchid1"];
; DOT: 	Node[[MAIN1:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN1]] ContextIds: 1 3",fillcolor="brown1",style="filled",style="filled",label="{OrigId: 8632435727821051414\nmain -\> _Z3foov}"];
; DOT: 	Node[[MAIN1]] -> Node[[FOO]][tooltip="ContextIds: 1",fillcolor="brown1"];
; DOT: 	Node[[MAIN1]] -> Node[[FOO2:0x[a-z0-9]+]][tooltip="ContextIds: 3",fillcolor="brown1"];
; DOT: 	Node[[MAIN2:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN2]] ContextIds: 2 4",fillcolor="cyan",style="filled",style="filled",label="{OrigId: 15025054523792398438\nmain -\> _Z3foov}"];
; DOT: 	Node[[MAIN2]] -> Node[[FOO]][tooltip="ContextIds: 2",fillcolor="cyan"];
; DOT: 	Node[[MAIN2]] -> Node[[FOO2]][tooltip="ContextIds: 4",fillcolor="cyan"];
; DOT: 	Node[[BAR:0x[a-z0-9]+]] [shape=record,tooltip="N[[BAR]] ContextIds: 3 4",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: Alloc2\n_Z3barv -\> alloc}"];
; DOT: 	Node[[FOO2]] [shape=record,tooltip="N[[FOO2]] ContextIds: 3 4",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: 0\n_Z3foov -\> _Z3barv}"];
; DOT: 	Node[[FOO2]] -> Node[[BAR]][tooltip="ContextIds: 3 4",fillcolor="mediumorchid1"];
; DOT: }


; DOTCLONED: digraph "cloned" {
; DOTCLONED: 	label="cloned";
; DOTCLONED: 	Node[[BAZ:0x[a-z0-9]+]] [shape=record,tooltip="N[[BAZ]] ContextIds: 1 2",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: Alloc0\n_Z3bazv -\> alloc}"];
; DOTCLONED: 	Node[[FOO2:0x[a-z0-9]+]] [shape=record,tooltip="N[[FOO2]] ContextIds: 1 2",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: 2732490490862098848\nnull call (external)}"];
; DOTCLONED: 	Node[[FOO2]] -> Node[[BAZ]][tooltip="ContextIds: 1 2",fillcolor="mediumorchid1"];
; DOTCLONED: 	Node[[MAIN1:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN1]] ContextIds: 1 3",fillcolor="brown1",style="filled",style="filled",label="{OrigId: 8632435727821051414\nmain -\> _Z3foov}"];
; DOTCLONED: 	Node[[MAIN1]] -> Node[[FOO2]][tooltip="ContextIds: 1",fillcolor="brown1"];
; DOTCLONED: 	Node[[MAIN1]] -> Node[[FOO:0x[a-z0-9]+]][tooltip="ContextIds: 3",fillcolor="brown1"];
; DOTCLONED: 	Node[[MAIN2:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN2]] ContextIds: 2 4",fillcolor="cyan",style="filled",style="filled",label="{OrigId: 15025054523792398438\nmain -\> _Z3foov}"];
; DOTCLONED: 	Node[[MAIN2]] -> Node[[FOO2]][tooltip="ContextIds: 2",fillcolor="cyan"];
; DOTCLONED: 	Node[[MAIN2]] -> Node[[FOO3:0x[a-z0-9]+]][tooltip="ContextIds: 4",fillcolor="cyan"];
; DOTCLONED: 	Node[[BAR:0x[a-z0-9]+]] [shape=record,tooltip="N[[BAR]] ContextIds: 3",fillcolor="brown1",style="filled",style="filled",label="{OrigId: Alloc2\n_Z3barv -\> alloc}"];
; DOTCLONED: 	Node[[FOO]] [shape=record,tooltip="N[[FOO]] ContextIds: 3",fillcolor="brown1",style="filled",style="filled",label="{OrigId: 0\n_Z3foov -\> _Z3barv}"];
; DOTCLONED: 	Node[[FOO]] -> Node[[BAR]][tooltip="ContextIds: 3",fillcolor="brown1"];
; DOTCLONED: 	Node[[FOO3]] [shape=record,tooltip="N[[FOO3]] ContextIds: 4",fillcolor="cyan",style="filled",color="blue",style="filled,bold,dashed",label="{OrigId: 0\n_Z3foov -\> _Z3barv}"];
; DOTCLONED: 	Node[[FOO3]] -> Node[[BAR2:0x[a-z0-9]+]][tooltip="ContextIds: 4",fillcolor="cyan"];
; DOTCLONED: 	Node[[BAR2]] [shape=record,tooltip="N[[BAR2]] ContextIds: 4",fillcolor="cyan",style="filled",color="blue",style="filled,bold,dashed",label="{OrigId: Alloc0\n_Z3barv -\> alloc}"];
; DOTCLONED: }
