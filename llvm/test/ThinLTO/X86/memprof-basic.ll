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
;; Code compiled with -mllvm -memprof-ave-lifetime-cold-threshold=5 so that the
;; memory freed after sleep(10) results in cold lifetimes.
;;
;; The IR was then reduced using llvm-reduce with the expected FileCheck input.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -thinlto-bc %s >%t.o
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:	-supports-hot-cold-new \
; RUN:	-r=%t.o,main,plx \
; RUN:	-r=%t.o,_ZdaPv, \
; RUN:	-r=%t.o,sleep, \
; RUN:	-r=%t.o,_Znam, \
; RUN:	-memprof-verify-ccg -memprof-verify-nodes -memprof-dump-ccg \
; RUN:	-memprof-export-to-dot -memprof-dot-file-path-prefix=%t. \
; RUN:	-stats -pass-remarks=memprof-context-disambiguation -save-temps \
; RUN:	-o %t.out 2>&1 | FileCheck %s --check-prefix=DUMP \
; RUN:	--check-prefix=STATS --check-prefix=STATS-BE --check-prefix=REMARKS

; RUN:	cat %t.ccg.postbuild.dot | FileCheck %s --check-prefix=DOT
;; We should have cloned bar, baz, and foo, for the cold memory allocation.
; RUN:	cat %t.ccg.cloned.dot | FileCheck %s --check-prefix=DOTCLONED

; RUN: llvm-dis %t.out.1.4.opt.bc -o - | FileCheck %s --check-prefix=IR


;; Try again but with distributed ThinLTO
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:	-supports-hot-cold-new \
; RUN:  -thinlto-distributed-indexes \
; RUN:	-r=%t.o,main,plx \
; RUN:	-r=%t.o,_ZdaPv, \
; RUN:	-r=%t.o,sleep, \
; RUN:	-r=%t.o,_Znam, \
; RUN:	-memprof-verify-ccg -memprof-verify-nodes -memprof-dump-ccg \
; RUN:	-memprof-export-to-dot -memprof-dot-file-path-prefix=%t2. \
; RUN:	-stats -pass-remarks=memprof-context-disambiguation \
; RUN:	-o %t2.out 2>&1 | FileCheck %s --check-prefix=DUMP \
; RUN:	--check-prefix=STATS

; RUN:	cat %t2.ccg.postbuild.dot | FileCheck %s --check-prefix=DOT
;; We should have cloned bar, baz, and foo, for the cold memory allocation.
; RUN:	cat %t2.ccg.cloned.dot | FileCheck %s --check-prefix=DOTCLONED

;; Check distributed index
; RUN: llvm-dis %t.o.thinlto.bc -o - | FileCheck %s --check-prefix=DISTRIB

;; Run ThinLTO backend
; RUN: opt -passes=memprof-context-disambiguation \
; RUN:	-memprof-import-summary=%t.o.thinlto.bc \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation \
; RUN:  %t.o -S 2>&1 | FileCheck %s --check-prefix=IR \
; RUN:  --check-prefix=STATS-BE --check-prefix=REMARKS

source_filename = "memprof-basic.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() #0 {
entry:
  %call = call ptr @_Z3foov(), !callsite !0
  %call1 = call ptr @_Z3foov(), !callsite !1
  ret i32 0
}

declare void @_ZdaPv()

declare i32 @sleep()

define internal ptr @_Z3barv() #0 {
entry:
  %call = call ptr @_Znam(i64 0), !memprof !2, !callsite !7
  ret ptr null
}

declare ptr @_Znam(i64)

define internal ptr @_Z3bazv() #0 {
entry:
  %call = call ptr @_Z3barv(), !callsite !8
  ret ptr null
}

define internal ptr @_Z3foov() #0 {
entry:
  %call = call ptr @_Z3bazv(), !callsite !9
  ret ptr null
}

; uselistorder directives
uselistorder ptr @_Z3foov, { 1, 0 }

attributes #0 = { noinline optnone }

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
; DUMP: 	Versions: 1 MIB:
; DUMP: 		AllocType 1 StackIds: 2, 3, 0
; DUMP: 		AllocType 2 StackIds: 2, 3, 1
; DUMP: 	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[BAZ:0x[a-z0-9]+]] AllocTypes: NotColdCold ContextIds: 1 2

; DUMP: Node [[BAZ]]
; DUMP: 	Callee: 11481133863268513686 (_Z3barv) Clones: 0 StackIds: 2	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[BAZ]] AllocTypes: NotColdCold ContextIds: 1 2
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAZ]] to Caller: [[FOO:0x[a-z0-9]+]] AllocTypes: NotColdCold ContextIds: 1 2

; DUMP: Node [[FOO]]
; DUMP: 	Callee: 1807954217441101578 (_Z3bazv) Clones: 0 StackIds: 3	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAZ]] to Caller: [[FOO]] AllocTypes: NotColdCold ContextIds: 1 2
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN1:0x[a-z0-9]+]] AllocTypes: NotCold ContextIds: 1
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN2:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 2

; DUMP: Node [[MAIN1]]
; DUMP: 	Callee: 8107868197919466657 (_Z3foov) Clones: 0 StackIds: 0	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 1
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN2]]
; DUMP: 	Callee: 8107868197919466657 (_Z3foov) Clones: 0 StackIds: 1	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 2
; DUMP: 	CallerEdges:

; DUMP: CCG after cloning:
; DUMP: Callsite Context Graph:
; DUMP: Node [[BAR]]
; DUMP: 	Versions: 1 MIB:
; DUMP:                 AllocType 1 StackIds: 2, 3, 0
; DUMP:                 AllocType 2 StackIds: 2, 3, 1
; DUMP:         (clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[BAZ]] AllocTypes: NotCold ContextIds: 1
; DUMP:		Clones: [[BAR2:0x[a-z0-9]+]]

; DUMP: Node [[BAZ]]
; DUMP: 	Callee: 11481133863268513686 (_Z3barv) Clones: 0 StackIds: 2    (clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[BAZ]] AllocTypes: NotCold ContextIds: 1
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAZ]] to Caller: [[FOO]] AllocTypes: NotCold ContextIds: 1
; DUMP:		Clones: [[BAZ2:0x[a-z0-9]+]]

; DUMP: Node [[FOO]]
; DUMP: 	Callee: 1807954217441101578 (_Z3bazv) Clones: 0 StackIds: 3    (clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAZ]] to Caller: [[FOO]] AllocTypes: NotCold ContextIds: 1
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 1
; DUMP:		Clones: [[FOO2:0x[a-z0-9]+]]

; DUMP: Node [[MAIN1]]
; DUMP: 	Callee: 8107868197919466657 (_Z3foov) Clones: 0 StackIds: 0     (clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 1
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN2]]
; DUMP: 	Callee: 8107868197919466657 (_Z3foov) Clones: 0 StackIds: 1     (clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 2
; DUMP: 	CallerEdges:

; DUMP: Node [[FOO2]]
; DUMP: 	Callee: 1807954217441101578 (_Z3bazv) Clones: 0 StackIds: 3    (clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAZ2]] to Caller: [[FOO2]] AllocTypes: Cold ContextIds: 2
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 2
; DUMP:		Clone of [[FOO]]

; DUMP: Node [[BAZ2]]
; DUMP: 	Callee: 11481133863268513686 (_Z3barv) Clones: 0 StackIds: 2    (clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR2]] to Caller: [[BAZ2]] AllocTypes: Cold ContextIds: 2
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAZ2]] to Caller: [[FOO2]] AllocTypes: Cold ContextIds: 2
; DUMP:		Clone of [[BAZ]]

; DUMP: Node [[BAR2]]
; DUMP: 	Versions: 1 MIB:
; DUMP:                 AllocType 1 StackIds: 2, 3, 0
; DUMP:                 AllocType 2 StackIds: 2, 3, 1
; DUMP:         (clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR2]] to Caller: [[BAZ2]] AllocTypes: Cold ContextIds: 2
; DUMP:		Clone of [[BAR]]


; REMARKS: call in clone main assigned to call function clone _Z3foov.memprof.1
; REMARKS: created clone _Z3barv.memprof.1
; REMARKS: call in clone _Z3barv marked with memprof allocation attribute notcold
; REMARKS: call in clone _Z3barv.memprof.1 marked with memprof allocation attribute cold
; REMARKS: created clone _Z3bazv.memprof.1
; REMARKS: call in clone _Z3bazv.memprof.1 assigned to call function clone _Z3barv.memprof.1
; REMARKS: created clone _Z3foov.memprof.1
; REMARKS: call in clone _Z3foov.memprof.1 assigned to call function clone _Z3bazv.memprof.1


; IR: define {{.*}} @main
;; The first call to foo does not allocate cold memory. It should call the
;; original functions, which ultimately call the original allocation decorated
;; with a "notcold" attribute.
; IR:   call {{.*}} @_Z3foov()
;; The second call to foo allocates cold memory. It should call cloned functions
;; which ultimately call a cloned allocation decorated with a "cold" attribute.
; IR:   call {{.*}} @_Z3foov.memprof.1()
; IR: define internal {{.*}} @_Z3barv()
; IR:   call {{.*}} @_Znam(i64 0) #[[NOTCOLD:[0-9]+]]
; IR: define internal {{.*}} @_Z3bazv()
; IR:   call {{.*}} @_Z3barv()
; IR: define internal {{.*}} @_Z3foov()
; IR:   call {{.*}} @_Z3bazv()
; IR: define internal {{.*}} @_Z3barv.memprof.1()
; IR:   call {{.*}} @_Znam(i64 0) #[[COLD:[0-9]+]]
; IR: define internal {{.*}} @_Z3bazv.memprof.1()
; IR:   call {{.*}} @_Z3barv.memprof.1()
; IR: define internal {{.*}} @_Z3foov.memprof.1()
; IR:   call {{.*}} @_Z3bazv.memprof.1()
; IR: attributes #[[NOTCOLD]] = { "memprof"="notcold" }
; IR: attributes #[[COLD]] = { "memprof"="cold" }


; STATS: 1 memprof-context-disambiguation - Number of cold static allocations (possibly cloned)
; STATS-BE: 1 memprof-context-disambiguation - Number of cold static allocations (possibly cloned) during ThinLTO backend
; STATS: 1 memprof-context-disambiguation - Number of not cold static allocations (possibly cloned)
; STATS-BE: 1 memprof-context-disambiguation - Number of not cold static allocations (possibly cloned) during ThinLTO backend
; STATS-BE: 2 memprof-context-disambiguation - Number of allocation versions (including clones) during ThinLTO backend
; STATS: 3 memprof-context-disambiguation - Number of function clones created during whole program analysis
; STATS-BE: 3 memprof-context-disambiguation - Number of function clones created during ThinLTO backend
; STATS-BE: 3 memprof-context-disambiguation - Number of functions that had clones created during ThinLTO backend
; STATS-BE: 2 memprof-context-disambiguation - Maximum number of allocation versions created for an original allocation during ThinLTO backend
; STATS-BE: 1 memprof-context-disambiguation - Number of original (not cloned) allocations with memprof profiles during ThinLTO backend


; DOT: digraph "postbuild" {
; DOT: 	label="postbuild";
; DOT: 	Node[[BAR:0x[a-z0-9]+]] [shape=record,tooltip="N[[BAR]] ContextIds: 1 2",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: Alloc0\n_Z3barv -\> alloc}"];
; DOT: 	Node[[BAZ:0x[a-z0-9]+]] [shape=record,tooltip="N[[BAZ]] ContextIds: 1 2",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: 12481870273128938184\n_Z3bazv -\> _Z3barv}"];
; DOT: 	Node[[BAZ]] -> Node[[BAR]][tooltip="ContextIds: 1 2",fillcolor="mediumorchid1"];
; DOT: 	Node[[FOO:0x[a-z0-9]+]] [shape=record,tooltip="N[[FOO]] ContextIds: 1 2",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: 2732490490862098848\n_Z3foov -\> _Z3bazv}"];
; DOT: 	Node[[FOO]] -> Node[[BAZ]][tooltip="ContextIds: 1 2",fillcolor="mediumorchid1"];
; DOT: 	Node[[MAIN1:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN1]] ContextIds: 1",fillcolor="brown1",style="filled",style="filled",label="{OrigId: 8632435727821051414\nmain -\> _Z3foov}"];
; DOT: 	Node[[MAIN1]] -> Node[[FOO]][tooltip="ContextIds: 1",fillcolor="brown1"];
; DOT: 	Node[[MAIN2:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN2]] ContextIds: 2",fillcolor="cyan",style="filled",style="filled",label="{OrigId: 15025054523792398438\nmain -\> _Z3foov}"];
; DOT: 	Node[[MAIN2]] -> Node[[FOO]][tooltip="ContextIds: 2",fillcolor="cyan"];
; DOT: }


; DOTCLONED: digraph "cloned" {
; DOTCLONED: 	label="cloned";
; DOTCLONED: 	Node[[BAR:0x[a-z0-9]+]] [shape=record,tooltip="N[[BAR]] ContextIds: 1",fillcolor="brown1",style="filled",style="filled",label="{OrigId: Alloc0\n_Z3barv -\> alloc}"];
; DOTCLONED: 	Node[[BAZ:0x[a-z0-9]+]] [shape=record,tooltip="N[[BAZ]] ContextIds: 1",fillcolor="brown1",style="filled",style="filled",label="{OrigId: 12481870273128938184\n_Z3bazv -\> _Z3barv}"];
; DOTCLONED: 	Node[[BAZ]] -> Node[[BAR]][tooltip="ContextIds: 1",fillcolor="brown1"];
; DOTCLONED: 	Node[[FOO:0x[a-z0-9]+]] [shape=record,tooltip="N[[FOO]] ContextIds: 1",fillcolor="brown1",style="filled",style="filled",label="{OrigId: 2732490490862098848\n_Z3foov -\> _Z3bazv}"];
; DOTCLONED: 	Node[[FOO]] -> Node[[BAZ]][tooltip="ContextIds: 1",fillcolor="brown1"];
; DOTCLONED: 	Node[[MAIN1:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN1]] ContextIds: 1",fillcolor="brown1",style="filled",style="filled",label="{OrigId: 8632435727821051414\nmain -\> _Z3foov}"];
; DOTCLONED: 	Node[[MAIN1]] -> Node[[FOO]][tooltip="ContextIds: 1",fillcolor="brown1"];
; DOTCLONED: 	Node[[MAIN2:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN2]] ContextIds: 2",fillcolor="cyan",style="filled",style="filled",label="{OrigId: 15025054523792398438\nmain -\> _Z3foov}"];
; DOTCLONED: 	Node[[MAIN2]] -> Node[[FOO2:0x[a-z0-9]+]][tooltip="ContextIds: 2",fillcolor="cyan"];
; DOTCLONED: 	Node[[FOO2]] [shape=record,tooltip="N[[FOO2]] ContextIds: 2",fillcolor="cyan",style="filled",color="blue",style="filled,bold,dashed",label="{OrigId: 0\n_Z3foov -\> _Z3bazv}"];
; DOTCLONED: 	Node[[FOO2]] -> Node[[BAZ2:0x[a-z0-9]+]][tooltip="ContextIds: 2",fillcolor="cyan"];
; DOTCLONED: 	Node[[BAZ2]] [shape=record,tooltip="N[[BAZ2]] ContextIds: 2",fillcolor="cyan",style="filled",color="blue",style="filled,bold,dashed",label="{OrigId: 0\n_Z3bazv -\> _Z3barv}"];
; DOTCLONED: 	Node[[BAZ2]] -> Node[[BAR2:0x[a-z0-9]+]][tooltip="ContextIds: 2",fillcolor="cyan"];
; DOTCLONED: 	Node[[BAR2]] [shape=record,tooltip="N[[BAR2]] ContextIds: 2",fillcolor="cyan",style="filled",color="blue",style="filled,bold,dashed",label="{OrigId: Alloc0\n_Z3barv -\> alloc}"];
; DOTCLONED: }


; DISTRIB: ^[[BAZ:[0-9]+]] = gv: (guid: 1807954217441101578, {{.*}} callsites: ((callee: ^[[BAR:[0-9]+]], clones: (0, 1)
; DISTRIB: ^[[FOO:[0-9]+]] = gv: (guid: 8107868197919466657, {{.*}} callsites: ((callee: ^[[BAZ]], clones: (0, 1)
; DISTRIB: ^[[BAR]] = gv: (guid: 11481133863268513686, {{.*}} allocs: ((versions: (notcold, cold)
; DISTRIB: ^[[MAIN:[0-9]+]] = gv: (guid: 15822663052811949562, {{.*}} callsites: ((callee: ^[[FOO]], clones: (0), {{.*}} (callee: ^[[FOO]], clones: (1)
