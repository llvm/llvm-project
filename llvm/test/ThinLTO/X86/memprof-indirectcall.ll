;; Tests callsite context graph generation for call graph containing indirect
;; calls. Currently this should result in conservative behavior, such that the
;; indirect call receives a null call in its graph node, to prevent subsequent
;; cloning.
;;
;; Original code looks like:
;;
;; char *foo() {
;;   return new char[10];
;; }
;; class A {
;; public:
;;     virtual char *x() { return foo(); }
;; };
;; class B : public A {
;; public:
;;     char *x() final { return foo(); }
;; };
;; char *bar(A *a) {
;;   return a->x();
;; }
;; int main(int argc, char **argv) {
;;   char *x = foo();
;;   char *y = foo();
;;   B b;
;;   char *z = bar(&b);
;;   char *w = bar(&b);
;;   A a;
;;   char *r = bar(&a);
;;   char *s = bar(&a);
;;   memset(x, 0, 10);
;;   memset(y, 0, 10);
;;   memset(z, 0, 10);
;;   memset(w, 0, 10);
;;   memset(r, 0, 10);
;;   memset(s, 0, 10);
;;   delete[] x;
;;   delete[] w;
;;   delete[] r;
;;   sleep(10);
;;   delete[] y;
;;   delete[] z;
;;   delete[] s;
;;   return 0;
;; }
;;
;; Code compiled with -mllvm -memprof-min-lifetime-cold-threshold=5 so that the
;; memory freed after sleep(10) results in cold lifetimes.
;;
;; Compiled without optimization to prevent inlining and devirtualization.
;;
;; The IR was then reduced using llvm-reduce with the expected FileCheck input.

; RUN: opt -thinlto-bc %s >%t.o
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,sleep, \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -r=%t.o,_ZdaPv, \
; RUN:  -r=%t.o,_ZTVN10__cxxabiv120__si_class_type_infoE, \
; RUN:  -r=%t.o,_ZTVN10__cxxabiv117__class_type_infoE, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes -memprof-dump-ccg \
; RUN:  -memprof-export-to-dot -memprof-dot-file-path-prefix=%t. \
; RUN:  -o %t.out 2>&1 | FileCheck %s --check-prefix=DUMP

; RUN:  cat %t.ccg.postbuild.dot | FileCheck %s --check-prefix=DOT


source_filename = "indirectcall.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZTVN10__cxxabiv120__si_class_type_infoE = external global ptr
@_ZTVN10__cxxabiv117__class_type_infoE = external global ptr

define internal ptr @_Z3barP1A(ptr %a) {
entry:
  ret ptr null
}

define i32 @main() {
entry:
  %call = call ptr @_Z3foov(), !callsite !0
  %call1 = call ptr @_Z3foov(), !callsite !1
  %call2 = call ptr @_Z3barP1A(ptr null), !callsite !2
  %call3 = call ptr @_Z3barP1A(ptr null), !callsite !3
  %call4 = call ptr @_Z3barP1A(ptr null), !callsite !4
  %call5 = call ptr @_Z3barP1A(ptr null), !callsite !5
  ret i32 0
}

declare void @_ZdaPv()

declare i32 @sleep()

define internal ptr @_ZN1A1xEv() {
entry:
  %call = call ptr @_Z3foov(), !callsite !6
  ret ptr null
}

define internal ptr @_ZN1B1xEv() {
entry:
  %call = call ptr @_Z3foov(), !callsite !7
  ret ptr null
}

define internal ptr @_Z3foov() {
entry:
  %call = call ptr @_Znam(i64 0), !memprof !8, !callsite !21
  ret ptr null
}

declare ptr @_Znam(i64)

; uselistorder directives
uselistorder ptr @_Z3foov, { 3, 2, 1, 0 }

!0 = !{i64 8632435727821051414}
!1 = !{i64 -3421689549917153178}
!2 = !{i64 6792096022461663180}
!3 = !{i64 -2709642582978494015}
!4 = !{i64 748269490701775343}
!5 = !{i64 -5747251260480066785}
!6 = !{i64 8256774051149711748}
!7 = !{i64 -4831879094954754638}
!8 = !{!9, !11, !13, !15, !17, !19}
!9 = !{!10, !"notcold"}
!10 = !{i64 2732490490862098848, i64 8256774051149711748, i64 -4820244510750103755, i64 748269490701775343}
!11 = !{!12, !"cold"}
!12 = !{i64 2732490490862098848, i64 8256774051149711748, i64 -4820244510750103755, i64 -5747251260480066785}
!13 = !{!14, !"notcold"}
!14 = !{i64 2732490490862098848, i64 8632435727821051414}
!15 = !{!16, !"cold"}
!16 = !{i64 2732490490862098848, i64 -4831879094954754638, i64 -4820244510750103755, i64 6792096022461663180}
!17 = !{!18, !"notcold"}
!18 = !{i64 2732490490862098848, i64 -4831879094954754638, i64 -4820244510750103755, i64 -2709642582978494015}
!19 = !{!20, !"cold"}
!20 = !{i64 2732490490862098848, i64 -3421689549917153178}
!21 = !{i64 2732490490862098848}


; DUMP: CCG before cloning:
; DUMP: Callsite Context Graph:
; DUMP: Node [[FOO:0x[a-z0-9]+]]
; DUMP: 	Versions: 1 MIB:
; DUMP: 		AllocType 1 StackIds: 6, 8, 4
; DUMP: 		AllocType 2 StackIds: 6, 8, 5
; DUMP: 		AllocType 1 StackIds: 0
; DUMP: 		AllocType 2 StackIds: 7, 8, 2
; DUMP: 		AllocType 1 StackIds: 7, 8, 3
; DUMP: 		AllocType 2 StackIds: 1
; DUMP: 	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2 3 4 5 6
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[AX:0x[a-z0-9]+]] AllocTypes: NotColdCold ContextIds: 1 2
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN1:0x[a-z0-9]+]] AllocTypes: NotCold ContextIds: 3
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[BX:0x[a-z0-9]+]] AllocTypes: NotColdCold ContextIds: 4 5
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN2:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 6

; DUMP: Node [[AX]]
; DUMP: 	Callee: 12914368124089294956 (_Z3foov) Clones: 0 StackIds: 6	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[AX]] AllocTypes: NotColdCold ContextIds: 1 2
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[AX]] to Caller: [[BAR:0x[a-z0-9]+]] AllocTypes: NotColdCold ContextIds: 1 2

;; Bar contains an indirect call, with multiple targets. It's call should be null.
; DUMP: Node [[BAR]]
; DUMP: 	null Call
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2 4 5
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[AX]] to Caller: [[BAR]] AllocTypes: NotColdCold ContextIds: 1 2
; DUMP: 		Edge from Callee [[BX]] to Caller: [[BAR]] AllocTypes: NotColdCold ContextIds: 4 5
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN3:0x[a-z0-9]+]] AllocTypes: NotCold ContextIds: 1
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN4:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 2
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN5:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 4
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN6:0x[a-z0-9]+]] AllocTypes: NotCold ContextIds: 5

; DUMP: Node [[MAIN3]]
; DUMP: 	Callee: 4095956691517954349 (_Z3barP1A) Clones: 0 StackIds: 4	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN3]] AllocTypes: NotCold ContextIds: 1
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN4]]
; DUMP: 	Callee: 4095956691517954349 (_Z3barP1A) Clones: 0 StackIds: 5	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN4]] AllocTypes: Cold ContextIds: 2
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN1]]
; DUMP: 	Callee: 12914368124089294956 (_Z3foov) Clones: 0 StackIds: 0	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 3
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 3
; DUMP: 	CallerEdges:

; DUMP: Node [[BX]]
; DUMP: 	Callee: 12914368124089294956 (_Z3foov) Clones: 0 StackIds: 7	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 4 5
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[BX]] AllocTypes: NotColdCold ContextIds: 4 5
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BX]] to Caller: [[BAR]] AllocTypes: NotColdCold ContextIds: 4 5

; DUMP: Node [[MAIN5]]
; DUMP: 	Callee: 4095956691517954349 (_Z3barP1A) Clones: 0 StackIds: 2	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 4
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN5]] AllocTypes: Cold ContextIds: 4
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN6]]
; DUMP: 	Callee: 4095956691517954349 (_Z3barP1A) Clones: 0 StackIds: 3	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 5
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN6]] AllocTypes: NotCold ContextIds: 5
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN2]]
; DUMP: 	Callee: 12914368124089294956 (_Z3foov) Clones: 0 StackIds: 1	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 6
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 6
; DUMP: 	CallerEdges:


; DOT: digraph "postbuild" {
; DOT: 	label="postbuild";
; DOT: 	Node[[FOO:0x[a-z0-9]+]] [shape=record,tooltip="N[[FOO]] ContextIds: 1 2 3 4 5 6",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: Alloc0\n_Z3foov -\> alloc}"];
; DOT: 	Node[[AX:0x[a-z0-9]+]] [shape=record,tooltip="N[[AX]] ContextIds: 1 2",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: 8256774051149711748\n_ZN1A1xEv -\> _Z3foov}"];
; DOT: 	Node[[AX]] -> Node[[FOO]][tooltip="ContextIds: 1 2",fillcolor="mediumorchid1"];
; DOT: 	Node[[BAR:0x[a-z0-9]+]] [shape=record,tooltip="N[[BAR]] ContextIds: 1 2 4 5",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: 13626499562959447861\nnull call (external)}"];
; DOT: 	Node[[BAR]] -> Node[[AX]][tooltip="ContextIds: 1 2",fillcolor="mediumorchid1"];
; DOT: 	Node[[BAR]] -> Node[[BX:0x[a-z0-9]+]][tooltip="ContextIds: 4 5",fillcolor="mediumorchid1"];
; DOT: 	Node[[MAIN1:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN1]] ContextIds: 1",fillcolor="brown1",style="filled",style="filled",label="{OrigId: 748269490701775343\nmain -\> _Z3barP1A}"];
; DOT: 	Node[[MAIN1]] -> Node[[BAR]][tooltip="ContextIds: 1",fillcolor="brown1"];
; DOT: 	Node[[MAIN2:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN2]] ContextIds: 2",fillcolor="cyan",style="filled",style="filled",label="{OrigId: 12699492813229484831\nmain -\> _Z3barP1A}"];
; DOT: 	Node[[MAIN2]] -> Node[[BAR]][tooltip="ContextIds: 2",fillcolor="cyan"];
; DOT: 	Node[[MAIN3:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN3]] ContextIds: 3",fillcolor="brown1",style="filled",style="filled",label="{OrigId: 8632435727821051414\nmain -\> _Z3foov}"];
; DOT: 	Node[[MAIN3]] -> Node[[FOO]][tooltip="ContextIds: 3",fillcolor="brown1"];
; DOT: 	Node[[BX]] [shape=record,tooltip="N[[BX]] ContextIds: 4 5",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: 13614864978754796978\n_ZN1B1xEv -\> _Z3foov}"];
; DOT: 	Node[[BX]] -> Node[[FOO]][tooltip="ContextIds: 4 5",fillcolor="mediumorchid1"];
; DOT: 	Node[[MAIN4:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN4]] ContextIds: 4",fillcolor="cyan",style="filled",style="filled",label="{OrigId: 6792096022461663180\nmain -\> _Z3barP1A}"];
; DOT: 	Node[[MAIN4]] -> Node[[BAR]][tooltip="ContextIds: 4",fillcolor="cyan"];
; DOT: 	Node[[MAIN5:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN5]] ContextIds: 5",fillcolor="brown1",style="filled",style="filled",label="{OrigId: 15737101490731057601\nmain -\> _Z3barP1A}"];
; DOT: 	Node[[MAIN5]] -> Node[[BAR]][tooltip="ContextIds: 5",fillcolor="brown1"];
; DOT: 	Node[[MAIN6:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN6]] ContextIds: 6",fillcolor="cyan",style="filled",style="filled",label="{OrigId: 15025054523792398438\nmain -\> _Z3foov}"];
; DOT: 	Node[[MAIN6]] -> Node[[FOO]][tooltip="ContextIds: 6",fillcolor="cyan"];
; DOT: }
