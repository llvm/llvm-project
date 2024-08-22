;; Tests callsite context graph generation for call graph containing indirect
;; calls. Currently this should result in conservative behavior, such that the
;; indirect call receives a null call in its graph node, to prevent subsequent
;; cloning. Also tests graph and IR cloning.
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
;; Code compiled with -mllvm -memprof-ave-lifetime-cold-threshold=5 so that the
;; memory freed after sleep(10) results in cold lifetimes.
;;
;; Compiled without optimization to prevent inlining and devirtualization.
;;
;; The IR was then reduced using llvm-reduce with the expected FileCheck input.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -thinlto-bc %s >%t.o
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,sleep, \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -r=%t.o,_ZdaPv, \
; RUN:  -r=%t.o,_ZTVN10__cxxabiv120__si_class_type_infoE, \
; RUN:  -r=%t.o,_ZTVN10__cxxabiv117__class_type_infoE, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes -memprof-dump-ccg \
; RUN:  -memprof-export-to-dot -memprof-dot-file-path-prefix=%t. \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation -save-temps \
; RUN:  -o %t.out 2>&1 | FileCheck %s --check-prefix=DUMP \
; RUN:  --check-prefix=STATS --check-prefix=STATS-BE --check-prefix=REMARKS

; RUN:  cat %t.ccg.postbuild.dot | FileCheck %s --check-prefix=DOT
;; We should only create a single clone of foo, for the direct call
;; from main allocating cold memory.
; RUN:  cat %t.ccg.cloned.dot | FileCheck %s --check-prefix=DOTCLONED

; RUN: llvm-dis %t.out.1.4.opt.bc -o - | FileCheck %s --check-prefix=IRNODIST


;; Try again but with distributed ThinLTO
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_ZdaPv, \
; RUN:  -r=%t.o,sleep, \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -r=%t.o,_ZTVN10__cxxabiv120__si_class_type_infoE, \
; RUN:  -r=%t.o,_ZTVN10__cxxabiv117__class_type_infoE, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes -memprof-dump-ccg \
; RUN:  -memprof-export-to-dot -memprof-dot-file-path-prefix=%t2. \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation \
; RUN:  -o %t2.out 2>&1 | FileCheck %s --check-prefix=DUMP \
; RUN:  --check-prefix=STATS

; RUN:  cat %t.ccg.postbuild.dot | FileCheck %s --check-prefix=DOT
;; We should only create a single clone of foo, for the direct call
;; from main allocating cold memory.
; RUN:  cat %t.ccg.cloned.dot | FileCheck %s --check-prefix=DOTCLONED

;; Run ThinLTO backend
; RUN: opt -passes=memprof-context-disambiguation \
; RUN:  -memprof-import-summary=%t.o.thinlto.bc \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation \
; RUN:  %t.o -S 2>&1 | FileCheck %s --check-prefix=IR \
; RUN:  --check-prefix=STATS-BE --check-prefix=REMARKS

source_filename = "indirectcall.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZTVN10__cxxabiv120__si_class_type_infoE = external global ptr
@_ZTVN10__cxxabiv117__class_type_infoE = external global ptr

define internal ptr @_Z3barP1A(ptr %a) #0 {
entry:
  ret ptr null
}

define i32 @main() #0 {
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

define internal ptr @_ZN1A1xEv() #0 {
entry:
  %call = call ptr @_Z3foov(), !callsite !6
  ret ptr null
}

define internal ptr @_ZN1B1xEv() #0 {
entry:
  %call = call ptr @_Z3foov(), !callsite !7
  ret ptr null
}

define internal ptr @_Z3foov() #0 {
entry:
  %call = call ptr @_Znam(i64 0), !memprof !8, !callsite !21
  ret ptr null
}

declare ptr @_Znam(i64)

; uselistorder directives
uselistorder ptr @_Z3foov, { 3, 2, 1, 0 }

attributes #0 = { noinline optnone }

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
; DUMP: 	Callee: 15844184524768596045 (_Z3foov) Clones: 0 StackIds: 6	(clone 0)
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
; DUMP: 	Callee: 2040285415115148168 (_Z3barP1A) Clones: 0 StackIds: 4	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN3]] AllocTypes: NotCold ContextIds: 1
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN4]]
; DUMP: 	Callee: 2040285415115148168 (_Z3barP1A) Clones: 0 StackIds: 5	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN4]] AllocTypes: Cold ContextIds: 2
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN1]]
; DUMP: 	Callee: 15844184524768596045 (_Z3foov) Clones: 0 StackIds: 0	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 3
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 3
; DUMP: 	CallerEdges:

; DUMP: Node [[BX]]
; DUMP: 	Callee: 15844184524768596045 (_Z3foov) Clones: 0 StackIds: 7	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 4 5
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[BX]] AllocTypes: NotColdCold ContextIds: 4 5
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BX]] to Caller: [[BAR]] AllocTypes: NotColdCold ContextIds: 4 5

; DUMP: Node [[MAIN5]]
; DUMP: 	Callee: 2040285415115148168 (_Z3barP1A) Clones: 0 StackIds: 2	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 4
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN5]] AllocTypes: Cold ContextIds: 4
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN6]]
; DUMP: 	Callee: 2040285415115148168 (_Z3barP1A) Clones: 0 StackIds: 3	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 5
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN6]] AllocTypes: NotCold ContextIds: 5
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN2]]
; DUMP: 	Callee: 15844184524768596045 (_Z3foov) Clones: 0 StackIds: 1	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 6
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 6
; DUMP: 	CallerEdges:

; DUMP: CCG after cloning:
; DUMP: Callsite Context Graph:
; DUMP: Node [[FOO]]
; DUMP:         Versions: 1 MIB:
; DUMP:                 AllocType 1 StackIds: 6, 8, 4
; DUMP:                 AllocType 2 StackIds: 6, 8, 5
; DUMP:                 AllocType 1 StackIds: 0
; DUMP:                 AllocType 2 StackIds: 7, 8, 2
; DUMP:                 AllocType 1 StackIds: 7, 8, 3
; DUMP:                 AllocType 2 StackIds: 1
; DUMP:         (clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2 3 4 5
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[AX]] AllocTypes: NotColdCold ContextIds: 1 2
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[BX]] AllocTypes: NotColdCold ContextIds: 4 5
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 3
; DUMP:		Clones: [[FOO2:0x[a-z0-9]+]]

; DUMP: Node [[AX]]
; DUMP: 	Callee: 15844184524768596045 (_Z3foov) Clones: 0 StackIds: 6    (clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[AX]] AllocTypes: NotColdCold ContextIds: 1 2
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[AX]] to Caller: [[BAR]] AllocTypes: NotColdCold ContextIds: 1 2

; DUMP: Node [[BAR]]
; DUMP: 	null Call
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2 4 5
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[AX]] to Caller: [[BAR]] AllocTypes: NotColdCold ContextIds: 1 2
; DUMP: 		Edge from Callee [[BX]] to Caller: [[BAR]] AllocTypes: NotColdCold ContextIds: 4 5
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN3]] AllocTypes: NotCold ContextIds: 1
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN4]] AllocTypes: Cold ContextIds: 2
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN5]] AllocTypes: Cold ContextIds: 4
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN6]] AllocTypes: NotCold ContextIds: 5

; DUMP: Node [[MAIN3]]
; DUMP: 	Callee: 2040285415115148168 (_Z3barP1A) Clones: 0 StackIds: 4   (clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN3]] AllocTypes: NotCold ContextIds: 1
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN4]]
; DUMP: 	Callee: 2040285415115148168 (_Z3barP1A) Clones: 0 StackIds: 5   (clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN4]] AllocTypes: Cold ContextIds: 2
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN1]]
; DUMP: 	Callee: 15844184524768596045 (_Z3foov) Clones: 0 StackIds: 0    (clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 3
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 3
; DUMP: 	CallerEdges:

; DUMP: Node [[BX]]
; DUMP: 	Callee: 15844184524768596045 (_Z3foov) Clones: 0 StackIds: 7    (clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 4 5
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[BX]] AllocTypes: NotColdCold ContextIds: 4 5
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BX]] to Caller: [[BAR]] AllocTypes: NotColdCold ContextIds: 4 5

; DUMP: Node [[MAIN5]]
; DUMP: 	Callee: 2040285415115148168 (_Z3barP1A) Clones: 0 StackIds: 2   (clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 4
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN5]] AllocTypes: Cold ContextIds: 4
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN6]]
; DUMP: 	Callee: 2040285415115148168 (_Z3barP1A) Clones: 0 StackIds: 3   (clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 5
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[MAIN6]] AllocTypes: NotCold ContextIds: 5
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN2]]
; DUMP: 	Callee: 15844184524768596045 (_Z3foov) Clones: 0 StackIds: 1    (clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 6
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 6
; DUMP: 	CallerEdges:

; DUMP: Node [[FOO2]]
; DUMP:         Versions: 1 MIB:
; DUMP:                 AllocType 1 StackIds: 6, 8, 4
; DUMP:                 AllocType 2 StackIds: 6, 8, 5
; DUMP:                 AllocType 1 StackIds: 0
; DUMP:                 AllocType 2 StackIds: 7, 8, 2
; DUMP:                 AllocType 1 StackIds: 7, 8, 3
; DUMP:                 AllocType 2 StackIds: 1
; DUMP:         (clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 6
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 6
; DUMP:		Clone of [[FOO]]


; REMARKS: call in clone main assigned to call function clone _Z3foov.memprof.1
; REMARKS: created clone _Z3foov.memprof.1
; REMARKS: call in clone _Z3foov marked with memprof allocation attribute notcold
; REMARKS: call in clone _Z3foov.memprof.1 marked with memprof allocation attribute cold


; IR: define {{.*}} @main(
; IR:   call {{.*}} @_Z3foov()
;; Only the second call to foo, which allocates cold memory via direct calls,
;; is replaced with a call to a clone that calls a cold allocation.
; IR:   call {{.*}} @_Z3foov.memprof.1()
; IR:   call {{.*}} @_Z3barP1A(
; IR:   call {{.*}} @_Z3barP1A(
; IR:   call {{.*}} @_Z3barP1A(
; IR:   call {{.*}} @_Z3barP1A(
; IR: define internal {{.*}} @_Z3foov()
; IR:   call {{.*}} @_Znam(i64 0) #[[NOTCOLD:[0-9]+]]
; IR: define internal {{.*}} @_Z3foov.memprof.1()
; IR:   call {{.*}} @_Znam(i64 0) #[[COLD:[0-9]+]]
; IR: attributes #[[NOTCOLD]] = { "memprof"="notcold" }
; IR: attributes #[[COLD]] = { "memprof"="cold" }

; IRNODIST: define {{.*}} @main(
; IRNODIST:   call {{.*}} @_Z3foov.argelim()
; IRNODIST:   call {{.*}} @_Z3foov.memprof.1.argelim()
; IRNODIST:   call {{.*}} @_Z3barP1A.argelim(
; IRNODIST:   call {{.*}} @_Z3barP1A.argelim(
; IRNODIST:   call {{.*}} @_Z3barP1A.argelim(
; IRNODIST:   call {{.*}} @_Z3barP1A.argelim(
; IRNODIST: define internal {{.*}} @_Z3foov.argelim()
; IRNODIST:   call {{.*}} @_Znam(i64 0) #[[NOTCOLD:[0-9]+]]
; IRNODIST: define internal {{.*}} @_Z3foov.memprof.1.argelim()
; IRNODIST:   call {{.*}} @_Znam(i64 0) #[[COLD:[0-9]+]]
; IRNODIST: attributes #[[NOTCOLD]] = { "memprof"="notcold" }
; IRNODIST: attributes #[[COLD]] = { "memprof"="cold" }

; STATS: 1 memprof-context-disambiguation - Number of cold static allocations (possibly cloned)
; STATS-BE: 1 memprof-context-disambiguation - Number of cold static allocations (possibly cloned) during ThinLTO backend
; STATS: 1 memprof-context-disambiguation - Number of not cold static allocations (possibly cloned)
; STATS-BE: 1 memprof-context-disambiguation - Number of not cold static allocations (possibly cloned) during ThinLTO backend
; STATS-BE: 2 memprof-context-disambiguation - Number of allocation versions (including clones) during ThinLTO backend
; STATS: 1 memprof-context-disambiguation - Number of function clones created during whole program analysis
; STATS-BE: 1 memprof-context-disambiguation - Number of function clones created during ThinLTO backend
; STATS-BE: 1 memprof-context-disambiguation - Number of functions that had clones created during ThinLTO backend
; STATS-BE: 2 memprof-context-disambiguation - Maximum number of allocation versions created for an original allocation during ThinLTO backend
; STATS-BE: 1 memprof-context-disambiguation - Number of original (not cloned) allocations with memprof profiles during ThinLTO backend


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


; DOTCLONED: digraph "cloned" {
; DOTCLONED: 	label="cloned";
; DOTCLONED: 	Node[[FOO2:0x[a-z0-9]+]] [shape=record,tooltip="N[[FOO2]] ContextIds: 1 2 3 4 5",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: Alloc0\n_Z3foov -\> alloc}"];
; DOTCLONED: 	Node[[AX:0x[a-z0-9]+]] [shape=record,tooltip="N[[AX]] ContextIds: 1 2",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: 8256774051149711748\n_ZN1A1xEv -\> _Z3foov}"];
; DOTCLONED: 	Node[[AX]] -> Node[[FOO2]][tooltip="ContextIds: 1 2",fillcolor="mediumorchid1"];
; DOTCLONED: 	Node[[BAR:0x[a-z0-9]+]] [shape=record,tooltip="N[[BAR]] ContextIds: 1 2 4 5",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: 13626499562959447861\nnull call (external)}"];
; DOTCLONED: 	Node[[BAR]] -> Node[[AX]][tooltip="ContextIds: 1 2",fillcolor="mediumorchid1"];
; DOTCLONED: 	Node[[BAR]] -> Node[[BX:0x[a-z0-9]+]][tooltip="ContextIds: 4 5",fillcolor="mediumorchid1"];
; DOTCLONED: 	Node[[MAIN1:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN1]] ContextIds: 1",fillcolor="brown1",style="filled",style="filled",label="{OrigId: 748269490701775343\nmain -\> _Z3barP1A}"];
; DOTCLONED: 	Node[[MAIN1]] -> Node[[BAR]][tooltip="ContextIds: 1",fillcolor="brown1"];
; DOTCLONED: 	Node[[MAIN2:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN2]] ContextIds: 2",fillcolor="cyan",style="filled",style="filled",label="{OrigId: 12699492813229484831\nmain -\> _Z3barP1A}"];
; DOTCLONED: 	Node[[MAIN2]] -> Node[[BAR]][tooltip="ContextIds: 2",fillcolor="cyan"];
; DOTCLONED: 	Node[[MAIN3:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN3]] ContextIds: 3",fillcolor="brown1",style="filled",style="filled",label="{OrigId: 8632435727821051414\nmain -\> _Z3foov}"];
; DOTCLONED: 	Node[[MAIN3]] -> Node[[FOO2]][tooltip="ContextIds: 3",fillcolor="brown1"];
; DOTCLONED: 	Node[[BX]] [shape=record,tooltip="N[[BX]] ContextIds: 4 5",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: 13614864978754796978\n_ZN1B1xEv -\> _Z3foov}"];
; DOTCLONED: 	Node[[BX]] -> Node[[FOO2]][tooltip="ContextIds: 4 5",fillcolor="mediumorchid1"];
; DOTCLONED: 	Node[[MAIN4:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN4]] ContextIds: 4",fillcolor="cyan",style="filled",style="filled",label="{OrigId: 6792096022461663180\nmain -\> _Z3barP1A}"];
; DOTCLONED: 	Node[[MAIN4]] -> Node[[BAR]][tooltip="ContextIds: 4",fillcolor="cyan"];
; DOTCLONED: 	Node[[MAIN5:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN5]] ContextIds: 5",fillcolor="brown1",style="filled",style="filled",label="{OrigId: 15737101490731057601\nmain -\> _Z3barP1A}"];
; DOTCLONED: 	Node[[MAIN5]] -> Node[[BAR]][tooltip="ContextIds: 5",fillcolor="brown1"];
; DOTCLONED: 	Node[[MAIN6:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN6]] ContextIds: 6",fillcolor="cyan",style="filled",style="filled",label="{OrigId: 15025054523792398438\nmain -\> _Z3foov}"];
; DOTCLONED: 	Node[[MAIN6]] -> Node[[FOO2:0x[a-z0-9]+]][tooltip="ContextIds: 6",fillcolor="cyan"];
; DOTCLONED: 	Node[[FOO2]] [shape=record,tooltip="N[[FOO2]] ContextIds: 6",fillcolor="cyan",style="filled",color="blue",style="filled,bold,dashed",label="{OrigId: Alloc0\n_Z3foov -\> alloc}"];
; DOTCLONED: }
