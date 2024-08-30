;; Test callsite context graph generation for call graph with with MIBs
;; that have pruned contexts that partially match multiple inlined
;; callsite contexts, requiring duplication of context ids and nodes
;; while matching callsite nodes onto the graph. Also tests graph and IR
;; cloning.
;;
;; Original code looks like:
;;
;; char *D() {
;;   return new char[10];
;; }
;;
;; char *F() {
;;   return D();
;; }
;;
;; char *C() {
;;   return D();
;; }
;;
;; char *B() {
;;   return C();
;; }
;;
;; char *E() {
;;   return C();
;; }
;; int main(int argc, char **argv) {
;;   char *x = B(); // cold
;;   char *y = E(); // cold
;;   char *z = F(); // default
;;   memset(x, 0, 10);
;;   memset(y, 0, 10);
;;   memset(z, 0, 10);
;;   delete[] z;
;;   sleep(10);
;;   delete[] x;
;;   delete[] y;
;;   return 0;
;; }
;;
;; Code compiled with -mllvm -memprof-ave-lifetime-cold-threshold=5 so that the
;; memory freed after sleep(10) results in cold lifetimes.
;;
;; The code below was created by forcing inlining of C into both B and E.
;; Since both allocation contexts via C are cold, the matched memprof
;; metadata has the context pruned above C's callsite. This requires
;; matching the stack node for C to callsites where it was inlined (i.e.
;; the callsites in B and E that have callsite metadata that includes C's).
;; It also requires duplication of that node in the graph as well as the
;; duplication of the context ids along that path through the graph,
;; so that we can represent the duplicated (via inlining) C callsite.
;;
;; The IR was then reduced using llvm-reduce with the expected FileCheck input.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -thinlto-bc %s >%t.o
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:	-supports-hot-cold-new \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_ZdaPv, \
; RUN:  -r=%t.o,sleep, \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes -memprof-dump-ccg \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation -save-temps \
; RUN:  -o %t.out 2>&1 | FileCheck %s --check-prefix=DUMP \
; RUN:  --check-prefix=STATS --check-prefix=STATS-BE --check-prefix=REMARKS

; RUN: llvm-dis %t.out.1.4.opt.bc -o - | FileCheck %s --check-prefix=IRNODIST


;; Try again but with distributed ThinLTO
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:	-supports-hot-cold-new \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_ZdaPv, \
; RUN:  -r=%t.o,sleep, \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes -memprof-dump-ccg \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation \
; RUN:  -o %t2.out 2>&1 | FileCheck %s --check-prefix=DUMP \
; RUN:  --check-prefix=STATS


;; Check distributed index
; RUN: llvm-dis %t.o.thinlto.bc -o - | FileCheck %s --check-prefix=DISTRIB

;; Run ThinLTO backend
; RUN: opt -passes=memprof-context-disambiguation \
; RUN:  -memprof-import-summary=%t.o.thinlto.bc \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation \
; RUN:  %t.o -S 2>&1 | FileCheck %s --check-prefix=IR \
; RUN:  --check-prefix=STATS-BE --check-prefix=REMARKS

source_filename = "duplicate-context-ids.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define internal ptr @_Z1Dv() #0 {
entry:
  %call = call ptr @_Znam(i64 0), !memprof !0, !callsite !5
  ret ptr null
}

declare ptr @_Znam(i64)

define internal ptr @_Z1Fv() #0 {
entry:
  %call = call ptr @_Z1Dv(), !callsite !6
  ret ptr null
}

define internal ptr @_Z1Cv() #0 {
entry:
  %call = call ptr @_Z1Dv(), !callsite !7
  ret ptr null
}

define internal ptr @_Z1Bv() #0 {
entry:
  %call.i = call ptr @_Z1Dv(), !callsite !8
  ret ptr null
}

define internal ptr @_Z1Ev() #0 {
entry:
  %call.i = call ptr @_Z1Dv(), !callsite !9
  ret ptr null
}

define i32 @main() #0 {
entry:
  call ptr @_Z1Bv()
  call ptr @_Z1Ev()
  call ptr @_Z1Fv()
  ret i32 0
}

declare void @_ZdaPv()

declare i32 @sleep()

attributes #0 = { noinline optnone}

!0 = !{!1, !3}
!1 = !{!2, !"cold"}
!2 = !{i64 6541423618768552252, i64 -6270142974039008131}
!3 = !{!4, !"notcold"}
!4 = !{i64 6541423618768552252, i64 -4903163940066524832}
!5 = !{i64 6541423618768552252}
!6 = !{i64 -4903163940066524832}
!7 = !{i64 -6270142974039008131}
!8 = !{i64 -6270142974039008131, i64 -184525619819294889}
!9 = !{i64 -6270142974039008131, i64 1905834578520680781}


;; After adding only the alloc node memprof metadata, we only have 2 contexts.

; DUMP: CCG before updating call stack chains:
; DUMP: Callsite Context Graph:
; DUMP: Node [[D:0x[a-z0-9]+]]
; DUMP: 	Versions: 1 MIB:
; DUMP: 		AllocType 2 StackIds: 0
; DUMP: 		AllocType 1 StackIds: 1
; DUMP: 	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[D]] to Caller: [[C:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 1
; DUMP: 		Edge from Callee [[D]] to Caller: [[F:0x[a-z0-9]+]] AllocTypes: NotCold ContextIds: 2

;; After updating for callsite metadata, we should have generated context ids 3 and 4,
;; along with 2 new nodes for those callsites. All have the same allocation type
;; behavior as the original C node.

; DUMP: CCG before cloning:
; DUMP: Callsite Context Graph:
; DUMP: Node [[D]]
; DUMP: 	Versions: 1 MIB:
; DUMP: 		AllocType 2 StackIds: 0
; DUMP: 		AllocType 1 StackIds: 1
; DUMP: 	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2 3 4
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[D]] to Caller: [[F]] AllocTypes: NotCold ContextIds: 2
; DUMP: 		Edge from Callee [[D]] to Caller: [[C1:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 3
; DUMP: 		Edge from Callee [[D]] to Caller: [[C2:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 4
; DUMP: 		Edge from Callee [[D]] to Caller: [[C0:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 1

; DUMP: CCG after cloning:
; DUMP: Callsite Context Graph:
; DUMP: Node [[D]]
; DUMP:         Versions: 1 MIB:
; DUMP:                 AllocType 2 StackIds: 0
; DUMP:                 AllocType 1 StackIds: 1
; DUMP:         (clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[D]] to Caller: [[F]] AllocTypes: NotCold ContextIds: 2
; DUMP:         Clones: [[D2:0x[a-z0-9]+]]

; DUMP: Node [[D2]]
; DUMP:         Versions: 1 MIB:
; DUMP:                 AllocType 2 StackIds: 0
; DUMP:                 AllocType 1 StackIds: 1
; DUMP:         (clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 1 3 4
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[D2]] to Caller: [[C0]] AllocTypes: Cold ContextIds: 1
; DUMP: 		Edge from Callee [[D2]] to Caller: [[C1]] AllocTypes: Cold ContextIds: 3
; DUMP: 		Edge from Callee [[D2]] to Caller: [[C2]] AllocTypes: Cold ContextIds: 4
; DUMP:         Clone of [[D]]

; REMARKS: created clone _Z1Dv.memprof.1
; REMARKS: call in clone _Z1Dv marked with memprof allocation attribute notcold
; REMARKS: call in clone _Z1Dv.memprof.1 marked with memprof allocation attribute cold
; REMARKS: call in clone _Z1Bv assigned to call function clone _Z1Dv.memprof.1
; REMARKS: call in clone _Z1Ev assigned to call function clone _Z1Dv.memprof.1


;; The allocation via F does not allocate cold memory. It should call the
;; original D, which ultimately call the original allocation decorated
;; with a "notcold" attribute.
; IR: define internal {{.*}} @_Z1Dv()
; IR:   call {{.*}} @_Znam(i64 0) #[[NOTCOLD:[0-9]+]]
; IR: define internal {{.*}} @_Z1Fv()
; IR:   call {{.*}} @_Z1Dv()
;; The allocations via B and E allocate cold memory. They should call the
;; cloned D, which ultimately call the cloned allocation decorated with a
;; "cold" attribute.
; IR: define internal {{.*}} @_Z1Bv()
; IR:   call {{.*}} @_Z1Dv.memprof.1()
; IR: define internal {{.*}} @_Z1Ev()
; IR:   call {{.*}} @_Z1Dv.memprof.1()
; IR: define internal {{.*}} @_Z1Dv.memprof.1()
; IR:   call {{.*}} @_Znam(i64 0) #[[COLD:[0-9]+]]
; IR: attributes #[[NOTCOLD]] = { "memprof"="notcold" }
; IR: attributes #[[COLD]] = { "memprof"="cold" }

; IRNODIST: define internal {{.*}} @_Z1Dv.retelim()
; IRNODIST:   call {{.*}} @_Znam(i64 0) #[[NOTCOLD:[0-9]+]]
; IRNODIST: define internal {{.*}} @_Z1Fv.retelim()
; IRNODIST:   call {{.*}} @_Z1Dv.retelim()
; IRNODIST: define internal {{.*}} @_Z1Bv.retelim()
; IRNODIST:   call {{.*}} @_Z1Dv.memprof.1.retelim()
; IRNODIST: define internal {{.*}} @_Z1Ev.retelim()
; IRNODIST:   call {{.*}} @_Z1Dv.memprof.1.retelim()
; IRNODIST: define internal {{.*}} @_Z1Dv.memprof.1.retelim()
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


; DISTRIB: ^[[E:[0-9]+]] = gv: (guid: 331966645857188136, {{.*}} callsites: ((callee: ^[[D:[0-9]+]], clones: (1)
; DISTRIB: ^[[D]] = gv: (guid: 11079124245221721799, {{.*}} allocs: ((versions: (notcold, cold)
; DISTRIB: ^[[F:[0-9]+]] = gv: (guid: 11254287701717398916, {{.*}} callsites: ((callee: ^[[D]], clones: (0)
; DISTRIB: ^[[B:[0-9]+]] = gv: (guid: 13579056193435805313, {{.*}} callsites: ((callee: ^[[D]], clones: (1)
; DISTRIB: ^[[C:[0-9]+]] = gv: (guid: 15101436305866936160, {{.*}} callsites: ((callee: ^[[D:[0-9]+]], clones: (1)
