;; Test context disambiguation for a callgraph containing multiple memprof
;; contexts and no inlining, where we need to perform additional cloning
;; during function assignment/cloning to handle the combination of contexts
;; to 2 different allocations.
;;
;; void E(char **buf1, char **buf2) {
;;   *buf1 = new char[10];
;;   *buf2 = new char[10];
;; }
;;
;; void B(char **buf1, char **buf2) {
;;   E(buf1, buf2);
;; }
;;
;; void C(char **buf1, char **buf2) {
;;   E(buf1, buf2);
;; }
;;
;; void D(char **buf1, char **buf2) {
;;   E(buf1, buf2);
;; }
;; int main(int argc, char **argv) {
;;   char *cold1, *cold2, *default1, *default2, *default3, *default4;
;;   B(&default1, &default2);
;;   C(&default3, &cold1);
;;   D(&cold2, &default4);
;;   memset(cold1, 0, 10);
;;   memset(cold2, 0, 10);
;;   memset(default1, 0, 10);
;;   memset(default2, 0, 10);
;;   memset(default3, 0, 10);
;;   memset(default4, 0, 10);
;;   delete[] default1;
;;   delete[] default2;
;;   delete[] default3;
;;   delete[] default4;
;;   sleep(10);
;;   delete[] cold1;
;;   delete[] cold2;
;;   return 0;
;; }
;;
;; Code compiled with -mllvm -memprof-min-lifetime-cold-threshold=5 so that the
;; memory freed after sleep(10) results in cold lifetimes.
;;
;; The IR was then reduced using llvm-reduce with the expected FileCheck input.

;; -stats requires asserts
; REQUIRES: asserts


; RUN: opt -thinlto-bc %s >%t.o
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_ZdaPv, \
; RUN:  -r=%t.o,sleep, \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes -memprof-dump-ccg \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation -save-temps \
; RUN:  -o %t.out 2>&1 | FileCheck %s --check-prefix=DUMP \
; RUN:  --check-prefix=STATS --check-prefix=STATS-BE --check-prefix=REMARKS

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
; RUN:  -stats -pass-remarks=memprof-context-disambiguation \
; RUN:  -o %t2.out 2>&1 | FileCheck %s --check-prefix=DUMP \
; RUN:  --check-prefix=STATS

;; Run ThinLTO backend
; RUN: opt -passes=memprof-context-disambiguation \
; RUN:  -memprof-import-summary=%t.o.thinlto.bc \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation \
; RUN:  %t.o -S 2>&1 | FileCheck %s --check-prefix=IR \
; RUN:  --check-prefix=STATS-BE --check-prefix=REMARKS


source_filename = "funcassigncloning.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline optnone
define internal void @_Z1EPPcS0_(ptr %buf1, ptr %buf2) #0 {
entry:
  %call = call ptr @_Znam(i64 noundef 10), !memprof !0, !callsite !7
  %call1 = call ptr @_Znam(i64 noundef 10), !memprof !8, !callsite !15
  ret void
}

declare ptr @_Znam(i64)

define internal void @_Z1BPPcS0_() {
entry:
  call void @_Z1EPPcS0_(ptr null, ptr null), !callsite !16
  ret void
}

define internal void @_Z1CPPcS0_() {
entry:
  call void @_Z1EPPcS0_(ptr null, ptr null), !callsite !17
  ret void
}

define internal void @_Z1DPPcS0_() {
entry:
  call void @_Z1EPPcS0_(ptr null, ptr null), !callsite !18
  ret void
}

; Function Attrs: noinline optnone
define i32 @main() #0 {
entry:
  call void @_Z1BPPcS0_()
  call void @_Z1CPPcS0_()
  call void @_Z1DPPcS0_()
  ret i32 0
}

declare void @_ZdaPv()

declare i32 @sleep()

; uselistorder directives
uselistorder ptr @_Znam, { 1, 0 }

attributes #0 = { noinline optnone }

!0 = !{!1, !3, !5}
!1 = !{!2, !"cold"}
!2 = !{i64 -3461278137325233666, i64 -7799663586031895603}
!3 = !{!4, !"notcold"}
!4 = !{i64 -3461278137325233666, i64 -3483158674395044949}
!5 = !{!6, !"notcold"}
!6 = !{i64 -3461278137325233666, i64 -2441057035866683071}
!7 = !{i64 -3461278137325233666}
!8 = !{!9, !11, !13}
!9 = !{!10, !"notcold"}
!10 = !{i64 -1415475215210681400, i64 -2441057035866683071}
!11 = !{!12, !"cold"}
!12 = !{i64 -1415475215210681400, i64 -3483158674395044949}
!13 = !{!14, !"notcold"}
!14 = !{i64 -1415475215210681400, i64 -7799663586031895603}
!15 = !{i64 -1415475215210681400}
!16 = !{i64 -2441057035866683071}
!17 = !{i64 -3483158674395044949}
!18 = !{i64 -7799663586031895603}


;; Originally we create a single clone of each call to new from E, since each
;; allocates cold memory for a single caller.

; DUMP: CCG after cloning:
; DUMP: Callsite Context Graph:
; DUMP: Node [[ENEW1ORIG:0x[a-z0-9]+]]
; DUMP:         Versions: 1 MIB:
; DUMP:                 AllocType 2 StackIds: 0
; DUMP:                 AllocType 1 StackIds: 1
; DUMP:                 AllocType 1 StackIds: 2
; DUMP:         (clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 2 3
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[ENEW1ORIG]] to Caller: [[C:0x[a-z0-9]+]] AllocTypes: NotCold ContextIds: 2
; DUMP: 		Edge from Callee [[ENEW1ORIG]] to Caller: [[B:0x[a-z0-9]+]] AllocTypes: NotCold ContextIds: 3
; DUMP: 	Clones: [[ENEW1CLONE:0x[a-z0-9]+]]

; DUMP: Node [[D:0x[a-z0-9]+]]
; DUMP: 	Callee: 10758063066234039248 (_Z1EPPcS0_) Clones: 0 StackIds: 0 (clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 6
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[ENEW1CLONE]] to Caller: [[D]] AllocTypes: Cold ContextIds: 1
; DUMP: 		Edge from Callee [[ENEW2ORIG:0x[a-z0-9]+]] to Caller: [[D]] AllocTypes: NotCold ContextIds: 6
; DUMP: 	CallerEdges:

; DUMP: Node [[C]]
; DUMP: 	Callee: 10758063066234039248 (_Z1EPPcS0_) Clones: 0 StackIds: 1 (clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 2 5
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[ENEW1ORIG]] to Caller: [[C]] AllocTypes: NotCold ContextIds: 2
; DUMP: 		Edge from Callee [[ENEW2CLONE:0x[a-z0-9]+]] to Caller: [[C]] AllocTypes: Cold ContextIds: 5
; DUMP: 	CallerEdges:

; DUMP: Node [[B]]
; DUMP: 	Callee: 10758063066234039248 (_Z1EPPcS0_) Clones: 0 StackIds: 2 (clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 3 4
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[ENEW1ORIG]] to Caller: [[B]] AllocTypes: NotCold ContextIds: 3
; DUMP: 		Edge from Callee [[ENEW2ORIG]] to Caller: [[B]] AllocTypes: NotCold ContextIds: 4
; DUMP: 	CallerEdges:

; DUMP: Node [[ENEW2ORIG]]
; DUMP:         Versions: 1 MIB:
; DUMP:                 AllocType 1 StackIds: 2
; DUMP:                 AllocType 2 StackIds: 1
; DUMP:                 AllocType 1 StackIds: 0
; DUMP:         (clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 4 6
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[ENEW2ORIG]] to Caller: [[B]] AllocTypes: NotCold ContextIds: 4
; DUMP: 		Edge from Callee [[ENEW2ORIG]] to Caller: [[D]] AllocTypes: NotCold ContextIds: 6
; DUMP: 	Clones: [[ENEW2CLONE]]

; DUMP: Node [[ENEW1CLONE]]
; DUMP:         Versions: 1 MIB:
; DUMP:                 AllocType 2 StackIds: 0
; DUMP:                 AllocType 1 StackIds: 1
; DUMP:                 AllocType 1 StackIds: 2
; DUMP:         (clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[ENEW1CLONE]] to Caller: [[D]] AllocTypes: Cold ContextIds: 1
; DUMP: 	Clone of [[ENEW1ORIG]]

; DUMP: Node [[ENEW2CLONE]]
; DUMP:         Versions: 1 MIB:
; DUMP:                 AllocType 1 StackIds: 2
; DUMP:                 AllocType 2 StackIds: 1
; DUMP:                 AllocType 1 StackIds: 0
; DUMP:         (clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 5
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[ENEW2CLONE]] to Caller: [[C]] AllocTypes: Cold ContextIds: 5
; DUMP: 	Clone of [[ENEW2ORIG]]


;; We greedily create a clone of E that is initially used by the clones of the
;; first call to new. However, we end up with an incompatible set of callers
;; given the second call to new which has clones with a different combination of
;; callers. Eventually, we create 2 more clones, and the first clone becomes dead.
; REMARKS: created clone _Z1EPPcS0_.memprof.1
; REMARKS: created clone _Z1EPPcS0_.memprof.2
; REMARKS: created clone _Z1EPPcS0_.memprof.3
; REMARKS: call in clone _Z1EPPcS0_ marked with memprof allocation attribute notcold
; REMARKS: call in clone _Z1EPPcS0_.memprof.2 marked with memprof allocation attribute cold
; REMARKS: call in clone _Z1EPPcS0_.memprof.3 marked with memprof allocation attribute notcold
; REMARKS: call in clone _Z1EPPcS0_ marked with memprof allocation attribute notcold
; REMARKS: call in clone _Z1EPPcS0_.memprof.2 marked with memprof allocation attribute notcold
; REMARKS: call in clone _Z1EPPcS0_.memprof.3 marked with memprof allocation attribute cold
; REMARKS: call in clone _Z1CPPcS0_ assigned to call function clone _Z1EPPcS0_.memprof.3
; REMARKS: call in clone _Z1DPPcS0_ assigned to call function clone _Z1EPPcS0_.memprof.2


;; Original version of E is used for the non-cold allocations, both from B.
; IR: define internal {{.*}} @_Z1EPPcS0_(
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[NOTCOLD:[0-9]+]]
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[NOTCOLD]]
; IR: define internal {{.*}} @_Z1BPPcS0_(
; IR:   call {{.*}} @_Z1EPPcS0_(
;; C calls a clone of E with the first new allocating cold memory and the
;; second allocating non-cold memory.
; IR: define internal {{.*}} @_Z1CPPcS0_(
; IR:   call {{.*}} @_Z1EPPcS0_.memprof.3(
;; D calls a clone of E with the first new allocating non-cold memory and the
;; second allocating cold memory.
; IR: define internal {{.*}} @_Z1DPPcS0_(
; IR:   call {{.*}} @_Z1EPPcS0_.memprof.2(
; IR: define internal {{.*}} @_Z1EPPcS0_.memprof.2(
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[COLD:[0-9]+]]
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[NOTCOLD]]
; IR: define internal {{.*}} @_Z1EPPcS0_.memprof.3(
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[NOTCOLD]]
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[COLD]]
; IR: attributes #[[NOTCOLD]] = { "memprof"="notcold" }
; IR: attributes #[[COLD]] = { "memprof"="cold" }


; STATS: 2 memprof-context-disambiguation - Number of cold static allocations (possibly cloned)
; STATS-BE: 2 memprof-context-disambiguation - Number of cold static allocations (possibly cloned) during ThinLTO backend
; STATS: 4 memprof-context-disambiguation - Number of not cold static allocations (possibly cloned)
; STATS-BE: 4 memprof-context-disambiguation - Number of not cold static allocations (possibly cloned) during ThinLTO backend
; STATS-BE: 8 memprof-context-disambiguation - Number of allocation versions (including clones) during ThinLTO backend
; STATS: 3 memprof-context-disambiguation - Number of function clones created during whole program analysis
; STATS-BE: 3 memprof-context-disambiguation - Number of function clones created during ThinLTO backend
; STATS-BE: 1 memprof-context-disambiguation - Number of functions that had clones created during ThinLTO backend
; STATS-BE: 4 memprof-context-disambiguation - Maximum number of allocation versions created for an original allocation during ThinLTO backend
; STATS-BE: 2 memprof-context-disambiguation - Number of original (not cloned) allocations with memprof profiles during ThinLTO backend
