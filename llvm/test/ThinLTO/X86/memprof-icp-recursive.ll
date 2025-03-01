;; This test case is the same as memprof-icp.ll but manually modified to
;; add recursive cycles into the allocation contexts, and it has
;; -memprof-allow-recursive-callsites enabled to test that handling.
;; The call contexts are somewhat nonsensical, but designed to provoke
;; several scenarios that were seen in large applications and caused
;; failures until they were addressed. In particular, converting mutual
;; recursion to direct recursion when stack nodes are matched onto inlined
;; callsites; edge partitioning due to multiple calls with the same debug
;; info (here because it is an indirect call) requiring updating/moving
;; of recursive contexts; and the necessary updates of recursive contexts
;; and edges during cloning.

;; The checks are however not modified from the original memprof-icp.ll test:
;; when basic recursive callsite handling is enabled (which currently doesn't
;; clone through recursive cycles but is designed to at least allow cloning
;; across them for the the non-recursive call sequence of the context) we
;; should successfully get the same cloning as without any recursion.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: split-file %s %t

;; For now explicitly turn on this handling, which is off by default.
; RUN: opt -thinlto-bc %t/main.ll -enable-memprof-indirect-call-support=true >%t/main.o
; RUN: opt -thinlto-bc %t/foo.ll -enable-memprof-indirect-call-support=true >%t/foo.o

;; First perform in-process ThinLTO
; RUN: llvm-lto2 run %t/main.o %t/foo.o -enable-memprof-context-disambiguation \
; RUN:	-enable-memprof-indirect-call-support=true \
; RUN:  -memprof-allow-recursive-callsites \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t/foo.o,_Z3fooR2B0j,plx \
; RUN:  -r=%t/foo.o,_ZN2B03barEj, \
; RUN:  -r=%t/foo.o,_ZN1B3barEj, \
; RUN:  -r=%t/main.o,_Z3fooR2B0j, \
; RUN:  -r=%t/main.o,_Znwm, \
; RUN:  -r=%t/main.o,_ZdlPvm, \
; RUN:  -r=%t/main.o,_Z8externalPi, \
; RUN:  -r=%t/main.o,main,plx \
; RUN:  -r=%t/main.o,_ZN2B03barEj,plx \
; RUN:  -r=%t/main.o,_ZN1B3barEj,plx \
; RUN:  -r=%t/main.o,_ZTV1B,plx \
; RUN:  -r=%t/main.o,_ZTVN10__cxxabiv120__si_class_type_infoE,plx \
; RUN:  -r=%t/main.o,_ZTS1B,plx \
; RUN:  -r=%t/main.o,_ZTVN10__cxxabiv117__class_type_infoE,plx \
; RUN:  -r=%t/main.o,_ZTS2B0,plx \
; RUN:  -r=%t/main.o,_ZTI2B0,plx \
; RUN:  -r=%t/main.o,_ZTI1B,plx \
; RUN:  -r=%t/main.o,_ZTV2B0,plx \
; RUN:	-thinlto-threads=1 \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes -stats \
; RUN:  -pass-remarks=. -save-temps \
; RUN:  -o %t.out 2>&1 | FileCheck %s --check-prefix=STATS \
; RUN:  --check-prefix=REMARKS

; RUN: llvm-dis %t.out.2.4.opt.bc -o - | FileCheck %s --check-prefix=IR

; REMARKS: call in clone main assigned to call function clone _Z3fooR2B0j.memprof.1
; REMARKS: call in clone main assigned to call function clone _Z3fooR2B0j.memprof.1
; REMARKS: created clone _ZN2B03barEj.memprof.1
; REMARKS: call in clone _ZN2B03barEj marked with memprof allocation attribute notcold
; REMARKS: call in clone _ZN2B03barEj.memprof.1 marked with memprof allocation attribute cold
; REMARKS: call in clone _ZN2B03barEj marked with memprof allocation attribute notcold
; REMARKS: call in clone _ZN2B03barEj.memprof.1 marked with memprof allocation attribute cold
; REMARKS: created clone _ZN1B3barEj.memprof.1
; REMARKS: call in clone _ZN1B3barEj marked with memprof allocation attribute notcold
; REMARKS: call in clone _ZN1B3barEj.memprof.1 marked with memprof allocation attribute cold
; REMARKS: call in clone _ZN1B3barEj marked with memprof allocation attribute notcold
; REMARKS: call in clone _ZN1B3barEj.memprof.1 marked with memprof allocation attribute cold
; REMARKS: created clone _Z3fooR2B0j.memprof.1
;; In each version of foo we should have promoted the indirect call to two conditional
;; direct calls, one to B::bar and one to B0::bar. The cloned version of foo should call
;; the cloned versions of bar for both promotions.
; REMARKS: Promote indirect call to _ZN1B3barEj with count 2 out of 4
; REMARKS: call in clone _Z3fooR2B0j promoted and assigned to call function clone _ZN1B3barEj
; REMARKS: Promote indirect call to _ZN1B3barEj with count 2 out of 4
; REMARKS: call in clone _Z3fooR2B0j.memprof.1 promoted and assigned to call function clone _ZN1B3barEj.memprof.1
; REMARKS: Promote indirect call to _ZN2B03barEj with count 2 out of 2
; REMARKS: call in clone _Z3fooR2B0j promoted and assigned to call function clone _ZN2B03barEj
; REMARKS: Promote indirect call to _ZN2B03barEj with count 2 out of 2
; REMARKS: call in clone _Z3fooR2B0j.memprof.1 promoted and assigned to call function clone _ZN2B03barEj.memprof.1
; REMARKS: created clone _ZN2B03barEj.memprof.1
; REMARKS: call in clone _ZN2B03barEj marked with memprof allocation attribute notcold
; REMARKS: call in clone _ZN2B03barEj.memprof.1 marked with memprof allocation attribute cold
; REMARKS: created clone _ZN1B3barEj.memprof.1
; REMARKS: call in clone _ZN1B3barEj marked with memprof allocation attribute notcold
; REMARKS: call in clone _ZN1B3barEj.memprof.1 marked with memprof allocation attribute cold

; STATS: 4 memprof-context-disambiguation - Number of cold static allocations (possibly cloned) during whole program analysis
; STATS: 8 memprof-context-disambiguation - Number of cold static allocations (possibly cloned) during ThinLTO backend
; STATS: 4 memprof-context-disambiguation - Number of not cold static allocations (possibly cloned) during whole program analysis
; STATS: 8 memprof-context-disambiguation - Number of not cold static allocations (possibly cloned) during ThinLTO backend
; STATS: 3 memprof-context-disambiguation - Number of function clones created during whole program analysis
; STATS: 5 memprof-context-disambiguation - Number of function clones created during ThinLTO backend

; IR: define {{.*}} @_Z3fooR2B0j(
; IR:   %[[R1:[0-9]+]] = icmp eq ptr %0, @_ZN1B3barEj
; IR:   br i1 %[[R1]], label %if.true.direct_targ, label %if.false.orig_indirect
; IR: if.true.direct_targ:
; IR:   call {{.*}} @_Znwm(i64 noundef 4) #[[NOTCOLD:[0-9]+]]
; IR: if.false.orig_indirect:
; IR:   %[[R2:[0-9]+]] = icmp eq ptr %0, @_ZN2B03barEj
; IR:   br i1 %[[R2]], label %if.true.direct_targ1, label %if.false.orig_indirect2
; IR: if.true.direct_targ1:
; IR:   call {{.*}} @_Znwm(i64 noundef 4) #[[NOTCOLD]]
; IR: if.false.orig_indirect2:
; IR:   call {{.*}} %0

; IR: define {{.*}} @_Z3fooR2B0j.memprof.1(
;; We should still compare against the original versions of bar since that is
;; what is in the vtable. However, we should have called the cloned versions
;; that perform cold allocations, which were subsequently inlined.
; IR:   %[[R3:[0-9]+]] = icmp eq ptr %0, @_ZN1B3barEj
; IR:   br i1 %[[R3]], label %if.true.direct_targ, label %if.false.orig_indirect
; IR: if.true.direct_targ:
; IR:   call {{.*}} @_Znwm(i64 noundef 4) #[[COLD:[0-9]+]]
; IR: if.false.orig_indirect:
; IR:   %[[R4:[0-9]+]] = icmp eq ptr %0, @_ZN2B03barEj
; IR:   br i1 %[[R4]], label %if.true.direct_targ1, label %if.false.orig_indirect2
; IR: if.true.direct_targ1:
; IR:   call {{.*}} @_Znwm(i64 noundef 4) #[[COLD]]
; IR: if.false.orig_indirect2:
; IR:   call {{.*}} %0

; IR: attributes #[[NOTCOLD]] = {{.*}} "memprof"="notcold"
; IR: attributes #[[COLD]] = {{.*}} "memprof"="cold"

;--- foo.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @_ZN2B03barEj(ptr %this, i32 %s)
declare i32 @_ZN1B3barEj(ptr %this, i32 %s)

define i32 @_Z3fooR2B0j(ptr %b) {
entry:
  %0 = load ptr, ptr %b, align 8
  %call = tail call i32 %0(ptr null, i32 0), !prof !0, !callsite !1
  ret i32 0
}

!0 = !{!"VP", i32 0, i64 4, i64 4445083295448962937, i64 2, i64 -2718743882639408571, i64 2}
!1 = !{i64 -2101080423462424381, i64 3456}

;--- main.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZTV1B = external constant { [3 x ptr] }
@_ZTVN10__cxxabiv120__si_class_type_infoE = external global [0 x ptr]
@_ZTS1B = external constant [3 x i8]
@_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
@_ZTS2B0 = external constant [4 x i8]
@_ZTI2B0 = external constant { ptr, ptr }
@_ZTI1B = external constant { ptr, ptr, ptr }
@_ZTV2B0 = external constant { [3 x ptr] }

define i32 @main() !prof !29 {
entry:
  %call2 = call i32 @_Z3fooR2B0j(ptr null, i32 0), !callsite !30
  %call4 = call i32 @_Z3fooR2B0j(ptr null, i32 0), !callsite !31
  %call6 = call i32 @_Z3fooR2B0j(ptr null, i32 0), !callsite !32
  %call8 = call i32 @_Z3fooR2B0j(ptr null, i32 0), !callsite !57
  ret i32 0
}

declare i32 @_Z3fooR2B0j(ptr, i32)

define i32 @_ZN2B03barEj(ptr %this, i32 %s) {
entry:
  %call = tail call ptr @_Znwm(i64 noundef 4) #0, !memprof !33, !callsite !38
  ;; Second allocation in this function, to ensure that indirect edges to the
  ;; same callee are partitioned correctly.
  %call2 = tail call ptr @_Znwm(i64 noundef 4) #0, !memprof !45, !callsite !50
  store volatile i32 0, ptr %call, align 4
  ret i32 0
}

declare ptr @_Znwm(i64)

declare void @_Z8externalPi()

declare void @_ZdlPvm()

define i32 @_ZN1B3barEj(ptr %this, i32 %s) {
entry:
  %call = tail call ptr @_Znwm(i64 noundef 4) #0, !memprof !39, !callsite !44
  ;; Second allocation in this function, to ensure that indirect edges to the
  ;; same callee are partitioned correctly.
  %call2 = tail call ptr @_Znwm(i64 noundef 4) #0, !memprof !51, !callsite !56
  store volatile i32 0, ptr %call, align 4
  ret i32 0
}

; uselistorder directives
uselistorder ptr @_Z3fooR2B0j, { 3, 2, 1, 0 }

attributes #0 = { builtin allocsize(0) }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9, !10, !11}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 13}
!4 = !{!"MaxCount", i64 4}
!5 = !{!"MaxInternalCount", i64 0}
!6 = !{!"MaxFunctionCount", i64 4}
!7 = !{!"NumCounts", i64 5}
!8 = !{!"NumFunctions", i64 5}
!9 = !{!"IsPartialProfile", i64 0}
!10 = !{!"PartialProfileRatio", double 0.000000e+00}
!11 = !{!"DetailedSummary", !12}
!12 = !{!13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28}
!13 = !{i32 10000, i64 0, i32 0}
!14 = !{i32 100000, i64 4, i32 2}
!15 = !{i32 200000, i64 4, i32 2}
!16 = !{i32 300000, i64 4, i32 2}
!17 = !{i32 400000, i64 4, i32 2}
!18 = !{i32 500000, i64 4, i32 2}
!19 = !{i32 600000, i64 4, i32 2}
!20 = !{i32 700000, i64 2, i32 4}
!21 = !{i32 800000, i64 2, i32 4}
!22 = !{i32 900000, i64 2, i32 4}
!23 = !{i32 950000, i64 2, i32 4}
!24 = !{i32 990000, i64 2, i32 4}
!25 = !{i32 999000, i64 2, i32 4}
!26 = !{i32 999900, i64 2, i32 4}
!27 = !{i32 999990, i64 2, i32 4}
!28 = !{i32 999999, i64 2, i32 4}
!29 = !{!"function_entry_count", i64 1}
!30 = !{i64 -6490791336773930154}
!31 = !{i64 5188446645037944434}
!32 = !{i64 5583420417449503557}
!57 = !{i64 132626519179914298}
!33 = !{!34, !36}
!34 = !{!35, !"notcold"}
!35 = !{i64 -852997907418798798, i64 -2101080423462424381, i64 3456, i64 -2101080423462424381, i64 3456, i64 5188446645037944434}
!36 = !{!37, !"cold"}
!37 = !{i64 -852997907418798798, i64 -2101080423462424381, i64 3456, i64 -2101080423462424381, i64 3456, i64 5583420417449503557}
!38 = !{i64 -852997907418798798}
!39 = !{!40, !42}
!40 = !{!41, !"notcold"}
!41 = !{i64 4457553070050523782, i64 -2101080423462424381, i64 3456, i64 -2101080423462424381, i64 3456, i64 132626519179914298}
!42 = !{!43, !"cold"}
!43 = !{i64 4457553070050523782, i64 -2101080423462424381, i64 3456, i64 -2101080423462424381, i64 3456, i64 -6490791336773930154}
!44 = !{i64 4457553070050523782}
!45 = !{!46, !48}
!46 = !{!47, !"notcold"}
!47 = !{i64 456, i64 -2101080423462424381, i64 3456, i64 5188446645037944434}
!48 = !{!49, !"cold"}
!49 = !{i64 456, i64 -2101080423462424381, i64 3456, i64 5583420417449503557}
!50 = !{i64 456}
!51 = !{!52, !54}
!52 = !{!53, !"notcold"}
!53 = !{i64 789, i64 -2101080423462424381, i64 3456, i64 132626519179914298}
!54 = !{!55, !"cold"}
!55 = !{i64 789, i64 -2101080423462424381, i64 3456, i64 -6490791336773930154}
!56 = !{i64 789}
