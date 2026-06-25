;; Test that MemProf ICP correctly updates MD_prof metadata for the fallback
;; call when some candidates are skipped during promotion.

; REQUIRES: asserts

; RUN: split-file %s %t
; RUN: opt -thinlto-bc %t/main.ll -o %t/main.o
; RUN: opt -thinlto-bc %t/foo.ll -o %t/foo.o

;; Perform ThinLTO. We provide the definition for _ZN2B03barEj but not
;;_ZN1B3barEj. With -memprof-require-definition-for-promotion, _ZN1B3barEj
;; should be skipped and _ZN2B03barEj should be promoted.
; RUN: llvm-lto2 run %t/main.o %t/foo.o -enable-memprof-context-disambiguation \
; RUN:	-enable-memprof-indirect-call-support=true \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t/foo.o,_Z3fooR2B0j,plx \
; RUN:  -r=%t/foo.o,_ZN2B03barEj, \
; RUN:  -r=%t/main.o,main,plx \
; RUN:  -r=%t/main.o,_Z3fooR2B0j, \
; RUN:  -r=%t/main.o,_ZN2B03barEj,plx \
; RUN:  -r=%t/main.o,_Znwm, \
; RUN:	-thinlto-threads=1 \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -save-temps \
; RUN:  -memprof-require-definition-for-promotion \
; RUN:  -o %t.out 2>&1 | FileCheck %s --check-prefix=REMARKS

; REMARKS: promoted and assigned to call function clone _ZN2B03barEj

; RUN: llvm-dis %t.out.2.4.opt.bc -o - | FileCheck %s --check-prefix=IR

;; Check that the fallback call has the correct VP metadata for the skipped
;; candidate (_ZN1B3barEj with MD5 4445083295448962937 and count 2).
; IR: define {{.*}} @_Z3fooR2B0j
; IR:   %[[R1:[0-9]+]] = icmp eq ptr %0, @_ZN2B03barEj
; IR:   br i1 %[[R1]], label %[[LABEL:.*]], label %if.false.orig_indirect
; IR: if.false.orig_indirect:
; IR:   tail call i32 %0(ptr null, i32 0), !prof ![[PROF:[0-9]+]]
; IR: ![[PROF]] = !{!"VP", i32 0, i64 2, i64 4445083295448962937, i64 2}

;--- foo.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @_ZN2B03barEj(ptr, i32)

define i32 @_Z3fooR2B0j(ptr %b) {
entry:
  %0 = load ptr, ptr %b, align 8
  %call = tail call i32 %0(ptr null, i32 0), !prof !1, !callsite !2
  ret i32 %call
}

;; VP metadata with two candidates:
;; 1. MD5 4445083295448962937 (_ZN1B3barEj), count 2
;; 2. MD5 -2718743882639408571 (_ZN2B03barEj), count 2
!1 = !{!"VP", i32 0, i64 4, i64 4445083295448962937, i64 2, i64 -2718743882639408571, i64 2}
!2 = !{i64 -2101080423462424381}

;--- main.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
entry:
  %call2 = call i32 @_Z3fooR2B0j(ptr null), !callsite !30
  %call4 = call i32 @_Z3fooR2B0j(ptr null), !callsite !31
  ret i32 0
}

declare i32 @_Z3fooR2B0j(ptr)

define i32 @_ZN2B03barEj(ptr %this, i32 %s) {
entry:
  %call = tail call ptr @_Znwm(i64 noundef 4) #0, !memprof !33, !callsite !38
  ret i32 %s
}

declare ptr @_Znwm(i64)

attributes #0 = { builtin allocsize(0) }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 4}

!30 = !{i64 -6490791336773930154}
!31 = !{i64 5188446645037944434}
!33 = !{!34, !36}
!34 = !{!35, !"notcold"}
!35 = !{i64 -852997907418798798, i64 -2101080423462424381, i64 -6490791336773930154}
!36 = !{!37, !"cold"}
!37 = !{i64 -852997907418798798, i64 -2101080423462424381, i64 5188446645037944434}
!38 = !{i64 -852997907418798798}
