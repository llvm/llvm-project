;; Test to make sure that missing tail call frames in memprof profiles are
;; identified but not cloned when there are multiple non-unique possible
;; tail call chains between the profiled frames.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -thinlto-bc %s >%t.o
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t.o,_Z3barv,plx \
; RUN:  -r=%t.o,_Z5blah1v,plx \
; RUN:  -r=%t.o,_Z5blah2v,plx \
; RUN:  -r=%t.o,_Z4baz1v,plx \
; RUN:  -r=%t.o,_Z4baz2v,plx \
; RUN:  -r=%t.o,_Z3foob,plx \
; RUN:  -r=%t.o,xyz,plx \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -stats -debug -save-temps \
; RUN:  -o %t.out 2>&1 | FileCheck %s --check-prefix=STATS --check-prefix=DEBUG

;; We should not see any type of cold attribute or cloning applied
; RUN: llvm-dis %t.out.1.4.opt.bc -o - | FileCheck %s --implicit-check-not cold \
; RUN:	--implicit-check-not ".memprof."

;; Try again but with distributed ThinLTO
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -r=%t.o,_Z3barv,plx \
; RUN:  -r=%t.o,_Z5blah1v,plx \
; RUN:  -r=%t.o,_Z5blah2v,plx \
; RUN:  -r=%t.o,_Z4baz1v,plx \
; RUN:  -r=%t.o,_Z4baz2v,plx \
; RUN:  -r=%t.o,_Z3foob,plx \
; RUN:  -r=%t.o,xyz,plx \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -stats -debug \
; RUN:  -o %t2.out 2>&1 | FileCheck %s --check-prefix=STATS --check-prefix=DEBUG

;; Run ThinLTO backend
;; We should not see any type of cold attribute or cloning applied
; RUN: opt -passes=memprof-context-disambiguation \
; RUN:  -memprof-import-summary=%t.o.thinlto.bc \
; RUN:  -stats %t.o -S 2>&1 | FileCheck %s --implicit-check-not cold \
; RUN:	--implicit-check-not ".memprof."

; DEBUG: Not found through unique tail call chain: 17377440600225628772 (_Z3barv) from 15822663052811949562 (main) that actually called 8716735811002003409 (xyz) (found multiple possible chains)

; STATS: 1 memprof-context-disambiguation - Number of profiled callees found via multiple tail call chains

source_filename = "tailcall-nonunique.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline
define dso_local ptr @_Z3barv() local_unnamed_addr #0 {
entry:
  %call = tail call ptr @_Znam(i64 10) #2, !memprof !0, !callsite !9
  ret ptr %call
}

; Function Attrs: nobuiltin allocsize(0)
declare ptr @_Znam(i64) #1

; Function Attrs: noinline
define dso_local ptr @_Z5blah1v() local_unnamed_addr #0 {
entry:
  %call = tail call ptr @_Z3barv()
  ret ptr %call
}

; Function Attrs: noinline
define dso_local ptr @_Z5blah2v() local_unnamed_addr #0 {
entry:
  %call = tail call ptr @_Z3barv()
  ret ptr %call
}

; Function Attrs: noinline
define dso_local ptr @_Z4baz1v() local_unnamed_addr #0 {
entry:
  %call = tail call ptr @_Z5blah1v()
  ret ptr %call
}

; Function Attrs: noinline
define dso_local ptr @_Z4baz2v() local_unnamed_addr #0 {
entry:
  %call = tail call ptr @_Z5blah2v()
  ret ptr %call
}

; Function Attrs: noinline
define dso_local ptr @_Z3foob(i1 %b) local_unnamed_addr #0 {
entry:
  br i1 %b, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %call = tail call ptr @_Z4baz1v()
  br label %return

if.else:                                          ; preds = %entry
  %call1 = tail call ptr @_Z4baz2v()
  br label %return

return:                                           ; preds = %if.else, %if.then
  %retval.0 = phi ptr [ %call, %if.then ], [ %call1, %if.else ]
  ret ptr %retval.0
}

; Function Attrs: noinline
define dso_local i32 @xyz() local_unnamed_addr #0 {
delete.end13:
  %call = tail call ptr @_Z3foob(i1 true)
  %call1 = tail call ptr @_Z3foob(i1 true)
  %call2 = tail call ptr @_Z3foob(i1 false)
  %call3 = tail call ptr @_Z3foob(i1 false)
  ret i32 0
}

define dso_local i32 @main() local_unnamed_addr #0 {
delete.end13:
  %call1 = tail call i32 @xyz(), !callsite !11
  ret i32 0
}

attributes #0 = { noinline }
attributes #1 = { nobuiltin allocsize(0) }
attributes #2 = { builtin allocsize(0) }

!0 = !{!5, !7}
!5 = !{!6, !"notcold"}
!6 = !{i64 3186456655321080972, i64 8632435727821051414}
!7 = !{!8, !"cold"}
!8 = !{i64 3186456655321080972, i64 -3421689549917153178}
!9 = !{i64 3186456655321080972}
!11 = !{i64 -3421689549917153178}
