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
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -stats -debug -save-temps \
; RUN:  -o %t.out 2>&1 | FileCheck %s --check-prefix=STATS

; RUN: llvm-dis %t.out.1.4.opt.bc -o - | FileCheck %s --check-prefix=IR

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
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -stats -debug \
; RUN:  -o %t2.out 2>&1 | FileCheck %s --check-prefix=STATS

;; Run ThinLTO backend
; RUN: opt -passes=memprof-context-disambiguation \
; RUN:  -memprof-import-summary=%t.o.thinlto.bc \
; RUN:  -stats %t.o -S 2>&1 | FileCheck %s --check-prefix=IR

; DEBUG: Not found through unique tail call chain: _Z3barv from main that actually called _Z3foob (found multiple possible chains)
; DEBUG: Not found through unique tail call chain: _Z3barv from main that actually called _Z3foob (found multiple possible chains)
; DEBUG: Not found through unique tail call chain: _Z3barv from main that actually called _Z3foob (found multiple possible chains)
; DEBUG: Not found through unique tail call chain: _Z3barv from main that actually called _Z3foob (found multiple possible chains)

; STATS: 4 memprof-context-disambiguation - Number of profiled callees found via multiple tail call chains

;; Check that all calls in the IR are to the original functions, leading to a
;; non-cold operator new call.

source_filename = "tailcall-nonunique.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline
; IR-LABEL: @_Z3barv()
define dso_local ptr @_Z3barv() local_unnamed_addr #0 {
entry:
  ; IR: call {{.*}} @_Znam(i64 10) #[[NOTCOLD:[0-9]+]]
  %call = tail call ptr @_Znam(i64 10) #2, !memprof !0, !callsite !9
  ret ptr %call
}

; Function Attrs: nobuiltin allocsize(0)
declare ptr @_Znam(i64) #1

; Function Attrs: noinline
; IR-LABEL: @_Z5blah1v()
define dso_local ptr @_Z5blah1v() local_unnamed_addr #0 {
entry:
  ; IR: call ptr @_Z3barv()
  %call = tail call ptr @_Z3barv()
  ret ptr %call
}

; Function Attrs: noinline
; IR-LABEL: @_Z5blah2v()
define dso_local ptr @_Z5blah2v() local_unnamed_addr #0 {
entry:
  ; IR: call ptr @_Z3barv()
  %call = tail call ptr @_Z3barv()
  ret ptr %call
}

; Function Attrs: noinline
; IR-LABEL: @_Z4baz1v()
define dso_local ptr @_Z4baz1v() local_unnamed_addr #0 {
entry:
  ; IR: call ptr @_Z5blah1v()
  %call = tail call ptr @_Z5blah1v()
  ret ptr %call
}

; Function Attrs: noinline
; IR-LABEL: @_Z4baz2v()
define dso_local ptr @_Z4baz2v() local_unnamed_addr #0 {
entry:
  ; IR: call ptr @_Z5blah2v()
  %call = tail call ptr @_Z5blah2v()
  ret ptr %call
}

; Function Attrs: noinline
; IR-LABEL: @_Z3foob(i1 %b)
define dso_local ptr @_Z3foob(i1 %b) local_unnamed_addr #0 {
entry:
  br i1 %b, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  ; IR: call ptr @_Z4baz1v()
  %call = tail call ptr @_Z4baz1v()
  br label %return

if.else:                                          ; preds = %entry
  ; IR: call ptr @_Z4baz2v()
  %call1 = tail call ptr @_Z4baz2v()
  br label %return

return:                                           ; preds = %if.else, %if.then
  %retval.0 = phi ptr [ %call, %if.then ], [ %call1, %if.else ]
  ret ptr %retval.0
}

; Function Attrs: noinline
; IR-LABEL: @main()
define dso_local i32 @main() local_unnamed_addr #0 {
delete.end13:
  ; IR: call ptr @_Z3foob(i1 true)
  %call = tail call ptr @_Z3foob(i1 true), !callsite !10
  ; IR: call ptr @_Z3foob(i1 true)
  %call1 = tail call ptr @_Z3foob(i1 true), !callsite !11
  ; IR: call ptr @_Z3foob(i1 false)
  %call2 = tail call ptr @_Z3foob(i1 false), !callsite !12
  ; IR: call ptr @_Z3foob(i1 false)
  %call3 = tail call ptr @_Z3foob(i1 false), !callsite !13
  ret i32 0
}

; IR: attributes #[[NOTCOLD]] = { builtin allocsize(0) "memprof"="notcold" }

attributes #0 = { noinline }
attributes #1 = { nobuiltin allocsize(0) }
attributes #2 = { builtin allocsize(0) }

!0 = !{!1, !3, !5, !7}
!1 = !{!2, !"notcold"}
!2 = !{i64 3186456655321080972, i64 6307901912192269588}
!3 = !{!4, !"cold"}
!4 = !{i64 3186456655321080972, i64 6792096022461663180}
!5 = !{!6, !"notcold"}
!6 = !{i64 3186456655321080972, i64 8632435727821051414}
!7 = !{!8, !"cold"}
!8 = !{i64 3186456655321080972, i64 -3421689549917153178}
!9 = !{i64 3186456655321080972}
!10 = !{i64 8632435727821051414}
!11 = !{i64 -3421689549917153178}
!12 = !{i64 6307901912192269588}
!13 = !{i64 6792096022461663180}
