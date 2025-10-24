;; Test that passing -supports-hot-cold-new to the LTO link allows context
;; disambiguation to proceed, and also prevents memprof metadata and attributes
;; from being removed from the LTO backend, and vice versa without passing
;; -supports-hot-cold-new.

; RUN: split-file %s %t

;; First check with -supports-hot-cold-new.
; RUN: opt -thinlto-bc %t/main.ll >%t/main.o
; RUN: opt -thinlto-bc %t/foo.ll >%t/foo.o
; RUN: llvm-lto2 run %t/main.o %t/foo.o -enable-memprof-context-disambiguation \
; RUN:	-supports-hot-cold-new \
; RUN:	-r=%t/main.o,main,plx \
; RUN:	-r=%t/main.o,bar,plx \
; RUN:	-r=%t/main.o,foo, \
; RUN:	-r=%t/main.o,_Znam, \
; RUN:	-r=%t/foo.o,foo,plx \
; RUN:	-r=%t/foo.o,_Znam, \
; RUN:	-memprof-dump-ccg \
; RUN:	-print-before=memprof-context-disambiguation \
; RUN:	-thinlto-threads=1 \
; RUN:	-o %t.out 2>&1 | FileCheck %s --check-prefix=DUMP --check-prefix=IR

; DUMP: Callsite Context Graph:

; IR: @main()
; IR: !memprof {{.*}} !callsite
; IR: @_Znam(i64 0) #[[ATTR:[0-9]+]]
; IR: @bar()
; IR: !memprof {{.*}} !callsite
; IR: @_Znam(i64 0) #[[ATTR:[0-9]+]]
; IR: attributes #[[ATTR]] = { "memprof"="cold" }

;; Next check without -supports-hot-cold-new, we should not perform
;; context disambiguation, and we should strip memprof metadata and
;; attributes before optimization.
; RUN: llvm-lto2 run %t/main.o %t/foo.o -enable-memprof-context-disambiguation \
; RUN:	-r=%t/main.o,main,plx \
; RUN:	-r=%t/main.o,bar,plx \
; RUN:	-r=%t/main.o,foo, \
; RUN:	-r=%t/main.o,_Znam, \
; RUN:	-r=%t/foo.o,foo,plx \
; RUN:	-r=%t/foo.o,_Znam, \
; RUN:	-memprof-dump-ccg \
; RUN:	-print-before=memprof-context-disambiguation \
; RUN:	-thinlto-threads=1 \
; RUN:	-o %t.out 2>&1 | FileCheck %s --allow-empty \
; RUN:  --implicit-check-not "Callsite Context Graph:" \
; RUN: 	--implicit-check-not "!memprof" --implicit-check-not "!callsite" \
; RUN: 	--implicit-check-not "memprof"="cold"

;--- main.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() #0 {
entry:
  %call = call ptr @_Znam(i64 0), !memprof !0, !callsite !5
  %call2 = call ptr @_Znam(i64 0) #1
  ret i32 0
}

define void @bar() {
  call void @foo()
  ret void
}

declare void @foo()

declare ptr @_Znam(i64)

attributes #0 = { noinline optnone }
attributes #1 = { "memprof"="cold" }

!0 = !{!1, !3}
!1 = !{!2, !"notcold"}
!2 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 8632435727821051414}
!3 = !{!4, !"cold"}
!4 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 -3421689549917153178}
!5 = !{i64 9086428284934609951}

;--- foo.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo() {
entry:
  %call = call ptr @_Znam(i64 0), !memprof !0, !callsite !5
  %call2 = call ptr @_Znam(i64 0) #1
  ret i32 0
}

declare ptr @_Znam(i64)

attributes #1 = { "memprof"="cold" }

!0 = !{!1, !3}
!1 = !{!2, !"notcold"}
!2 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 8632435727821051414}
!3 = !{!4, !"cold"}
!4 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 -3421689549917153178}
!5 = !{i64 9086428284934609951}
