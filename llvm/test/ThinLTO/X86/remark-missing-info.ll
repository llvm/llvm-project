;; Test that we emit a basic remark during the Thin link when context size info is missing.

; RUN: opt -thinlto-bc %s -o %t.o
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:	-supports-hot-cold-new \
; RUN:	-r=%t.o,main,plx \
; RUN:	-r=%t.o,_Znam, \
; RUN:	-pass-remarks-output=%t.remarks.yaml \
; RUN:	-o %t.out 2>&1
; RUN: FileCheck %s < %t.remarks.yaml

; CHECK: MemProf hinting: NotCold is NotCold after cloning (context id 1)
; CHECK: MemProf hinting: Cold is Cold after cloning (context id 2)

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() #0 {
entry:
  %call = call noundef ptr @_Z3foov(), !callsite !0
  %call1 = call noundef ptr @_Z3foov(), !callsite !1
  ret i32 0
}

define internal ptr @_Z3barv() #3 {
entry:
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #6, !memprof !2, !callsite !7
  ret ptr null
}

declare ptr @_Znam(i64)

define internal ptr @_Z3bazv() #4 {
entry:
  %call = call noundef ptr @_Z3barv(), !callsite !8
  ret ptr null
}

define internal ptr @_Z3foov() #5 {
entry:
  %call = call noundef ptr @_Z3bazv(), !callsite !9
  ret ptr null
}

attributes #0 = { "tune-cpu"="generic" }
attributes #3 = { "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }
attributes #4 = { "stack-protector-buffer-size"="8" }
attributes #5 = { noinline }
attributes #6 = { builtin }

!llvm.module.flags = !{!20, !21}

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
!20 = !{i32 7, !"Dwarf Version", i32 5}
!21 = !{i32 2, !"Debug Info Version", i32 3}
