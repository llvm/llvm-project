; Thanks to Shannon Kuntz for the original source for this test case.
;
; RUN: opt < %s -tapir2target -tapir-target=cilk -S
; RUN: opt < %s -passes='tapir2target' -tapir-target=cilk -S

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noreturn nounwind uwtable
define i32 @main() local_unnamed_addr #0 {
entry:
  tail call void @exit(i32 13) #2
  unreachable
}

; Function Attrs: noreturn nounwind
declare void @exit(i32) local_unnamed_addr #1

attributes #0 = { noreturn nounwind uwtable }
attributes #1 = { noreturn nounwind }
attributes #2 = { noreturn nounwind }
