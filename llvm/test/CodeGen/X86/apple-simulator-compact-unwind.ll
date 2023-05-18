; RUN: llc -mtriple x86_64-apple-ios-simulator -filetype=obj -o - %s | \
; RUN: llvm-objdump --macho --unwind-info - | \
; RUN: FileCheck %s

; RUN: llc -mtriple x86_64-apple-ios -filetype=obj -o - %s | \
; RUN: llvm-objdump --macho --unwind-info - | \
; RUN: FileCheck %s

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: noinline nounwind optnone ssp uwtable
define i32 @rdar104359594() #0 {
entry:
  ret i32 0
}

attributes #0 = { noinline nounwind optnone ssp uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtin-calloc" "no-builtin-stpcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="core2" "target-features"="+cmov,+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+ssse3,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"uwtable", i32 2}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{!"clang"}

; Check that we generate compact unwind for simulators, which have always
; supported it.
; CHECK: Contents of __compact_unwind section:
