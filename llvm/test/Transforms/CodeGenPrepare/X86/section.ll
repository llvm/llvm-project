; RUN: opt < %s -codegenprepare -S | FileCheck %s
; RUN: llc < %s | FileCheck --check-prefix=ASM1 %s
; RUN: llc < %s -function-sections | FileCheck --check-prefix=ASM2 %s

target triple = "x86_64-pc-linux-gnu"

; This tests that hot/cold functions get correct section prefix assigned

; CHECK: hot_func1{{.*}}!section_prefix ![[HOT_ID:[0-9]+]]
; ASM1: .section .text.hot.,"ax",@progbits
; ASM2: .section .text.hot.hot_func1,"ax",@progbits
; The entry is hot
define void @hot_func1() !prof !15 {
  ret void
}

; CHECK: hot_func2{{.*}}!section_prefix ![[HOT_ID:[0-9]+]]
; Entry is cold but inner block is hot
define void @hot_func2(i32 %n) !prof !16 {
entry:
  %n.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %n, i32* %n.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:
  %0 = load i32, i32* %i, align 4
  %1 = load i32, i32* %n.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end, !prof !19

for.body:
  %2 = load i32, i32* %i, align 4
  %inc = add nsw i32 %2, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:
  ret void
}

; For instrumentation based PGO, we should only look at block counts,
; not call site VP metadata (which can exist on value profiled memcpy,
; or possibly left behind after static analysis based devirtualization).
; CHECK: cold_func1{{.*}}!section_prefix ![[COLD_ID:[0-9]+]]
; ASM1: .section .text.unlikely.,"ax",@progbits
; ASM2: .section .text.unlikely.cold_func1,"ax",@progbits
define void @cold_func1() !prof !16 {
  call void @hot_func1(), !prof !17
  call void @hot_func1(), !prof !17
  ret void
}

; CHECK: cold_func2{{.*}}!section_prefix ![[COLD_ID]]
define void @cold_func2() !prof !16 {
  call void @hot_func1(), !prof !17
  call void @hot_func1(), !prof !18
  call void @hot_func1(), !prof !18
  ret void
}

; CHECK: cold_func3{{.*}}!section_prefix ![[COLD_ID]]
define void @cold_func3() !prof !16 {
  call void @hot_func1(), !prof !18
  ret void
}

; CHECK: ![[HOT_ID]] = !{!"function_section_prefix", !"hot"}
; CHECK: ![[COLD_ID]] = !{!"function_section_prefix", !"unlikely"}
!llvm.module.flags = !{!1}
!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 10000}
!5 = !{!"MaxCount", i64 1000}
!6 = !{!"MaxInternalCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 1000}
!8 = !{!"NumCounts", i64 3}
!9 = !{!"NumFunctions", i64 3}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14}
!12 = !{i32 10000, i64 100, i32 1}
!13 = !{i32 999000, i64 100, i32 1}
!14 = !{i32 999999, i64 1, i32 2}
!15 = !{!"function_entry_count", i64 1000}
!16 = !{!"function_entry_count", i64 1}
!17 = !{!"branch_weights", i32 80}
!18 = !{!"branch_weights", i32 1}
!19 = !{!"branch_weights", i32 1000, i32 1}
