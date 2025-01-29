; RUN: llc -mtriple=arm-none-eabi < %s | FileCheck %s
; RUN: llc -mtriple=arm-none-eabi < %s | llvm-mc --triple=arm-none-eabi -mcpu=cortex-m3

; Check that, when an aligned loop is the first thing in a function, we do not
; emit an invalid .loc directive, which is rejected by the assembly parser.

; CHECK-NOT: .loc    0
; CHECK: .loc    1 2 3 prologue_end
; CHECK-NOT: .loc 0

define dso_local void @foo() "target-cpu"="cortex-m3" !dbg !8 {
entry:
  br label %while.body, !dbg !11

while.body:
  br label %while.body, !dbg !11
}


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 20.0.0git (git@github.com:llvm/llvm-project.git 1c984b86b389bbc71c8c2988d1d707e2f32878bd)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/work/scratch")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocation(line: 2, column: 3, scope: !8)
