; Verify that composite CS profiles use an ordered offset table and can be
; loaded by the sample-profile pass.
; RUN: llvm-profdata merge --sample --extbinary-composite-prof --extbinary %S/Inputs/composite-cs-profile.proftext -o %t.prof
; RUN: llvm-profdata show --sample --show-sec-info-only %t.prof | FileCheck %s --check-prefix=SECTION
; RUN: opt -S %s -passes=sample-profile -sample-profile-file=%t.prof | FileCheck %s --check-prefix=IR

; SECTION: CompositeFuncOffsetTableSection {{.*}} Flags: {ordered}
; IR-LABEL: define void @test()
; IR-SAME: !prof ![[ENTRY_COUNT:[0-9]+]]
; IR: ![[ENTRY_COUNT]] = !{!"function_entry_count", i64 10}

define void @test() #0 !dbg !4 {
entry:
  ret void, !dbg !7
}

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "llvm", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "composite-cs-profile.c", directory: "/")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 1, column: 1, scope: !4)
