; Verify that typified CS profiles use an ordered offset table, can be loaded by
; the sample-profile pass, and provide payload data used for inlining.
; RUN: llvm-profdata merge --sample --extbinary-force-typified-prof --extbinary %S/Inputs/typified-cs-profile.proftext -o %t.prof
; RUN: llvm-profdata show --sample --show-sec-info-only %t.prof | FileCheck %s --check-prefix=SECTION
; RUN: opt -S %s -passes=sample-profile -sample-profile-file=%t.prof | FileCheck %s --check-prefix=IR
; RUN: llvm-profdata merge --sample --extbinary-force-typified-prof --extbinary \
; RUN:   %S/Inputs/indirect-call-csspgo.prof -o %t.payload.prof
; RUN: opt -S %S/csspgo-inline-icall.ll -passes=sample-profile \
; RUN:   -sample-profile-file=%t.payload.prof \
; RUN:   -sample-profile-icp-relative-hotness=1 -pass-remarks=sample-profile \
; RUN:   -sample-profile-inline-size=0 -o /dev/null 2>&1 | \
; RUN:   FileCheck %s --check-prefix=PAYLOAD

; SECTION: TypifiedFuncOffsetTableSection {{.*}} Flags: {ordered}
; IR-LABEL: define void @test()
; IR-SAME: !prof ![[ENTRY_COUNT:[0-9]+]]
; IR: ![[ENTRY_COUNT]] = !{!"function_entry_count", i64 10}
; PAYLOAD: remark: test.cc:4:0: '_Z3foov' inlined into 'test'

define void @test() #0 !dbg !4 {
entry:
  ret void, !dbg !7
}

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "llvm", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "typified-cs-profile.c", directory: "/")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 1, column: 1, scope: !4)
