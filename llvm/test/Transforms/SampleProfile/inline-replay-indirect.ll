; Inline replay must not crash on an indirect call in a replayed function: the
; remarks only name direct callees, so there is no replay decision for it.

; RUN: opt < %s -passes=sample-profile -S \
; RUN:   -sample-profile-file=%S/Inputs/inline-replay-indirect.prof \
; RUN:   -sample-profile-inline-replay=%S/Inputs/inline-replay-indirect.txt \
; RUN:   -pass-remarks=inline 2>&1 | FileCheck %s
; RUN: opt < %s -passes=sample-profile -S \
; RUN:   -sample-profile-file=%S/Inputs/inline-replay-indirect.prof \
; RUN:   -sample-profile-inline-replay=%S/Inputs/inline-replay-indirect.txt \
; RUN:   -sample-profile-inline-replay-scope=Module \
; RUN:   -sample-profile-inline-replay-fallback=NeverInline \
; RUN:   -pass-remarks=inline 2>&1 | FileCheck %s

; CHECK: remark: t.c:5:7: 'callee' inlined into 'caller' {{.*}}at callsite caller:2:7;
; CHECK-LABEL: define i32 @caller(
; CHECK:         call i32 %fp(i32 %x)
; CHECK-NOT:     call i32 @callee(
; CHECK:         ret i32

define i32 @callee(i32 %x) #0 !dbg !10 {
  ret i32 %x, !dbg !11
}

define i32 @caller(ptr %fp, i32 %x) #0 !dbg !12 {
entry:
  %a = call i32 %fp(i32 %x), !dbg !13
  %b = call i32 @callee(i32 %a), !dbg !14
  ret i32 %b, !dbg !15
}

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "t.c", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DISubroutineType(types: !{})
!10 = distinct !DISubprogram(name: "callee", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DILocation(line: 1, column: 1, scope: !10)
!12 = distinct !DISubprogram(name: "caller", scope: !1, file: !1, line: 3, type: !5, scopeLine: 3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!13 = !DILocation(line: 4, column: 7, scope: !12)
!14 = !DILocation(line: 5, column: 7, scope: !12)
!15 = !DILocation(line: 6, column: 3, scope: !12)
