; RUN: opt -inline -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function @bar contains instruction %cmp which is not associated to any debug
; location. This test verifies that the inliner doesn't incorrectly attribute
; the callsite debug location to %cmp.

define i32 @bar(i32 %a, i32 %b) #0 !dbg !6 {
entry:
  %inc = add i32 %a, 1, !dbg !8
  %cmp = icmp slt i32 %inc, %b
  %select = select i1 %cmp, i32 %a, i32 %b, !dbg !8
  ret i32 %select, !dbg !8
}


; CHECK-LABEL: define i32 @baz(
; CHECK: entry:
; CHECK:   %[[INC:[a-z0-9.]+]] = add i32 %a, 1, !dbg ![[DL:[0-9]+]]
; CHECK:   %[[CMP:[a-z0-9.]+]] = icmp slt i32 %[[INC]], %b
; CHECK-NOT: !dbg
; CHECK:   %[[SELECT:[a-z0-9.]+]] = select i1 %[[CMP]], i32 %a, i32 %b, !dbg ![[DL]]
;
; ![[DL]] = !DILocation(line: 3, scope: !{{.*}}, inlinedAt: {{.*}})

define i32 @baz(i32 %a, i32 %b) !dbg !9 {
entry:
  %call = tail call i32 @bar(i32 %a, i32 %b), !dbg !10
  ret i32 %call, !dbg !10
}

attributes #0 = { alwaysinline }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 2, type: !7, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 3, scope: !6)
!9 = distinct !DISubprogram(name: "baz", scope: !1, file: !1, line: 11, type: !7, isLocal: false, isDefinition: true, scopeLine: 11, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!10 = !DILocation(line: 12, scope: !9)
