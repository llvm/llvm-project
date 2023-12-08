; RUN: llc -disable-debug-info-print=true -exception-model=dwarf -o - %s | FileCheck %s
; RUN: llc -disable-debug-info-print=true -exception-model=sjlj -o - %s | FileCheck %s --check-prefix=SJLJ-CHECK

define i16 @main() nounwind !dbg !7 {
entry:
  ret i16 0, !dbg !9
}

define i16 @helper() !dbg !10 {
entry:
  ret i16 0, !dbg !11
}


; CHECK: main
; CHECK-NOT: cfi_startproc
; CHECK-NOT: .file
; CHECK-NOT: .loc
; CHECK: helper
; CHECK: cfi_startproc
; CHECK-NOT: .file
; CHECK-NOT: .loc
; CHECK: cfi_endproc

; SJLJ-CHECK: main
; SJLJ-CHECK-NOT: cfi_startproc
; SJLJ-CHECK-NOT: .file
; SJLJ-CHECK-NOT: .loc
; SJLJ-CHECK: helper
; SJLJ-CHECK-NOT: cfi_startproc
; SJLJ-CHECK-NOT: .file
; SJLJ-CHECK-NOT: .loc


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "unwind-tables.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 2, column: 3, scope: !7)
!10 = distinct !DISubprogram(name: "helper", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!11 = !DILocation(line: 2, column: 3, scope: !10)
