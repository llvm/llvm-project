; RUN: llc -o - %s -fast-isel -stop-before=finalize-isel | FileCheck %s
; Make sure fast-isel produces DBG_VALUE instructions even if no debug printer
; is scheduled because of -stop-before.
target triple="aarch64--"

; CHECK-LABEL: name: func
; CHECK: DBG_VALUE
define void @func(i32 %a) !dbg !4 {
  call void @llvm.dbg.declare(metadata i32 %a, metadata !5, metadata !DIExpression()), !dbg !7
  ret void
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #0
attributes #0 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "fast-isel-dbg.ll", directory: "/")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "func", scope: null, isLocal: false, isDefinition: true, isOptimized: false, unit: !0)
!5 = !DILocalVariable(name: "a", arg: 1, scope: !4, file: !1, line: 17, type: !6)
!6 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!7 = !DILocation(line: 17, scope: !4)
