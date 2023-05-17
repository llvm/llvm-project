; RUN: llc -O0 -global-isel < %s | FileCheck %s
; REQUIRES: aarch64-registered-target
target triple = "aarch64-unknown-linux-gnu"

define void @MAIN_() #0 {
L.entry:
  %0 = load <{ float, float }>, <{ float, float }>* undef, align 1
  ; CHECK: DEBUG_VALUE: localvar
  ; CHECK: DEBUG_VALUE: localvar
  call void @llvm.dbg.value(metadata <{ float, float }> %0, metadata !10, metadata !DIExpression()), !dbg !13
  unreachable
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "frame-pointer"="non-leaf" }
attributes #1 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !2, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !3, retainedTypes: !3, globals: !3, imports: !4)
!2 = !DIFile(filename: "input", directory: "/")
!3 = !{}
!4 = !{!5}
!5 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !6, entity: !9, file: !2, line: 18)
!6 = distinct !DISubprogram(name: "p", scope: !1, file: !2, line: 18, type: !7, isLocal: false, isDefinition: true, scopeLine: 18, isOptimized: false, unit: !1)
!7 = !DISubroutineType(cc: DW_CC_program, types: !8)
!8 = !{null}
!9 = !DIModule(scope: !1, name: "mod")
!10 = !DILocalVariable(name: "localvar", scope: !11, file: !2, type: !12)
!11 = !DILexicalBlock(scope: !6, file: !2, line: 18, column: 1)
!12 = !DIBasicType(name: "complex", size: 64, align: 32, encoding: DW_ATE_complex_float)
!13 = !DILocation(line: 0, scope: !11)
