; RUN: llc < %s - | FileCheck %s
;
; FIXME: A potentially more interesting test case would be:
; %call = @bar()
; dbg.value j=0
; %call2 = @bar()
; dbg.value j=%call
;
; We cannot current handle the above sequence because codegenprepare
; hoists the second dbg.value above %call2, which then appears to
; conflict with j=0. It does this because SelectionDAG cannot handle
; global debug values.

target triple = "x86_64-apple-darwin10.0.0"

;CHECK:        ## DW_OP_consts
;CHECK-NEXT:  .byte	42
define i32 @foobar() nounwind readonly noinline ssp !dbg !0 {
entry:
  tail call void @llvm.dbg.value(metadata i32 42, metadata !6, metadata !DIExpression()), !dbg !9
  %call = tail call i32 @bar(), !dbg !11
  tail call void @llvm.dbg.value(metadata i32 %call, metadata !6, metadata !DIExpression()), !dbg !11
  %call2 = tail call i32 @bar(), !dbg !11
  %add = add nsw i32 %call2, %call, !dbg !12
  ret i32 %add, !dbg !10
}

declare void @llvm.dbg.value(metadata, metadata, metadata) nounwind readnone
declare i32 @bar() nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!17}

!0 = distinct !DISubprogram(name: "foobar", linkageName: "foobar", line: 12, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !2, file: !15, scope: !1, type: !3, variables: !14)
!1 = !DIFile(filename: "mu.c", directory: "/private/tmp")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 2.9 (trunk 114183)", isOptimized: true, emissionKind: FullDebug, file: !15, enums: !16, retainedTypes: !16, imports:  null)
!3 = !DISubroutineType(types: !4)
!4 = !{!5}
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !DILocalVariable(name: "j", line: 15, scope: !7, file: !1, type: !5)
!7 = distinct !DILexicalBlock(line: 12, column: 52, file: !15, scope: !0)
!8 = !{i32 42}
!9 = !DILocation(line: 15, column: 12, scope: !7)
!10 = !DILocation(line: 23, column: 3, scope: !7)
!11 = !DILocation(line: 17, column: 3, scope: !7)
!12 = !DILocation(line: 18, column: 3, scope: !7)
!14 = !{!6}
!15 = !DIFile(filename: "mu.c", directory: "/private/tmp")
!16 = !{}
!17 = !{i32 1, !"Debug Info Version", i32 3}
