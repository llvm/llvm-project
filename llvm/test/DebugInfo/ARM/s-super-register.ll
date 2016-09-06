; RUN: llc < %s - -filetype=obj | llvm-dwarfdump -debug-dump=loc - | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-macosx10.6.7"

; The S registers on ARM are expressed as pieces of their super-registers in DWARF.
;
; 0x90   DW_OP_regx of super-register
; 0x93   DW_OP_piece
; 0x9d   DW_OP_bit_piece
; CHECK:            Location description: 90 {{.. .. ((93 ..)|(9d .. ..)) $}}

define void @_Z3foov() optsize ssp !dbg !1 {
entry:
  %call = tail call float @_Z3barv() optsize, !dbg !11
  tail call void @llvm.dbg.value(metadata float %call, i64 0, metadata !5, metadata !DIExpression()), !dbg !11
  %call16 = tail call float @_Z2f2v() optsize, !dbg !12
  %cmp7 = fcmp olt float %call, %call16, !dbg !12
  br i1 %cmp7, label %for.body, label %for.end, !dbg !12

for.body:                                         ; preds = %entry, %for.body
  %k.08 = phi float [ %inc, %for.body ], [ %call, %entry ]
  %call4 = tail call float @_Z2f3f(float %k.08) optsize, !dbg !13
  %inc = fadd float %k.08, 1.000000e+00, !dbg !14
  %call1 = tail call float @_Z2f2v() optsize, !dbg !12
  %cmp = fcmp olt float %inc, %call1, !dbg !12
  br i1 %cmp, label %for.body, label %for.end, !dbg !12

for.end:                                          ; preds = %for.body, %entry
  ret void, !dbg !15
}

declare float @_Z3barv() optsize

declare float @_Z2f2v() optsize

declare float @_Z2f3f(float) optsize

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!20}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.0 (trunk 130845)", isOptimized: true, emissionKind: FullDebug, file: !18, enums: !19, retainedTypes: !19, imports:  null)
!1 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 5, file: !18, scope: !2, type: !3, variables: !17)
!2 = !DIFile(filename: "k.cc", directory: "/private/tmp")
!3 = !DISubroutineType(types: !4)
!4 = !{null}
!5 = !DILocalVariable(name: "k", line: 6, scope: !6, file: !2, type: !7)
!6 = distinct !DILexicalBlock(line: 5, column: 12, file: !18, scope: !1)
!7 = !DIBasicType(tag: DW_TAG_base_type, name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!8 = !DILocalVariable(name: "y", line: 8, scope: !9, file: !2, type: !7)
!9 = distinct !DILexicalBlock(line: 7, column: 25, file: !18, scope: !10)
!10 = distinct !DILexicalBlock(line: 7, column: 3, file: !18, scope: !6)
!11 = !DILocation(line: 6, column: 18, scope: !6)
!12 = !DILocation(line: 7, column: 3, scope: !6)
!13 = !DILocation(line: 8, column: 20, scope: !9)
!14 = !DILocation(line: 7, column: 20, scope: !10)
!15 = !DILocation(line: 10, column: 1, scope: !6)
!17 = !{!5, !8}
!18 = !DIFile(filename: "k.cc", directory: "/private/tmp")
!19 = !{}
!20 = !{i32 1, !"Debug Info Version", i32 3}
