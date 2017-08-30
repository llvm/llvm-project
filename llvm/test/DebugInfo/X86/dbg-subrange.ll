; RUN: llc -O0 < %s | FileCheck %s
; Radar 10464995
source_filename = "test/DebugInfo/X86/dbg-subrange.ll"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.2"

@s = common global [4294967296 x i8] zeroinitializer, align 16, !dbg !0
; CHECK: .quad 4294967296 ## DW_AT_count

; Function Attrs: nounwind ssp uwtable
define void @bar() #0 !dbg !11 {
entry:
  store i8 97, i8* getelementptr inbounds ([4294967296 x i8], [4294967296 x i8]* @s, i32 0, i64 0), align 1, !dbg !14
  ret void, !dbg !16
}

attributes #0 = { nounwind ssp uwtable }

!llvm.dbg.cu = !{!7}
!llvm.module.flags = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "s", scope: null, file: !2, line: 2, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "small.c", directory: "/private/tmp")
!3 = !DICompositeType(tag: DW_TAG_array_type, baseType: !4, size: 34359738368, align: 8, elements: !5)
!4 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!5 = !{!6}
!6 = !DISubrange(count: 4294967296)
!7 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 3.1 (trunk 144833)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !8, retainedTypes: !8, globals: !9, imports: !8)
!8 = !{}
!9 = !{!0}
!10 = !{i32 1, !"Debug Info Version", i32 3}
!11 = distinct !DISubprogram(name: "bar", scope: !2, file: !2, line: 4, type: !12, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !7)
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = !DILocation(line: 5, column: 3, scope: !15)
!15 = distinct !DILexicalBlock(scope: !11, file: !2, line: 4, column: 1)
!16 = !DILocation(line: 6, column: 1, scope: !15)

