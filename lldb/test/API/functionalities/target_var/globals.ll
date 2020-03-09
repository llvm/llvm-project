source_filename = "globals.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

@i = global i32 42, align 4
@p = global i32* @i, align 8, !dbg !0, !dbg !6

; Function Attrs: noinline nounwind optnone ssp uwtable
define i32 @main() #0 !dbg !15 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  %0 = load i32*, i32** @p, align 8, !dbg !18
  %1 = load i32, i32* %0, align 4, !dbg !18
  ret i32 %1, !dbg !18
}

attributes #0 = { noinline nounwind optnone ssp uwtable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12, !13}
!llvm.ident = !{!14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression(DW_OP_deref))
!1 = distinct !DIGlobalVariable(name: "i", scope: !2, file: !3, line: 1, type: !9, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, emissionKind: FullDebug, globals: !5)
!3 = !DIFile(filename: "globals.c", directory: "/")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "p", scope: !2, file: !3, line: 2, type: !8, isLocal: false, isDefinition: true)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 7, !"PIC Level", i32 2}
!14 = !{!"clang version 8.0.0 (trunk 340838) (llvm/trunk 340843)"}
!15 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 4, type: !16, isLocal: false, isDefinition: true, scopeLine: 4, isOptimized: false, unit: !2, retainedNodes: !4)
!16 = !DISubroutineType(types: !17)
!17 = !{!9}
!18 = !DILocation(line: 5, scope: !15)
