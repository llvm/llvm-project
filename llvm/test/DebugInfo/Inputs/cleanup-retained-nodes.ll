target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@var = global i8 0, align 4, !dbg !7

!llvm.module.flags = !{!0, !1, !2}
!llvm.dbg.cu = !{!3}

!0 = !{i32 7, !"Dwarf Version", i32 2}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !5, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!4 = !{}
!5 = !DIFile(filename: "tmp2.cpp", directory: "/tmp/")
!6 = !DICompositeType(tag: DW_TAG_class_type, scope: !3, file: !5, line: 212, size: 8, flags: DIFlagTypePassByValue, elements: !4, identifier: "type_global_in_another_module")
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(name: "var", scope: !3, file: !5, line: 1, type: !6, isLocal: false, isDefinition: true)
