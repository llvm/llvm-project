; RUN: opt < %s -passes=globalopt -S | llvm-as | llvm-dis | FileCheck %s

; The %struct.S type would not get emitted after @s was removed, resulting in
; llvm-as failing to parse the dbg.value intrinsic using that type. However,
; with opaque pointers this is no longer necessary.

; CHECK-NOT: @s

; CHECK: call void @llvm.dbg.value(metadata !DIArgList(ptr poison

%struct.S = type { i32 }

@s = internal global [1 x %struct.S] zeroinitializer, align 4, !dbg !0
@idx = dso_local global i32 0, align 4, !dbg !6

define dso_local void @fn() !dbg !20 {
entry:
  %0 = load i32, ptr @idx, align 4, !dbg !26
  %idxprom = sext i32 %0 to i64, !dbg !26
  call void @llvm.dbg.value(metadata !DIArgList(ptr @s, i64 %idxprom), metadata !24, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 4, DW_OP_mul, DW_OP_plus, DW_OP_stack_value)), !dbg !26
  ret void, !dbg !26
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!15, !16, !17, !18}
!llvm.ident = !{!19}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "s", scope: !2, file: !3, line: 1, type: !9, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 13.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "foo.c", directory: "/")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "idx", scope: !2, file: !3, line: 2, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 32, elements: !13)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !3, line: 1, size: 32, elements: !11)
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "data", scope: !10, file: !3, line: 1, baseType: !8, size: 32)
!13 = !{!14}
!14 = !DISubrange(count: 1)
!15 = !{i32 7, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{i32 1, !"wchar_size", i32 4}
!18 = !{i32 7, !"uwtable", i32 1}
!19 = !{!"clang version 13.0.0"}
!20 = distinct !DISubprogram(name: "fn", scope: !3, file: !3, line: 4, type: !21, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !23)
!21 = !DISubroutineType(types: !22)
!22 = !{null}
!23 = !{!24}
!24 = !DILocalVariable(name: "local", scope: !20, file: !3, line: 5, type: !25)
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!26 = !DILocation(line: 5, scope: !20)
