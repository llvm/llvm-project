;RUN: opt -S -passes=globalopt -f %s | FileCheck %s

;CHECK: @foo = {{.*}}, !dbg !0
@foo = global i64 ptrtoint (ptr @baa to i64), align 8, !dbg !0
@baa = common global [1 x i64] zeroinitializer, align 8, !dbg !6

; Function Attrs: noinline nounwind optnone uwtable
define void @fun() #0 !dbg !16 {
entry:
  %0 = load i64, ptr @foo, align 8, !dbg !19
  %1 = inttoptr i64 %0 to ptr, !dbg !19
  %cmp = icmp ugt ptr @baa, %1, !dbg !20
  %conv = zext i1 %cmp to i32, !dbg !20
  store i64 0, ptr @foo, align 8, !dbg !21
  ret void, !dbg !22
}

attributes #0 = { noinline nounwind optnone uwtable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "foo", scope: !2, file: !3, line: 2, type: !9, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 6.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "shrink-address-to-bool.c", directory: "/")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "baa", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!8 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 64, elements: !10)
!9 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!10 = !{!11}
!11 = !DISubrange(count: 1)
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{!"clang version 6.0.0 "}
!16 = distinct !DISubprogram(name: "fun", scope: !3, file: !3, line: 4, type: !17, isLocal: false, isDefinition: true, scopeLine: 4, isOptimized: false, unit: !2, retainedNodes: !4)
!17 = !DISubroutineType(types: !18)
!18 = !{null}
!19 = !DILocation(line: 5, column: 9, scope: !16)
!20 = !DILocation(line: 5, column: 7, scope: !16)
!21 = !DILocation(line: 6, column: 7, scope: !16)
!22 = !DILocation(line: 7, column: 1, scope: !16)
