; ModuleID = 'pr-46466.cpp'
source_filename = "pr-46466.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux"

%struct.Struct = type { i8 }

@S = dso_local global %struct.Struct zeroinitializer, align 1, !dbg !0

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef i32 @_Z4testv() #0 !dbg !18 {
entry:
  ret i32 1, !dbg !22
}

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12, !13, !14, !15, !16}
!llvm.ident = !{!17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "S", scope: !2, file: !3, line: 8, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 20.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "pr-46466.cpp", directory: "/data/projects/scripts/regression-suite/input/general")
!4 = !{!0}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Struct", file: !3, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !6, identifier: "_ZTS6Struct")
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "U", scope: !5, file: !3, line: 5, baseType: !8, size: 8)
!8 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "Union", scope: !5, file: !3, line: 2, size: 8, flags: DIFlagTypePassByValue, elements: !9, identifier: "_ZTSN6Struct5UnionE")
!9 = !{}
!10 = !{i32 7, !"Dwarf Version", i32 5}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 8, !"PIC Level", i32 2}
!14 = !{i32 7, !"PIE Level", i32 2}
!15 = !{i32 7, !"uwtable", i32 2}
!16 = !{i32 7, !"frame-pointer", i32 2}
!17 = !{!"clang version 20.0.0"}
!18 = distinct !DISubprogram(name: "test", linkageName: "_Z4testv", scope: !3, file: !3, line: 9, type: !19, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2)
!19 = !DISubroutineType(types: !20)
!20 = !{!21}
!21 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!22 = !DILocation(line: 10, column: 3, scope: !18)
