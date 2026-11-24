; RUN: opt < %s -O2 -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv4t-none-unknown-eabi"

@foo.coefficient1 = internal unnamed_addr global ptr null, align 4, !dbg !0
@iirLow1 = external dso_local local_unnamed_addr global ptr, align 4

; Function Attrs: nounwind
define dso_local void @foo(i32 %i2) local_unnamed_addr #0 !dbg !2 {
entry:
  call void @llvm.dbg.value(metadata i32 %i2, metadata !11, metadata !DIExpression()), !dbg !18
  %0 = load ptr, ptr @iirLow1, align 4, !dbg !19
  store i32 0, ptr %0, align 4, !dbg !24
  %1 = ptrtoint ptr %0 to i32, !dbg !27
  store i32 %1, ptr @foo.coefficient1, align 4, !dbg !28
  ret void, !dbg !29
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!7}
!llvm.module.flags = !{!13, !14, !15, !16}
!llvm.ident = !{!17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "coefficient1", scope: !2, file: !3, line: 5, type: !12, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 3, type: !4, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: true, unit: !7, retainedNodes: !10)
!3 = !DIFile(filename: "norecurse_debug.c", directory: "/")
!4 = !DISubroutineType(types: !5)
!5 = !{null, !6}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !8, globals: !9, nameTableKind: None)
!8 = !{}
!9 = !{!0}
!10 = !{!11}
!11 = !DILocalVariable(name: "i2", arg: 1, scope: !2, file: !3, line: 3, type: !6)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 32)
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 4}
!16 = !{i32 1, !"min_enum_size", i32 4}
!17 = !{!""}
!18 = !DILocation(line: 3, column: 14, scope: !2)
!19 = !DILocation(line: 7, column: 6, scope: !2)
!24 = !DILocation(line: 7, column: 14, scope: !2)
!27 = !DILocation(line: 9, column: 20, scope: !2)
!28 = !DILocation(line: 9, column: 18, scope: !2)
!29 = !DILocation(line: 10, column: 1, scope: !2)

; CHECK: attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none) }
; CHECK-NOT: foo.coefficient1
