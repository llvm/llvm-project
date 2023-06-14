; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mattr=avx512bw,avx512vl -o - %s -stop-before livedebugvalues | FileCheck %s

;; Check we won't salvage debug info for vector type.
; CHECK-NOT: DBG_VALUE

define <8 x i32> @foo(ptr %0, <8 x i32> %1, i8 %2, i8 %3) {
  %5 = call <8 x i32> @llvm.smax.v8i32(<8 x i32> %1, <8 x i32> zeroinitializer)
  %6 = add nsw <8 x i32> %1, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  call void @llvm.dbg.value(metadata <8 x i32> %6, metadata !4, metadata !DIExpression()), !dbg !15
  %7 = bitcast i8 %2 to <8 x i1>
  %8 = select <8 x i1> %7, <8 x i32> %6, <8 x i32> %5
  %9 = call <8 x i32> @llvm.smax.v8i32(<8 x i32> %8, <8 x i32> zeroinitializer)
  %10 = bitcast i8 %3 to <8 x i1>
  %11 = select <8 x i1> %10, <8 x i32> %9, <8 x i32> <i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255>
  ret <8 x i32> %11
}

declare <8 x i32> @llvm.smax.v8i32(<8 x i32>, <8 x i32>)
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 16.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2)
!1 = !DIFile(filename: "a.cpp", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DILocalVariable(name: "a", arg: 2, scope: !5, file: !1, line: 12, type: !11)
!5 = distinct !DISubprogram(name: "foo", scope: !6, file: !1, line: 12, type: !7, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !9, retainedNodes: !10)
!6 = !DINamespace(name: "ns1", scope: null)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !DISubprogram(name: "foo", scope: !6, file: !1, line: 132, type: !7, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!10 = !{!4}
!11 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 256, flags: DIFlagVector, elements: !13)
!12 = !DIBasicType(name: "long long", size: 64, encoding: DW_ATE_signed)
!13 = !{!14}
!14 = !DISubrange(count: 4)
!15 = !DILocation(line: 0, scope: !5, inlinedAt: !16)
!16 = !DILocation(line: 18, scope: !17)
!17 = distinct !DISubprogram(name: "foo", scope: null, file: !1, type: !7, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
