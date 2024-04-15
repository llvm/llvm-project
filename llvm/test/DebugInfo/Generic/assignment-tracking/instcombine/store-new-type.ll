; RUN: opt %s -passes=instcombine -S | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators %s -passes=instcombine -S | FileCheck %s

;; Based on test/Transforms/InstCombine/shufflevec-bitcast.ll in which the
;; store of <4 x i4> is replaced with a store of type <2 x i8>. Debug info
;; added by hand. Check the DIAssignID attachment on the store is preserved.

define <2 x i4> @shuf_bitcast_insert_use2(<2 x i8> %v, i8 %x, ptr %p) {
; CHECK-LABEL: @shuf_bitcast_insert_use2(
; CHECK-NEXT:    [[I:%.*]] = insertelement <2 x i8> [[V:%.*]], i8 [[X:%.*]], i64 0
; CHECK-NEXT:    store <2 x i8> [[I]], ptr [[P:%.*]], align 2, !DIAssignID ![[ID:[0-9]+]]
; CHECK-NEXT:    dbg.assign(metadata <2 x i8> %i, {{.+}}, {{.+}}, metadata ![[ID]], metadata ptr %p,{{.+}})
; CHECK-NEXT:    [[R:%.*]] = bitcast i8 [[X]] to <2 x i4>
; CHECK-NEXT:    ret <2 x i4> [[R]]
;
  %i = insertelement <2 x i8> %v, i8 %x, i32 0
  %b = bitcast <2 x i8> %i to <4 x i4>
  store <4 x i4> %b, ptr %p, !DIAssignID !37
  call void @llvm.dbg.assign(metadata <4 x i4> %b, metadata !28, metadata !DIExpression(), metadata !37, metadata ptr %p, metadata !DIExpression()), !dbg !26
  %r = shufflevector <4 x i4> %b, <4 x i4> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x i4> %r
}

declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13}
!llvm.ident = !{!19}

!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 16.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "/")
!4 = !{}
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "int4", file: !3, line: 1, baseType: !8)
!8 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 128, flags: DIFlagVector, elements: !10)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{!11}
!11 = !DISubrange(count: 4)
!12 = !{i32 7, !"Dwarf Version", i32 5}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{!"clang version 16.0.0"}
!20 = distinct !DISubprogram(name: "f", linkageName: "_Z1fi", scope: !3, file: !3, line: 3, type: !21, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !23)
!21 = !DISubroutineType(types: !22)
!22 = !{!7, !9}
!23 = !{}
!26 = !DILocation(line: 0, scope: !20)
!28 = !DILocalVariable(name: "a", scope: !20, file: !3, line: 3, type: !29)
!29 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 256, elements: !30)
!30 = !{!31}
!31 = !DISubrange(count: 2)
!37 = distinct !DIAssignID()

