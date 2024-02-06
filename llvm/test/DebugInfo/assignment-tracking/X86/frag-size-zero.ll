; RUN: llc %s -stop-after=finalize-isel -o - | FileCheck %s


; RUN: llc --try-experimental-debuginfo-iterators %s -stop-after=finalize-isel -o - | FileCheck %s

;; Check that a zero-sized fragment (the final dbg.assign) is ignored by
;; AssignmentTrackingAnalysis.

; CHECK: stack:
; CHECK-NEXT: - { id: 0, name: m4, type: default, offset: 0, size: 32, alignment: 8,
; CHECK-NEXT:     stack-id: default, callee-saved-register: '', callee-saved-restored: true,
; CHECK-NEXT:     debug-info-variable: '![[#]]', debug-info-expression: '!DIExpression()',
; CHECK-NEXT:     debug-info-location: '![[#]]' }

target triple = "x86_64-unknown-unknown"

%struct.mm = type { [8 x i32] }

@m1 = local_unnamed_addr global %struct.mm zeroinitializer, align 4, !dbg !0

define dso_local i32 @main() local_unnamed_addr #0 !dbg !23 {
entry:
  %m4 = alloca %struct.mm, align 8, !DIAssignID !28
  call void @llvm.dbg.assign(metadata i1 undef, metadata !27, metadata !DIExpression(), metadata !28, metadata ptr %m4, metadata !DIExpression()), !dbg !29
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %m4, ptr noundef nonnull align 4 dereferenceable(32) @m1, i64 32, i1 false), !dbg !31, !DIAssignID !36
  call void @llvm.dbg.assign(metadata i1 undef, metadata !27, metadata !DIExpression(), metadata !36, metadata ptr %m4, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.assign(metadata i1 undef, metadata !27, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 0), metadata !43, metadata ptr %m4, metadata !DIExpression(DW_OP_plus_uconst, 8)), !dbg !29
  ret i32 0, !dbg !45
}


declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #2
declare void @foo(i32 noundef) local_unnamed_addr #3
declare i32 @bar(ptr noundef byval(%struct.mm) align 8, ptr noundef byval(%struct.mm) align 8) local_unnamed_addr #3
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #4

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!19, !20, !21}
!llvm.ident = !{!22}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "m1", scope: !2, file: !9, line: 1, type: !10, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 17.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4, globals: !6, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "<stdin>", directory: "/")
!4 = !{!5}
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!6 = !{!0}
!9 = !DIFile(filename: "repro.c", directory: "/")
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "mm", file: !9, line: 1, size: 256, elements: !11)
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !10, file: !9, line: 1, baseType: !13, size: 256)
!13 = !DICompositeType(tag: DW_TAG_array_type, baseType: !14, size: 256, elements: !15)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !{!16}
!16 = !DISubrange(count: 8)
!19 = !{i32 2, !"Debug Info Version", i32 3}
!20 = !{i32 1, !"wchar_size", i32 4}
!21 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!22 = !{!"clang version 17.0.0"}
!23 = distinct !DISubprogram(name: "main", scope: !9, file: !9, line: 7, type: !24, scopeLine: 8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !26)
!24 = !DISubroutineType(types: !25)
!25 = !{!14}
!26 = !{!27}
!27 = !DILocalVariable(name: "m4", scope: !23, file: !9, line: 9, type: !10)
!28 = distinct !DIAssignID()
!29 = !DILocation(line: 0, scope: !23)
!31 = !DILocation(line: 11, column: 8, scope: !23)
!36 = distinct !DIAssignID()
!43 = distinct !DIAssignID()
!45 = !DILocation(line: 19, column: 3, scope: !23)
