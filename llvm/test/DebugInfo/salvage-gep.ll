; RUN: opt %s -passes=dce -S | FileCheck %s

; Tests the salvaging of GEP instructions, specifically struct indexing,
; non-constant array indexing, and non-constant array indexing into an array of
; a type with width 0.

%struct.S = type { i32, i32 }
%zero = type [0 x [10 x i32]]

;; The constant and variable offsets should be applied correctly.
; CHECK: call void @llvm.dbg.value(metadata !DIArgList(ptr %ptr, i64 %offset),
; CHECK-SAME: ![[VAR_OFFSET_PTR:[0-9]+]],
; CHECK-SAME: !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 8, DW_OP_mul, DW_OP_plus, DW_OP_plus_uconst, 4, DW_OP_stack_value))

;; The variable offset should be ignored, as it applies to a type of width 0,
;; leaving only the constant offset.
; CHECK: call void @llvm.dbg.value(metadata ptr %zptr,
; CHECK-SAME: ![[VAR_ZERO_PTR:[0-9]+]],
; CHECK-SAME: !DIExpression(DW_OP_plus_uconst, 44, DW_OP_stack_value))

; CHECK: ![[VAR_OFFSET_PTR]] = !DILocalVariable(name: "offset_ptr"
; CHECK: ![[VAR_ZERO_PTR]] = !DILocalVariable(name: "zero_ptr"

define void @"?foo@@YAXPEAUS@@_J@Z"(ptr %ptr, ptr %zptr, i64 %offset) !dbg !8 {
entry:
  call void @llvm.dbg.value(metadata i64 %offset, metadata !20, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata ptr %ptr, metadata !21, metadata !DIExpression()), !dbg !24
  %arrayidx = getelementptr inbounds %struct.S, ptr %ptr, i64 %offset, !dbg !25
  %b = getelementptr inbounds %struct.S, ptr %arrayidx, i32 0, i32 1, !dbg !25
  %c = getelementptr inbounds %zero, ptr %zptr, i64 %offset, i32 1, i32 1, !dbg !25
  call void @llvm.dbg.value(metadata ptr %b, metadata !22, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata ptr %c, metadata !27, metadata !DIExpression()), !dbg !24
  ret void, !dbg !26
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 11.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "salvage-gep.cpp", directory: "/")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 11.0.0"}
!8 = distinct !DISubprogram(name: "foo", linkageName: "?foo@@YAXPEAUS@@_J@Z", scope: !9, file: !9, line: 7, type: !10, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !19)
!9 = !DIFile(filename: ".\\salvage-gep.cpp", directory: "/")
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12, !18}
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !9, line: 2, size: 64, flags: DIFlagTypePassByValue, elements: !14, identifier: ".?AUS@@")
!14 = !{!15, !17}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !13, file: !9, line: 3, baseType: !16, size: 32)
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !13, file: !9, line: 4, baseType: !16, size: 32, offset: 32)
!18 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!19 = !{!20, !21, !22}
!20 = !DILocalVariable(name: "offset", arg: 2, scope: !8, file: !9, line: 7, type: !18)
!21 = !DILocalVariable(name: "ptr", arg: 1, scope: !8, file: !9, line: 7, type: !12)
!22 = !DILocalVariable(name: "offset_ptr", scope: !8, file: !9, line: 8, type: !23)
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!24 = !DILocation(line: 0, scope: !8)
!25 = !DILocation(line: 8, scope: !8)
!26 = !DILocation(line: 9, scope: !8)
!27 = !DILocalVariable(name: "zero_ptr", scope: !8, file: !9, line: 8, type: !23)
