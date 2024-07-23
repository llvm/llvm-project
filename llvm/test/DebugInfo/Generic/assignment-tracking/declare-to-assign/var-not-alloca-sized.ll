; RUN: opt -passes=declare-to-assign -S %s -o - | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators -passes=declare-to-assign -S %s -o - | FileCheck %s

;; The variable doesn't fill the whole alloca which has a range of different
;; sized stores to it, overlapping (or not) the variable in various ways. Check
;; the fragment is truncated to represent the intersect between the store and
;; the variable. If that intersect has exactly the same offset and size as the
;; variable then a fragment should not be produced (the whole variable is
;; covered by the store).
;;
;; Check directives written inline.

%struct.two = type { i32, i32 }

define dso_local noundef i32 @_Z3funv() #0 !dbg !10 {
entry:
  %0 = alloca [4 x i16], align 4
  call void @llvm.dbg.declare(metadata ptr %0, metadata !15, metadata !DIExpression()), !dbg !16
; CHECK: %0 = alloca [4 x i16], align 4, !DIAssignID ![[ID1:[0-9]+]]
; CHECK-NEXT: #dbg_assign(i1 undef, ![[#]], !DIExpression(), ![[ID1]], ptr %0, !DIExpression(),
  %a = getelementptr inbounds [4 x i16], ptr %0, i32 0, i32 0
  %a.5 = getelementptr inbounds [4 x i16], ptr %0, i32 0, i32 1
  %b = getelementptr inbounds [4 x i16], ptr %0, i32 0, i32 2
  store i64 1, ptr %a, align 4
; CHECK: store i64 1, ptr %a, align 4, !DIAssignID ![[ID2:[0-9]+]]
; CHECK-NEXT: #dbg_assign(i64 1, ![[#]], !DIExpression(), ![[ID2]], ptr %a, !DIExpression(),
  store i64 2, ptr %b, align 4
;; %b is outside the variable bounds, no debug intrinsic needed.
  store i16 3, ptr %a.5, align 4
; CHECK: store i16 3, ptr %a.5, align 4, !DIAssignID ![[ID3:[0-9]+]]
; CHECK-NEXT: #dbg_assign(i16 3, ![[#]], !DIExpression(DW_OP_LLVM_fragment, 16, 16), ![[ID3]], ptr %a.5, !DIExpression(),
  store i32 4, ptr %a.5, align 4
; CHECK: store i32 4, ptr %a.5, align 4, !DIAssignID ![[ID4:[0-9]+]]
; CHECK-NEXT: #dbg_assign(i32 4, ![[#]], !DIExpression(DW_OP_LLVM_fragment, 16, 16), ![[ID4]], ptr %a.5, !DIExpression(),
  store i32 5, ptr %a, align 4
; CHECK: store i32 5, ptr %a, align 4, !DIAssignID ![[ID5:[0-9]+]]
; CHECK-NEXT: #dbg_assign(i32 5, ![[#]], !DIExpression(), ![[ID5]], ptr %a, !DIExpression(),
  ret i32 0
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare i64 @_Z3getv()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 17.0.0)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 17.0.0)"}
!10 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !1, file: !1, line: 3, type: !11, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{}
!15 = !DILocalVariable(name: "a", scope: !10, file: !1, line: 4, type: !13)
!16 = !DILocation(line: 4, column: 9, scope: !10)
!17 = !DILocalVariable(name: "b", scope: !10, file: !1, line: 4, type: !13)
!18 = !DILocation(line: 4, column: 11, scope: !10)
!19 = !DILocalVariable(scope: !10, file: !1, line: 4, type: !20)
!20 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "two", file: !1, line: 1, size: 64, flags: DIFlagTypePassByValue, elements: !21, identifier: "_ZTS3two")
!21 = !{!22, !23}
!22 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !20, file: !1, line: 1, baseType: !13, size: 32)
!23 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !20, file: !1, line: 1, baseType: !13, size: 32, offset: 32)
!25 = !DILocation(line: 4, column: 16, scope: !10)
