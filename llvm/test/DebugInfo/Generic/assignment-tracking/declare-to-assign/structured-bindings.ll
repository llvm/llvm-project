; RUN: opt -passes=declare-to-assign -S %s -o - | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators -passes=declare-to-assign -S %s -o - | FileCheck %s

;; Check assignment tracking debug info for structured bindings. FIXME only
;; variables at offset 0 in the backing alloca are currently tracked with the
;; feature.

;; struct two { int a, b; };
;; two get();
;; int fun() {
;;   auto [a,b] = get();
;;   return a * b;
;; }

; CHECK: %0 = alloca %struct.two, align 4, !DIAssignID ![[ID1:[0-9]+]]
; CHECK-NEXT: llvm.dbg.assign(metadata i1 undef, metadata ![[AGGR:[0-9]+]], metadata !DIExpression(), metadata ![[ID1]], metadata ptr %0, metadata !DIExpression())
; CHECK-NEXT: llvm.dbg.assign(metadata i1 undef, metadata ![[A:[0-9]+]], metadata !DIExpression(), metadata ![[ID1]], metadata ptr %0, metadata !DIExpression())
; CHECK-NEXT: llvm.dbg.declare(metadata ptr %0, metadata ![[B:[0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 4))

; CHECK: store i64 %call, ptr %0, align 4,{{.*}}, !DIAssignID ![[ID2:[0-9]+]]
; CHECK-NEXT: llvm.dbg.assign(metadata i64 %call, metadata ![[AGGR]], metadata !DIExpression(), metadata ![[ID2]], metadata ptr %0, metadata !DIExpression())
; CHECK-NEXT: llvm.dbg.assign(metadata i64 %call, metadata ![[A]], metadata !DIExpression(), metadata ![[ID2]], metadata ptr %0, metadata !DIExpression())

; CHECK: ![[AGGR]] = !DILocalVariable(scope:
; CHECK: ![[A]] = !DILocalVariable(name: "a", scope:
; CHECK: ![[B]] = !DILocalVariable(name: "b", scope:

%struct.two = type { i32, i32 }

define dso_local noundef i32 @_Z3funv() #0 !dbg !10 {
entry:
  %0 = alloca %struct.two, align 4
  call void @llvm.dbg.declare(metadata ptr %0, metadata !15, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata ptr %0, metadata !17, metadata !DIExpression(DW_OP_plus_uconst, 4)), !dbg !18
  call void @llvm.dbg.declare(metadata ptr %0, metadata !19, metadata !DIExpression()), !dbg !24
  %call = call i64 @_Z3getv(), !dbg !25
  store i64 %call, ptr %0, align 4, !dbg !25
  %a = getelementptr inbounds %struct.two, ptr %0, i32 0, i32 0, !dbg !16
  %1 = load i32, ptr %a, align 4, !dbg !26
  %b = getelementptr inbounds %struct.two, ptr %0, i32 0, i32 1, !dbg !18
  %2 = load i32, ptr %b, align 4, !dbg !27
  %mul = mul nsw i32 %1, %2, !dbg !28
  ret i32 %mul, !dbg !29
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
!24 = !DILocation(line: 4, column: 8, scope: !10)
!25 = !DILocation(line: 4, column: 16, scope: !10)
!26 = !DILocation(line: 5, column: 10, scope: !10)
!27 = !DILocation(line: 5, column: 14, scope: !10)
!28 = !DILocation(line: 5, column: 12, scope: !10)
!29 = !DILocation(line: 5, column: 3, scope: !10)
