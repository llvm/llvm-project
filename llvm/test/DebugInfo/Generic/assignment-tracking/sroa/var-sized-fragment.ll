; RUN: opt -S %s -o - -passes=sroa | FileCheck %s

;; SROA splits the alloca into two. Each slice already has a 32-bit variable
;; associated with it (the structured binding variables). Check SROA doesn't
;; mistakenly create 32 bit fragments for them.

;; NOTE: The upper 32 bits are currently
;; not tracked with assignment tracking (due to the dbg.declare containing an
;; expression).
;; The dbg intrinsics for the unnamed aggregate variable have been commented
;; out to reduce test clutter.

;; From C++ source:
;; class two {public:int a; int b;}
;; two get();
;; int fun() {
;;   auto [a, b] = get();
;;   return a;
;; }

;; FIXME: Variable 'b' gets an incorrect location (value and expression) - see
;; llvm.org/PR61981. This check just ensures that no fragment info is added to
;; the dbg.value.
; CHECK: dbg.value(metadata i32 %.sroa.0.0.extract.trunc, metadata ![[B:[0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 4))
; CHECK: dbg.value(metadata i32 %.sroa.0.0.extract.trunc, metadata ![[A:[0-9]+]], metadata !DIExpression())
; CHECK: ![[A]] = !DILocalVariable(name: "a",
; CHECK: ![[B]] = !DILocalVariable(name: "b",

%class.two = type { i32, i32 }

define dso_local noundef i32 @_Z3funv() #0 !dbg !10 {
entry:
  %0 = alloca %class.two, align 4, !DIAssignID !23
  ;;call void @llvm.dbg.assign(metadata i1 undef, metadata !17, metadata !DIExpression(), metadata !23, metadata ptr %0, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.assign(metadata i1 undef, metadata !15, metadata !DIExpression(), metadata !23, metadata ptr %0, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.declare(metadata ptr %0, metadata !16, metadata !DIExpression(DW_OP_plus_uconst, 4)), !dbg !26
  %call = call i64 @_Z3getv()
  store i64 %call, ptr %0, align 4, !DIAssignID !28
  ;;call void @llvm.dbg.assign(metadata i64 %call, metadata !17, metadata !DIExpression(), metadata !28, metadata ptr %0, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.assign(metadata i64 %call, metadata !15, metadata !DIExpression(), metadata !28, metadata ptr %0, metadata !DIExpression()), !dbg !24
  %a = getelementptr inbounds %class.two, ptr %0, i32 0, i32 0
  %1 = load i32, ptr %a, align 4
  ret i32 %1
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #2
declare !dbg !38 i64 @_Z3getv() #3
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #2

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 17.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "h")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 17.0.0"}
!10 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !1, file: !1, line: 3, type: !11, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{!15, !16, !17}
!15 = !DILocalVariable(name: "a", scope: !10, file: !1, line: 4, type: !13)
!16 = !DILocalVariable(name: "b", scope: !10, file: !1, line: 4, type: !13)
!17 = !DILocalVariable(scope: !10, file: !1, line: 4, type: !18)
!18 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "two", file: !19, line: 1, size: 64, flags: DIFlagTypePassByValue, elements: !20, identifier: "_ZTS3two")
!19 = !DIFile(filename: "./include.h", directory: "/")
!20 = !{!21, !22}
!21 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !18, file: !19, line: 1, baseType: !13, size: 32, flags: DIFlagPublic)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !18, file: !19, line: 1, baseType: !13, size: 32, offset: 32, flags: DIFlagPublic)
!23 = distinct !DIAssignID()
!24 = !DILocation(line: 0, scope: !10)
!26 = !DILocation(line: 4, column: 12, scope: !10)
!28 = distinct !DIAssignID()
!38 = !DISubprogram(name: "get", linkageName: "_Z3getv", scope: !1, file: !1, line: 2, type: !39, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !41)
!39 = !DISubroutineType(types: !40)
!40 = !{!18}
!41 = !{}
