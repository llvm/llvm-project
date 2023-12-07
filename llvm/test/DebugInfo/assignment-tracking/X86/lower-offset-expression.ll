; RUN: llc %s -stop-after=finalize-isel -o - \
; RUN: | FileCheck %s

;; Handwritten test.

;; Here we have dbg.assign intrinsics with fragments (in the value-expression)
;; and address-expressions that involve arithmetic. The generated DBG_VALUE
;; intructions needs a DIExpression that:
;;     a) Uses the fragment from the value-expression,
;;     b) Uses the offset expression of the address-expression,
;;     c) Has a DW_OP_deref appended.

; CHECK: DBG_VALUE %stack.0.a, $noreg, {{.+}}, !DIExpression(DW_OP_plus_uconst, 8, DW_OP_deref, DW_OP_LLVM_fragment, 64, 32), debug-location

target triple = "x86_64-unknown-linux-gnu"

define dso_local void @fun() !dbg !7 {
entry:
  %a = alloca <4 x i32>, !DIAssignID !24
  call void @llvm.dbg.assign(metadata i32 undef, metadata !16, metadata !DIExpression(), metadata !24, metadata ptr %a, metadata !DIExpression()), !dbg !34
  ;; unlink and undef a dbg.assign to avoid using sidetable for var loc.
  call void @llvm.dbg.assign(metadata i32 undef, metadata !16, metadata !DIExpression(), metadata !26, metadata ptr undef, metadata !DIExpression()), !dbg !34
  %idx2 = getelementptr inbounds i32, i32* %a, i32 2
  store i32 100, i32* %idx2, !DIAssignID !25
  call void @llvm.dbg.assign(metadata i32 100, metadata !16, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 32), metadata !25, metadata i32* %a, metadata !DIExpression(DW_OP_plus_uconst, 8)), !dbg !34
  ret void
}

declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !1000}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0"}
!7 = distinct !DISubprogram(name: "fun", linkageName: "fun", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10, !10, !10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!16}
!16 = !DILocalVariable(name: "quad", scope: !7, file: !1, line: 4, type: !17)
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 128, elements: !18)
!18 = !{!19}
!19 = !DISubrange(count: 4)
!23 = !DILocation(line: 0, scope: !7)
!24 = distinct !DIAssignID()
!25 = distinct !DIAssignID()
!26 = distinct !DIAssignID()
!34 = !DILocation(line: 0, column: 0, scope: !7)
!44 = !{!45, !45, i64 0}
!45 = !{!"float", !46, i64 0}
!46 = !{!"omnipotent char", !47, i64 0}
!47 = !{!"Simple C++ TBAA"}
!48 = !DILocation(line: 11, column: 3, scope: !7)
!49 = !DILocation(line: 12, column: 1, scope: !7)
!50 = !DISubprogram(name: "get", linkageName: "_Z3getv", scope: !1, file: !1, line: 1, type: !51, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!51 = !DISubroutineType(types: !52)
!52 = !{!10}
!53 = !DISubprogram(name: "ext", linkageName: "_Z3extPf", scope: !1, file: !1, line: 2, type: !54, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!54 = !DISubroutineType(types: !55)
!55 = !{!10, !56}
!56 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)

!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
