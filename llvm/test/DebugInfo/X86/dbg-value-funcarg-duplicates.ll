; RUN: llc -mtriple=x86_64-unknown-linux-gnu -start-after=codegenprepare -stop-before=finalize-isel -o - %s -experimental-debug-variable-locations=false | FileCheck %s

; Input to this test was created by reducing a Swift file using bugpoint

; CHECK-DAG: ![[LHS:.*]] = !DILocalVariable(name: "lhs"

define hidden i64 @"_wideDivide42"(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8) local_unnamed_addr !dbg !16 {
; CHECK-LABEL: name:            _wideDivide42
; CHECK-NOT:  DBG_VALUE
; CHECK:      DBG_VALUE $rcx, $noreg, ![[LHS]], !DIExpression(DW_OP_LLVM_fragment, 0, 64)
; CHECK-NEXT: DBG_VALUE $r8, $noreg, ![[LHS]], !DIExpression(DW_OP_LLVM_fragment, 64, 64)
; CHECK-NEXT: DBG_VALUE $r9, $noreg, ![[LHS]], !DIExpression(DW_OP_LLVM_fragment, 128, 64)
; CHECK-NEXT: DBG_VALUE %fixed-stack.{{.+}}, ![[LHS]], !DIExpression(DW_OP_LLVM_fragment, 192, 64)
; The duplicates should be removed:
; CHECK-NOT:  DBG_VALUE

entry:
  %9 = alloca i64, align 8
  call void @llvm.dbg.value(metadata i64 %3, metadata !24, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64)), !dbg !67
  call void @llvm.dbg.value(metadata i64 %4, metadata !24, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64)), !dbg !67
  call void @llvm.dbg.value(metadata i64 %3, metadata !24, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64)), !dbg !67
  call void @llvm.dbg.value(metadata i64 %4, metadata !24, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64)), !dbg !67
  call void @llvm.dbg.value(metadata i64 %5, metadata !24, metadata !DIExpression(DW_OP_LLVM_fragment, 128, 64)), !dbg !67
  call void @llvm.dbg.value(metadata i64 %6, metadata !24, metadata !DIExpression(DW_OP_LLVM_fragment, 192, 64)), !dbg !67
  br i1 poison, label %11, label %10, !dbg !68

10:                                               ; preds = %entry
  tail call void asm sideeffect "", "n"(i32 7) #7
  unreachable

11:                                               ; preds = %entry
  tail call void @abort()
  unreachable
}

declare void @abort()

declare void @llvm.dbg.value(metadata, metadata, metadata)

attributes #7 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13}
!llvm.linker.options = !{!14, !15}

!0 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !1, producer: "Swift", isOptimized: true, runtimeVersion: 6, emissionKind: FullDebug)
!1 = !DIFile(filename: "Int128.swift", directory: "")
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{!"-lswiftCore"}
!15 = !{!"-lobjc"}
!16 = distinct !DISubprogram(name: "_wideDivide42", scope: !0, file: !1, line: 222, type: !17, scopeLine: 222, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !23)
!17 = !DISubroutineType(types: !18)
!18 = !{!19, !20, !20, !20, !20, !20, !20}
!19 = !DICompositeType(tag: DW_TAG_structure_type, name: "4 x UInt64", flags: DIFlagFwdDecl, runtimeLang: DW_LANG_Swift)
!20 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "UInt64", scope: !1, file: !1, size: 64, elements: !22, runtimeLang: DW_LANG_Swift)
!22 = !{}
!23 = !{!24, !27}
!24 = !DILocalVariable(name: "lhs", arg: 1, scope: !16, file: !1, line: 223, type: !25, flags: DIFlagArtificial)
!25 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !26)
!26 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "2 x 2 x UInt64", file: !1, size: 256, elements: !22, runtimeLang: DW_LANG_Swift)
!27 = !DILocalVariable(name: "rhs", arg: 2, scope: !16, file: !1, line: 223, type: !28, flags: DIFlagArtificial)
!28 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !29)
!29 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "2 x UInt64", file: !1, size: 128, elements: !22, runtimeLang: DW_LANG_Swift)
!67 = !DILocation(line: 0, scope: !16)
!68 = !DILocation(line: 225, column: 9, scope: !16)
