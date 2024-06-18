; RUN: opt %s -passes=hotcoldsplit -S | FileCheck %s

;; Check the extracted DIAssignID gets remapped.

; CHECK-LABEL: define void @_foo()
; CHECK: common.ret:
; CHECK-NEXT: #dbg_assign(i64 0, ![[#]], !DIExpression(DW_OP_LLVM_fragment, 0, 64), ![[ID1:[0-9]+]], {{.*}}, !DIExpression(),

; CHECK-LABEL: define internal void @_foo.cold.1()
; CHECK: store i64 0, ptr null, align 8, !DIAssignID ![[ID2:[0-9]+]]

; CHECK-DAG: ![[ID1]] = distinct !DIAssignID()
; CHECK-DAG: ![[ID2]] = distinct !DIAssignID()

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @_foo() !dbg !4 {
entry:
  br i1 false, label %if.then7, label %common.ret

common.ret:                                       ; preds = %entry
  call void @llvm.dbg.assign(metadata i64 0, metadata !7, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64), metadata !12, metadata ptr null, metadata !DIExpression()), !dbg !13
  ret void

if.then7:                                         ; preds = %entry
  %call21 = load i1, ptr null, align 4294967296
  store i64 0, ptr null, align 8, !DIAssignID !12
  unreachable
}

declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "file.cpp", directory: "foo")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", linkageName: "_foo", scope: !5, file: !1, line: 425, type: !6, scopeLine: 425, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!5 = !DINamespace(name: "llvm", scope: null)
!6 = distinct !DISubroutineType(types: !2)
!7 = !DILocalVariable(name: "Path", scope: !4, file: !1, line: 436, type: !8)
!8 = !DIDerivedType(tag: DW_TAG_typedef, name: "string", scope: !9, file: !1, line: 79, baseType: !10)
!9 = !DINamespace(name: "std", scope: null)
!10 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "basic_string<char, std::char_traits<char>, std::allocator<char> >", scope: !11, file: !1, line: 85, size: 256, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !2, templateParams: !2, identifier: "_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE")
!11 = !DINamespace(name: "__cxx11", scope: !9, exportSymbols: true)
!12 = distinct !DIAssignID()
!13 = !DILocation(line: 0, scope: !4)
