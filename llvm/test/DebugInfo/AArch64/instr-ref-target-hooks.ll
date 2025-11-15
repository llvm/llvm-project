; Test to ensure that variable "__last" is properly recovered at the end of the livedebugvalues pass when Instruction Referencing-based LiveDebugValues is used.
; This testcase was obtained by looking at FileCheck.cpp and reducing it down via llvm-reduce.

; RUN: llc -mtriple=aarch64-apple-darwin -o - %s -stop-after=livedebugvalues -O2 -experimental-debug-variable-locations | FileCheck %s

; CHECK: ![[LOC:[0-9]+]] = !DILocalVariable(name: "__last",
; CHECK: DBG_VALUE_LIST ![[LOC]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_plus_uconst, 8, DW_OP_deref), $sp


declare void @_ZdlPvm()
define fastcc void @"_ZNSt3__111__introsortINS_17_ClassicAlgPolicyERZL18DumpAnnotatedInputRN4llvm11raw_ostreamERKNS2_16FileCheckRequestE20DumpInputFilterValuejNS2_9StringRefERNS_6vectorI15InputAnnotationNS_9allocatorISB_EEEEjE3$_0PSB_Lb0EEEvT1_SJ_T0_NS_15iterator_traitsISJ_E15difference_typeEb"(ptr %__first, ptr %__last, i1 %cmp, ptr %__first.addr.0, ptr %Label3.i.i.i241, ptr %__pivot.sroa.9113.8.copyload.i, ptr %0, ptr %1) !dbg !4 {
  br label %while.cond
while.cond:                                       ; preds = %if.end16, %entry
  br i1 %cmp, label %if.then13, label %if.end16
if.then13:                                        ; preds = %while.cond
  %cmp.i = icmp eq ptr %__first, %__last
  %or.cond.i = select i1 %cmp.i, i1 false, i1 false
    #dbg_value(ptr %__last, !10, !DIExpression(), !16)
  br i1 %or.cond.i, label %common.ret, label %for.body.i, !dbg !23
common.ret:                                       ; preds = %for.body.i, %if.then13
  ret void
for.body.i:                                       ; preds = %if.then13
  %InputLine.i.i = getelementptr i8, ptr %__first.addr.0, i64 132
  br label %common.ret
if.end16:                                         ; preds = %while.cond
  %__pivot.sroa.13.8.copyload.i = load i64, ptr null, align 8
  call void @_ZdlPvm()
  store ptr %__pivot.sroa.9113.8.copyload.i, ptr %0, align 8
  store i64 %__pivot.sroa.13.8.copyload.i, ptr %1, align 8
  store i64 0, ptr %__first, align 8
  store i32 0, ptr %__first.addr.0, align 8
  store i32 1, ptr %Label3.i.i.i241, align 4
  br label %while.cond
}
!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !2, producer: "clang version 22.0.0git (git@github.com:llvm/llvm-project.git 46a3b4d5dc6dd9449ec7c0c9065552368cdf41d6)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !3, retainedTypes: !3, globals: !3, imports: !3, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/Library/Developer/CommandLineTools/SDKs/MacOSX15.3.sdk", sdk: "MacOSX15.3.sdk")
!2 = !DIFile(filename: "/Users/shubhamrastogi/Development/llvm-project-instr-ref/llvm-project/llvm/utils/FileCheck/FileCheck.cpp", directory: "/Users/shubhamrastogi/Development/llvm-project-instr-ref/llvm-project/build-instr-ref-stage2", checksumkind: CSK_MD5, checksum: "fa5f53f1b5782eb8b92fadec416b8941")
!3 = !{}
!4 = distinct !DISubprogram(name: "__introsort<std::__1::_ClassicAlgPolicy, (lambda at /Users/shubhamrastogi/Development/llvm-project-instr-ref/llvm-project/llvm/utils/FileCheck/FileCheck.cpp:544:14) &, InputAnnotation *, false>", linkageName: "_ZNSt3__111__introsortINS_17_ClassicAlgPolicyERZL18DumpAnnotatedInputRN4llvm11raw_ostreamERKNS2_16FileCheckRequestE20DumpInputFilterValuejNS2_9StringRefERNS_6vectorI15InputAnnotationNS_9allocatorISB_EEEEjE3$_0PSB_Lb0EEEvT1_SJ_T0_NS_15iterator_traitsISJ_E15difference_typeEb", scope: !6, file: !5, line: 758, type: !8, scopeLine: 762, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !1, templateParams: !3, retainedNodes: !3, keyInstructions: true)
!5 = !DIFile(filename: "/Library/Developer/CommandLineTools/SDKs/MacOSX15.3.sdk/usr/include/c++/v1/__algorithm/sort.h", directory: "")
!6 = !DINamespace(name: "__1", scope: !7, exportSymbols: true)
!7 = !DINamespace(name: "std", scope: null)
!8 = !DISubroutineType(cc: DW_CC_nocall, types: !9)
!9 = !{null}
!10 = !DILocalVariable(name: "__last", arg: 2, scope: !11, file: !5, line: 284, type: !13)
!11 = distinct !DISubprogram(name: "__insertion_sort<std::__1::_ClassicAlgPolicy, (lambda at /Users/shubhamrastogi/Development/llvm-project-instr-ref/llvm-project/llvm/utils/FileCheck/FileCheck.cpp:544:14) &, InputAnnotation *>", linkageName: "_ZNSt3__116__insertion_sortB8nn180100INS_17_ClassicAlgPolicyERZL18DumpAnnotatedInputRN4llvm11raw_ostreamERKNS2_16FileCheckRequestE20DumpInputFilterValuejNS2_9StringRefERNS_6vectorI15InputAnnotationNS_9allocatorISB_EEEEjE3$_0PSB_EEvT1_SJ_T0_", scope: !6, file: !5, line: 284, type: !12, scopeLine: 284, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !1, templateParams: !3, retainedNodes: !3, keyInstructions: true)
!12 = distinct !DISubroutineType(types: !9)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "InputAnnotation", file: !15, line: 323, size: 768, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !3, identifier: "_ZTS15InputAnnotation")
!15 = !DIFile(filename: "llvm/utils/FileCheck/FileCheck.cpp", directory: "/Users/shubhamrastogi/Development/llvm-project-instr-ref/llvm-project", checksumkind: CSK_MD5, checksum: "fa5f53f1b5782eb8b92fadec416b8941")
!16 = !DILocation(line: 0, scope: !11, inlinedAt: !17)
!17 = distinct !DILocation(line: 800, column: 9, scope: !18)
!18 = distinct !DILexicalBlock(scope: !22, file: !5, line: 799, column: 23)
!22 = distinct !DILexicalBlock(scope: !4, file: !5, line: 770, column: 16)
!23 = !DILocation(line: 288, column: 15, scope: !24, inlinedAt: !17, atomGroup: 1, atomRank: 1)
!24 = distinct !DILexicalBlock(scope: !11, file: !5, line: 288, column: 7)
