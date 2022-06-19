; RUN: llc %s -o - -start-after=codegenprepare -stop-before=finalize-isel \
; RUN:     -mtriple=x86_64-unknown-unknown -experimental-debug-variable-locations \
; RUN: | FileCheck %s

; Test for a crash / weird behaviour when llvm.frameaddress.* is called.
; Today, as a concession, we emit a DBG_PHI allowing the frame register value
; to be read, but it could be expressed in other ways. Check that this works
; outside of the entry block, to avoid the frame register being recognised
; as def'd by frame-setup code or as a function argument.

; CHECK-LABEL: bb.1.notentry:
; CHECK: DBG_PHI $rbp

declare void @llvm.dbg.value(metadata, metadata, metadata)
declare i8 *@llvm.frameaddress.p0i8(i32)

 ; Function Attrs: mustprogress nofree nosync nounwind sspstrong uwtable
define hidden i8 * @foo() !dbg !7 {
entry:
  br label  %notentry

notentry:
  %0 = tail call i8* @llvm.frameaddress.p0i8(i32 0), !dbg !12
  call void @llvm.dbg.value(metadata i8* %0, metadata !11, metadata !DIExpression()), !dbg !12
  ret i8 *%0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "foo.cpp", directory: ".")
!2 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_signed)
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !1, file: !1, line: 6, type: !8, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{!2, !2}
!10 = !{!11}
!11 = !DILocalVariable(name: "baz", scope: !7, file: !1, line: 7, type: !2)
!12 = !DILocation(line: 10, scope: !7)
