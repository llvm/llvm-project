;RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s
;
; LexicalScope objects were not cleared when a nodebug function is handled in
; LiveDebugValues. This may lead to an assertion in the constructor for LexicalScope,
; triggered by LiveDebugValues when another (debug) function is handled later.
;
; This minimal example does not leave much to check for, so we just make sure we get
; reasonable output, preserving function labels and a DBG_VALUE comment.
;
; CHECK-LABEL: foo:
; CHECK-NEXT:  Lfunc_begin0:
; CHECK:       Lfunc_end0:
; CHECK-LABEL: bar:
; CHECK-NEXT:  Lfunc_begin1:
; CHECK:       #DEBUG_VALUE: foo:x <-
; CHECK:       Lfunc_end1:

define i32 @foo() {
entry:
  ret i32 0
}

define i32 @bar(i32 %x) !dbg !50 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %x, i64 0, metadata !41, metadata !43), !dbg !52
  %tobool.i = icmp eq i32 %x, 0
  br i1 %tobool.i, label %foo.exit, label %if.then.i

if.then.i:
  br label %foo.exit

foo.exit:
  %x.addr.0.i = phi i32 [ 1, %if.then.i ], [ 0, %entry ]
  ret i32 %x.addr.0.i
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.dbg.cu = !{!0, !3}
!llvm.ident = !{!5, !5}
!llvm.module.flags = !{!6, !7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "foo.cpp", directory: "c:\temp")
!2 = !{}
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!4 = !DIFile(filename: "bar.cpp", directory: "c:\temp")
!5 = !{!"clang version 4.0.0"}
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!36 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !37, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !40)
!37 = !DISubroutineType(types: !38)
!38 = !{!39, !39}
!39 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!40 = !{!41}
!41 = !DILocalVariable(name: "x", arg: 1, scope: !36, file: !1, line: 1, type: !39)
!43 = !DIExpression()
!50 = distinct !DISubprogram(name: "bar", scope: !4, file: !4, line: 3, type: !51, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !3, variables: !2)
!51 = !DISubroutineType(types: !2)
!52 = !DILocation(line: 1, scope: !36, inlinedAt: !53)
!53 = distinct !DILocation(line: 5, scope: !50)
