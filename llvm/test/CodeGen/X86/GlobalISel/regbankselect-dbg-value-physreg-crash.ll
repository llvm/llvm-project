; RUN: llc -mtriple=x86_64-linux-gnu -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s

declare void @llvm.dbg.value(metadata, metadata, metadata)

; CHECK-LABEL: foo:
; CHECK:       #DEBUG_VALUE: foo:p <- $rdi
; CHECK:       retq
define void @foo(ptr %p) !dbg !0 {
entry:
  call void @llvm.dbg.value(metadata ptr %p, metadata !4, metadata !DIExpression()), !dbg !5
  ret void, !dbg !5
}

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!8, !9}

!0 = distinct !DISubprogram(name: "foo", scope: !2, file: !2, line: 1, type: !3, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !7)
!1 = distinct !DICompileUnit(language: DW_LANG_C11, file: !2, emissionKind: FullDebug)
!2 = !DIFile(filename: "regbankselect-dbg-value-physreg-crash.c", directory: "/")
!3 = !DISubroutineType(types: !6)
!4 = !DILocalVariable(name: "p", arg: 1, scope: !0, file: !2, line: 1, type: !10)
!5 = !DILocation(line: 1, column: 1, scope: !0)
!6 = !{null, !10}
!7 = !{!4}
!8 = !{i32 7, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
