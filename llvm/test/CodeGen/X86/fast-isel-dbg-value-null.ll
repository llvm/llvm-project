; RUN: llc -O0 -fast-isel -fast-isel-abort=3 -mtriple=x86_64-unknown-unknown \
; RUN:   -stop-after=finalize-isel %s -o - | FileCheck %s

; FastISel describes null and integer-to-pointer constants as immediates, like
; SelectionDAG and GlobalISel do, instead of dropping them. The integer is
; zero-extended or truncated to the pointer width, as inttoptr specifies.

; CHECK: ![[NULL:[0-9]+]] = !DILocalVariable(name: "null_var"
; CHECK: ![[NULL_AS1:[0-9]+]] = !DILocalVariable(name: "null_as1_var"
; CHECK: ![[INTTOPTR:[0-9]+]] = !DILocalVariable(name: "inttoptr_var"
; CHECK: ![[NARROW:[0-9]+]] = !DILocalVariable(name: "narrow_inttoptr_var"
; CHECK: ![[WIDE:[0-9]+]] = !DILocalVariable(name: "wide_inttoptr_var"

; CHECK-LABEL: name: dbg_value_null
; CHECK:      DBG_VALUE 0, 0, ![[NULL]], !DIExpression()
; CHECK-NEXT: DBG_VALUE 0, 0, ![[NULL_AS1]], !DIExpression()
; CHECK-NEXT: DBG_VALUE 1234, 0, ![[INTTOPTR]], !DIExpression()
; CHECK-NEXT: DBG_VALUE 4294967295, 0, ![[NARROW]], !DIExpression()
; CHECK-NEXT: DBG_VALUE 1, 0, ![[WIDE]], !DIExpression()

declare void @use(ptr)

define void @dbg_value_null(ptr %p) !dbg !5 {
entry:
    #dbg_value(ptr null, !10, !DIExpression(), !15)
    #dbg_value(ptr addrspace(1) null, !11, !DIExpression(), !15)
    #dbg_value(ptr inttoptr (i64 1234 to ptr), !12, !DIExpression(), !15)
    #dbg_value(ptr inttoptr (i32 -1 to ptr), !13, !DIExpression(), !15)
    #dbg_value(ptr inttoptr (i128 18446744073709551617 to ptr), !14, !DIExpression(), !15)
  call void @use(ptr %p), !dbg !15
  ret void, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "handwritten", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "dbg_value_null", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!9 = !{!10, !11, !12, !13, !14}
!10 = !DILocalVariable(name: "null_var", scope: !5, file: !1, line: 2, type: !8)
!11 = !DILocalVariable(name: "null_as1_var", scope: !5, file: !1, line: 3, type: !8)
!12 = !DILocalVariable(name: "inttoptr_var", scope: !5, file: !1, line: 4, type: !8)
!13 = !DILocalVariable(name: "narrow_inttoptr_var", scope: !5, file: !1, line: 5, type: !8)
!14 = !DILocalVariable(name: "wide_inttoptr_var", scope: !5, file: !1, line: 6, type: !8)
!15 = !DILocation(line: 2, column: 1, scope: !5)
