;; CodeView has no way to name a symbol in a local variable's location: a
;; definition range holds a single register, and the fallback for anything else
;; only covers immediates. Check that a variable holding the address of a
;; global keeps the register location CodeView can consume rather than being
;; described by a symbol and so marked optimized out.

; RUN: llc -O2 -mtriple=x86_64-pc-windows-msvc -stop-after=finalize-isel < %s \
; RUN:   | FileCheck %s --check-prefix=MIR --implicit-check-not='DBG_VALUE @g'
; RUN: llc -O2 -mtriple=x86_64-pc-windows-msvc -filetype=obj < %s \
; RUN:   | llvm-readobj --codeview - | FileCheck %s --check-prefix=CODEVIEW \
; RUN:       --implicit-check-not=IsOptimizedOut

@g = global i64 0, align 8

; MIR-LABEL: name: codeview_global
; MIR: DBG_INSTR_REF ![[#]], !DIExpression(DW_OP_LLVM_arg, 0)
;
; CODEVIEW:      VarName: p
; CODEVIEW-NEXT: }
; CODEVIEW-NEXT: DefRangeRegisterSym {
; CODEVIEW:        Register: R{{[A-Z0-9]+}}
define void @codeview_global() !dbg !6 {
entry:
    #dbg_value(ptr @g, !9, !DIExpression(), !10)
  %box = tail call ptr @alloc(), !dbg !10
  store ptr @g, ptr %box, align 8, !dbg !10
  tail call void @sink(ptr %box), !dbg !10
  ret void, !dbg !10
}

declare void @sink(ptr)
declare ptr @alloc()

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 2, !"CodeView", i32 1}
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang", isOptimized: true, emissionKind: FullDebug)
!3 = !DIFile(filename: "t.c", directory: "/")
!4 = !DISubroutineType(types: !{null})
!5 = !DIBasicType(name: "long long", size: 64, encoding: DW_ATE_signed)
!6 = distinct !DISubprogram(name: "codeview_global", scope: !3, file: !3, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!8 = distinct !DILexicalBlock(scope: !6, file: !3, line: 2, column: 1)
!9 = !DILocalVariable(name: "p", scope: !8, file: !3, line: 2, type: !7)
!10 = !DILocation(line: 2, column: 1, scope: !8)
