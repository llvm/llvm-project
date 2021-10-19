; RUN: llc %s -stop-before wasm-nullify-dbg-value-lists -o - | FileCheck %s --check-prefix=BEFORE
; RUN: llc %s -stop-after wasm-nullify-dbg-value-lists -o - | FileCheck %s --check-prefix=AFTER

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; WebAssembly backend does not currently handle DBG_VALUE_LIST instructions
; correctly. In the meantime, they are converted to 'DBG_VALUE $noreg's in
; WebAssemblyNullifyDebugValueLists pass.

; BEFORE: DBG_VALUE_LIST
; AFTER-NOT: DBG_VALUE_LIST
; AFTER: DBG_VALUE $noreg, $noreg
define i32 @dbg_value_list_test() !dbg !6 {
entry:
  %0 = call i32 @foo(), !dbg !9
  %1 = call i32 @foo(), !dbg !10
  %2 = add i32 %0, %1, !dbg !11
  ; This DIArgList operand generates a DBG_VALUE_LIST instruction
  call void @llvm.dbg.value(metadata !DIArgList(i32 %0, i32 %1), metadata !8, metadata !DIExpression()), !dbg !11
  ret i32 %2, !dbg !12
}

declare i32 @foo()
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git ed7aaf832444411ce93aa0443425ce401f5c7a8e)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/home/llvm-project")
!2 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = distinct !DISubprogram(name: "", scope: !1, file: !1, line: 3, type: !7, scopeLine: 3, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!7 = !DISubroutineType(types: !{null})
!8 = !DILocalVariable(name: "i", scope: !6, file: !1, line: 4, type: !2)
!9 = !DILocation(line: 4, column: 11, scope: !6)
!10 = !DILocation(line: 5, column: 11, scope: !6)
!11 = !DILocation(line: 6, column: 3, scope: !6)
!12 = !DILocation(line: 7, column: 1, scope: !6)
