; RUN: llc -mtriple=bpfel -filetype=asm %s -o - | FileCheck %s
; RUN: llc -mtriple=bpfeb -filetype=asm %s -o - | FileCheck %s
; The fallback trap prototype must follow global types and variables.
; CHECK: BTF_KIND_INT(id = 1)
; CHECK: BTF_KIND_VAR(id = 2)
; CHECK: BTF_KIND_FUNC_PROTO(id = 3)
; CHECK: BTF_KIND_FUNC(id = 4)

@g = global i32 1, !dbg !0
declare !dbg !8 void @__bpf_trap() section ".ksyms"
!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7}
!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 1, type: !4, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, emissionKind: FullDebug, globals: !5)
!3 = !DIFile(filename: "test.c", directory: "/")
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !{!0}
!6 = !{i32 7, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !DISubprogram(name: "__bpf_trap", scope: !3, file: !3, type: !9, spFlags: 0)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
