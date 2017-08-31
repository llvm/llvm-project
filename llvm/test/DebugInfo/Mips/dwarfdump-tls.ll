; RUN: llc -O0 -march=mips -mcpu=mips32r2 -filetype=obj -o=%t-32.o < %s
; RUN: llvm-dwarfdump %t-32.o 2>&1 | FileCheck %s
; RUN: llc -O0 -march=mips64 -mcpu=mips64r2 -filetype=obj -o=%t-64.o < %s
; RUN: llvm-dwarfdump %t-64.o 2>&1 | FileCheck %s

@x = thread_local global i32 5, align 4, !dbg !0

; CHECK-NOT: error: failed to compute relocation: R_MIPS_TLS_DTPREL

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 4.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "tls.c", directory: "/tmp")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
