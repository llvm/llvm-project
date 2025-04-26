; RUN: llc -O0 -mtriple=mips-elf -mcpu=mips32r2 -filetype=asm < %s | FileCheck %s -check-prefix=CHECK-WORD
; RUN: llc -O0 -mtriple=mips64-elf -mcpu=mips64r2 -filetype=asm < %s | FileCheck %s -check-prefix=CHECK-DWORD

@x = thread_local global i32 5, align 4, !dbg !0

; CHECK-WORD: .dtprelword x+32768
; CHECK-DWORD: .dtpreldword x+32768

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 4.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "tls.c", directory: "/tmp")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 4.0.0"}
