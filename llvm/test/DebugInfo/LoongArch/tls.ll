; RUN: llc -O0 -mtriple=loongarch32 -filetype=asm < %s | FileCheck %s -check-prefix=ASM-DTPREL32
; RUN: llc -O0 -mtriple=loongarch64 -filetype=asm < %s | FileCheck %s -check-prefix=ASM-DTPREL64
; RUN: llc -O0 -mtriple=loongarch32 -filetype=obj -o %32.o %s
; RUN: llvm-readobj -r %32.o | FileCheck --check-prefix=OBJ-DTPREL32 %s
; RUN: llc -O0 -mtriple=loongarch64 -filetype=obj -o %64.o %s
; RUN: llvm-readobj -r %64.o | FileCheck --check-prefix=OBJ-DTPREL64 %s

@x = thread_local global i32 5, align 4, !dbg !0

; ASM-DTPREL32: .dtprelword x
; ASM-DTPREL64: .dtpreldword x

; OBJ-DTPREL32: R_LARCH_TLS_DTPREL32 x
; OBJ-DTPREL64: R_LARCH_TLS_DTPREL64 x

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
