; RUN: llc -mtriple armv7-linux -relocation-model=ropi -o - %s | FileCheck %s
; RUN: llc -mtriple armv7-linux -relocation-model=ropi-rwpi -o - %s | FileCheck %s

@global = constant i32 -1414812757, align 4, !dbg !0

; CHECK:        .section        .debug_info
; CHECK:        .byte 2 @ DW_AT_location
; DW_OP_addrx 0x0
; CHECK-NEXT:   .byte 161
; CHECK-NEXT:   .byte 0

; CHECK:        .section        .debug_addr
; CHECK:        .long   global

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "global", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 15.0.0 (https://github.com/llvm/llvm-project.git 3948962b454022c2c8de6f67942a9cbd1f0351a0)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "ropi.c", directory: "/tmp")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !6)
!6 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!7 = !{i32 7, !"Dwarf Version", i32 5}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project.git 3948962b454022c2c8de6f67942a9cbd1f0351a0)"}

