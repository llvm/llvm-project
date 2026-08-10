; RUN: llc -mtriple=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mtriple=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   double a;
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

@a = dso_local local_unnamed_addr global double 0.000000e+00, align 8, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

; CHECK:             .section .BTF,"",@progbits
; CHECK-NEXT:        .short 60319 # 0xeb9f
; CHECK-NEXT:        .byte 1
; CHECK-NEXT:        .byte 0
; CHECK-NEXT:        .long 24
; CHECK-NEXT:        .long 0
; CHECK-NEXT:        .long 52
; CHECK-NEXT:        .long 52
; CHECK-NEXT:        .long 15
; [1] double, size=8 bytes (64 bits)
; CHECK-NEXT:        .long 1 # BTF_KIND_FLOAT(id = 1)
; CHECK-NEXT:        .long 268435456 # 0x10000000
; CHECK-NEXT:        .long 8
; [2] a, type=double (1), global
; CHECK-NEXT:        .long 8 # BTF_KIND_VAR(id = 2)
; CHECK-NEXT:        .long 234881024 # 0xe000000
; CHECK-NEXT:        .long 1
; CHECK-NEXT:        .long 1
; [3] .bss, 1 var, {a, offset=&a, size=8 bytes}
; CHECK-NEXT:        .long 10 # BTF_KIND_DATASEC(id = 3)
; CHECK-NEXT:        .long 251658241 # 0xf000001
; CHECK-NEXT:        .long 0
; CHECK-NEXT:        .long 2
; CHECK-NEXT:        .long a
; CHECK-NEXT:        .long 8
; CHECK-NEXT:        .byte 0 # string offset=0
; CHECK-NEXT:        .ascii "double" # string offset=1
; CHECK-NEXT:        .byte 0
; CHECK-NEXT:        .byte 97 # string offset=8
; CHECK-NEXT:        .byte 0
; CHECK-NEXT:        .ascii ".bss" # string offset=10
; CHECK-NEXT:        .byte 0

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 11.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/home/yhs/tmp")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!7 = !{i32 7, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 11.0.0 "}
