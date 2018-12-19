; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   int * restrict p;
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

@p = common dso_local local_unnamed_addr global i32* null, align 8, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   40
; CHECK-NEXT:        .long   40
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   0                       # BTF_KIND_RESTRICT(id = 1)
; CHECK-NEXT:        .long   184549376               # 0xb000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   0                       # BTF_KIND_PTR(id = 2)
; CHECK-NEXT:        .long   33554432                # 0x2000000
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   1                       # BTF_KIND_INT(id = 3)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                # 0x1000020
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .ascii  "int"                   # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .section        .BTF.ext,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   8                       # FuncInfo
; CHECK-NEXT:        .long   16                      # LineInfo

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "p", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 8.0.0 (trunk 344789) (llvm/trunk 344782)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/home/yhs/tmp")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !7)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{!"clang version 8.0.0 (trunk 344789) (llvm/trunk 344782)"}
