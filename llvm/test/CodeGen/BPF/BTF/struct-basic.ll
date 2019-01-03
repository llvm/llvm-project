; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   struct t1 {char m1; int n1;} a;
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

%struct.t1 = type { i8, i32 }

@a = common dso_local local_unnamed_addr global %struct.t1 zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14}
!llvm.ident = !{!15}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   68
; CHECK-NEXT:        .long   68
; CHECK-NEXT:        .long   19
; CHECK-NEXT:        .long   1                       # BTF_KIND_STRUCT(id = 1)
; CHECK-NEXT:        .long   67108866                # 0x4000002
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   32
; CHECK-NEXT:        .long   10                      # BTF_KIND_INT(id = 2)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   16777224                # 0x1000008
; CHECK-NEXT:        .long   15                      # BTF_KIND_INT(id = 3)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                # 0x1000020
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .ascii  "t1"                    # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "m1"                    # string offset=4
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "n1"                    # string offset=7
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "char"                  # string offset=10
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "int"                   # string offset=15
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
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 8.0.0 (trunk 345296) (llvm/trunk 345297)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/home/yhs/tmp")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t1", file: !3, line: 1, size: 64, elements: !7)
!7 = !{!8, !10}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "m1", scope: !6, file: !3, line: 1, baseType: !9, size: 8)
!9 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "n1", scope: !6, file: !3, line: 1, baseType: !11, size: 32, offset: 32)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{!"clang version 8.0.0 (trunk 345296) (llvm/trunk 345297)"}
