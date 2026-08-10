; RUN: llc -mtriple=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mtriple=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   typedef unsigned _int;
;   typedef _int __int;
;   __int a[10];
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

@a = common dso_local local_unnamed_addr global [10 x i32] zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14}
!llvm.ident = !{!15}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   80
; CHECK-NEXT:        .long   80
; CHECK-NEXT:        .long   45
; CHECK-NEXT:        .long   1                       # BTF_KIND_TYPEDEF(id = 1)
; CHECK-NEXT:        .long   134217728               # 0x8000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   7                       # BTF_KIND_TYPEDEF(id = 2)
; CHECK-NEXT:        .long   134217728               # 0x8000000
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   12                      # BTF_KIND_INT(id = 3)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   32                      # 0x20
; CHECK-NEXT:        .long   0                       # BTF_KIND_ARRAY(id = 4)
; CHECK-NEXT:        .long   50331648                # 0x3000000
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   10
; CHECK-NEXT:        .long   25                      # BTF_KIND_INT(id = 5)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   32                      # 0x20
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .ascii  "__int"                 # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "_int"                  # string offset=7
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "unsigned int"          # string offset=12
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "__ARRAY_SIZE_TYPE__"   # string offset=25
; CHECK-NEXT:        .byte   0

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 3, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 8.0.0 (trunk 345296) (llvm/trunk 345297)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/home/yhs/tmp")
!4 = !{}
!5 = !{!0}
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 320, elements: !10)
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "__int", file: !3, line: 2, baseType: !8)
!8 = !DIDerivedType(tag: DW_TAG_typedef, name: "_int", file: !3, line: 1, baseType: !9)
!9 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!10 = !{!11}
!11 = !DISubrange(count: 10)
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{!"clang version 8.0.0 (trunk 345296) (llvm/trunk 345297)"}
