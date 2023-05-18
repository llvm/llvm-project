; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck %s

; Source code:
;   struct s { int i; } __attribute__((aligned(16)));
;   struct s a;          // .bss
;   struct s b = { 0 };  // .bss
;   struct s c = { 1 };  // .data
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

%struct.s = type { i32, [12 x i8] }

@a = dso_local local_unnamed_addr global %struct.s zeroinitializer, align 16, !dbg !11
@b = dso_local local_unnamed_addr global %struct.s { i32 0, [12 x i8] undef }, align 16, !dbg !0
@c = dso_local local_unnamed_addr global %struct.s { i32 1, [12 x i8] undef }, align 16, !dbg !5

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13, !14, !15, !16}
!llvm.ident = !{!17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 3, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 16.0.0 (https://github.com/llvm/llvm-project.git 3191e8e19f1a7007ddd0e55cee60a51a058c99f5)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/home/eddy/work/tmp", checksumkind: CSK_MD5, checksum: "b9d0621d30812c09bd3c6894f89ff5e4")
!4 = !{!0, !5, !11}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "c", scope: !2, file: !3, line: 4, type: !7, isLocal: false, isDefinition: true)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s", file: !3, line: 1, size: 128, align: 128, elements: !8)
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !7, file: !3, line: 1, baseType: !10, size: 32)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIGlobalVariableExpression(var: !12, expr: !DIExpression())
!12 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 2, type: !7, isLocal: false, isDefinition: true)
!13 = !{i32 7, !"Dwarf Version", i32 5}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 4}
!16 = !{i32 7, !"frame-pointer", i32 2}
!17 = !{!"clang version 16.0.0 (https://github.com/llvm/llvm-project.git 3191e8e19f1a7007ddd0e55cee60a51a058c99f5)"}

; CHECK:         .section        .BTF,"",@progbits
; CHECK-NEXT:    .short  60319                           # 0xeb9f
; CHECK-NEXT:    .byte   1
; CHECK-NEXT:    .byte   0
; CHECK-NEXT:    .long   24
; CHECK-NEXT:    .long   0
; CHECK-NEXT:    .long   148
; CHECK-NEXT:    .long   148
; CHECK-NEXT:    .long   26
; CHECK-NEXT:    .long   1                               # BTF_KIND_STRUCT(id = 1)
; CHECK-NEXT:    .long   67108865                        # 0x4000001
; CHECK-NEXT:    .long   16
; CHECK-NEXT:    .long   3
; CHECK-NEXT:    .long   2
; CHECK-NEXT:    .long   0                               # 0x0
; CHECK-NEXT:    .long   5                               # BTF_KIND_INT(id = 2)
; CHECK-NEXT:    .long   16777216                        # 0x1000000
; CHECK-NEXT:    .long   4
; CHECK-NEXT:    .long   16777248                        # 0x1000020
; CHECK-NEXT:    .long   9                               # BTF_KIND_VAR(id = 3)
; CHECK-NEXT:    .long   234881024                       # 0xe000000
; CHECK-NEXT:    .long   1
; CHECK-NEXT:    .long   1
; CHECK-NEXT:    .long   11                              # BTF_KIND_VAR(id = 4)
; CHECK-NEXT:    .long   234881024                       # 0xe000000
; CHECK-NEXT:    .long   1
; CHECK-NEXT:    .long   1
; CHECK-NEXT:    .long   13                              # BTF_KIND_VAR(id = 5)
; CHECK-NEXT:    .long   234881024                       # 0xe000000
; CHECK-NEXT:    .long   1
; CHECK-NEXT:    .long   1
; CHECK-NEXT:    .long   15                              # BTF_KIND_DATASEC(id = 6)
; CHECK-NEXT:    .long   251658242                       # 0xf000002
; CHECK-NEXT:    .long   0
; CHECK-NEXT:    .long   3
; CHECK-NEXT:    .long   a
; CHECK-NEXT:    .long   16
; CHECK-NEXT:    .long   4
; CHECK-NEXT:    .long   b
; CHECK-NEXT:    .long   16
; CHECK-NEXT:    .long   20                              # BTF_KIND_DATASEC(id = 7)
; CHECK-NEXT:    .long   251658241                       # 0xf000001
; CHECK-NEXT:    .long   0
; CHECK-NEXT:    .long   5
; CHECK-NEXT:    .long   c
; CHECK-NEXT:    .long   16
; CHECK-NEXT:    .byte   0                               # string offset=0
; CHECK-NEXT:    .byte   115                             # string offset=1
; CHECK-NEXT:    .byte   0
; CHECK-NEXT:    .byte   105                             # string offset=3
; CHECK-NEXT:    .byte   0
; CHECK-NEXT:    .ascii  "int"                           # string offset=5
; CHECK-NEXT:    .byte   0
; CHECK-NEXT:    .byte   97                              # string offset=9
; CHECK-NEXT:    .byte   0
; CHECK-NEXT:    .byte   98                              # string offset=11
; CHECK-NEXT:    .byte   0
; CHECK-NEXT:    .byte   99                              # string offset=13
; CHECK-NEXT:    .byte   0
; CHECK-NEXT:    .ascii  ".bss"                          # string offset=15
; CHECK-NEXT:    .byte   0
; CHECK-NEXT:    .ascii  ".data"                         # string offset=20
; CHECK-NEXT:    .byte   0
