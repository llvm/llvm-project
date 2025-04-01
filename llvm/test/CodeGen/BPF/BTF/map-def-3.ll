; RUN: llc -mtriple=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mtriple=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
;
; Source code:
;   struct key_type {
;     int a1;
;   };
;   const struct key_type __attribute__((section(".maps"))) hash_map;
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t3.c

%struct.key_type = type { i32 }

@hash_map = dso_local local_unnamed_addr constant %struct.key_type zeroinitializer, section ".maps", align 4, !dbg !0

; CHECK:             .long   1                               # BTF_KIND_INT(id = 1)
; CHECK-NEXT:        .long   16777216                        # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                        # 0x1000020
; CHECK-NEXT:        .long   0                               # BTF_KIND_CONST(id = 2)
; CHECK-NEXT:        .long   167772160                       # 0xa000000
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   5                               # BTF_KIND_STRUCT(id = 3)
; CHECK-NEXT:        .long   67108865                        # 0x4000001
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   14
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   0                               # 0x0
; CHECK-NEXT:        .long   17                              # BTF_KIND_VAR(id = 4)
; CHECK-NEXT:        .long   234881024                       # 0xe000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   26                              # BTF_KIND_DATASEC(id = 5)
; CHECK-NEXT:        .long   251658241                       # 0xf000001
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   hash_map
; CHECK-NEXT:        .long   4

; CHECK:             .ascii  "int"                           # string offset=1
; CHECK:             .ascii  "key_type"                      # string offset=5
; CHECK:             .ascii  "a1"                            # string offset=14
; CHECK:             .ascii  "hash_map"                      # string offset=17
; CHECK:             .ascii  ".maps"                         # string offset=26


!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13}
!llvm.ident = !{!14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "hash_map", scope: !2, file: !3, line: 4, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git 5bd074629f00d4798674b411cf00216f38016483)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "t3.c", directory: "/tmp/home/yhs/tmp1")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !7)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "key_type", file: !3, line: 1, size: 32, elements: !8)
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "a1", scope: !7, file: !3, line: 2, baseType: !10, size: 32)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{i32 7, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 5bd074629f00d4798674b411cf00216f38016483)"}
