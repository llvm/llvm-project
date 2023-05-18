; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   #define __tag1 __attribute__((btf_decl_tag("tag1")))
;   typedef struct { int a; } __s __tag1;
;   typedef unsigned * __u __tag1;
;   __s a;
;   __u u;
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

%struct.__s = type { i32 }

@a = dso_local local_unnamed_addr global %struct.__s zeroinitializer, align 4, !dbg !0
@u = dso_local local_unnamed_addr global ptr null, align 8, !dbg !5

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!17, !18, !19, !20}
!llvm.ident = !{!21}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 4, type: !12, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 14.0.0 (https://github.com/llvm/llvm-project.git 219b26fbcd70273ddfd4ead9387f7c69b7eb4570)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/tmp/home/yhs/work/tests/llvm/btf_tag")
!4 = !{!0, !5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "u", scope: !2, file: !3, line: 5, type: !7, isLocal: false, isDefinition: true)
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "__u", file: !3, line: 3, baseType: !8, annotations: !10)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!9 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!10 = !{!11}
!11 = !{!"btf_decl_tag", !"tag1"}
!12 = !DIDerivedType(tag: DW_TAG_typedef, name: "__s", file: !3, line: 2, baseType: !13, annotations: !10)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !3, line: 2, size: 32, elements: !14)
!14 = !{!15}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !13, file: !3, line: 2, baseType: !16, size: 32)
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !{i32 7, !"Dwarf Version", i32 4}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{i32 1, !"wchar_size", i32 4}
!20 = !{i32 7, !"frame-pointer", i32 2}
!21 = !{!"clang version 14.0.0 (https://github.com/llvm/llvm-project.git 219b26fbcd70273ddfd4ead9387f7c69b7eb4570)"}

; CHECK:             .long   1                               # BTF_KIND_TYPEDEF(id = 1)
; CHECK-NEXT:        .long   134217728                       # 0x8000000
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   5                               # BTF_KIND_DECL_TAG(id = 2)
; CHECK-NEXT:        .long   285212672                       # 0x11000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   4294967295
; CHECK-NEXT:        .long   0                               # BTF_KIND_STRUCT(id = 3)
; CHECK-NEXT:        .long   67108865                        # 0x4000001
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   10
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   0                               # 0x0
; CHECK-NEXT:        .long   12                              # BTF_KIND_INT(id = 4)
; CHECK-NEXT:        .long   16777216                        # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                        # 0x1000020
; CHECK-NEXT:        .long   10                              # BTF_KIND_VAR(id = 5)
; CHECK-NEXT:        .long   234881024                       # 0xe000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   16                              # BTF_KIND_TYPEDEF(id = 6)
; CHECK-NEXT:        .long   134217728                       # 0x8000000
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   5                               # BTF_KIND_DECL_TAG(id = 7)
; CHECK-NEXT:        .long   285212672                       # 0x11000000
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .long   4294967295
; CHECK-NEXT:        .long   0                               # BTF_KIND_PTR(id = 8)
; CHECK-NEXT:        .long   33554432                        # 0x2000000
; CHECK-NEXT:        .long   9
; CHECK-NEXT:        .long   20                              # BTF_KIND_INT(id = 9)
; CHECK-NEXT:        .long   16777216                        # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   32                              # 0x20

; CHECK:        .ascii  "__s"                           # string offset=1
; CHECK:        .ascii  "tag1"                          # string offset=5
; CHECK:        .byte   97                              # string offset=10
; CHECK:        .ascii  "int"                           # string offset=12
; CHECK:        .ascii  "__u"                           # string offset=16
; CHECK:        .ascii  "unsigned int"                  # string offset=20

