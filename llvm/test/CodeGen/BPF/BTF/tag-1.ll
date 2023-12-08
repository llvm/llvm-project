; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   #define __tag1 __attribute__((btf_decl_tag("tag1")))
;   #define __tag2 __attribute__((btf_decl_tag("tag2")))
;   struct t1 {
;     int a1;
;     int a2 __tag1 __tag2;
;   } __tag1 __tag2;
;   struct t1 g1 __tag1 __tag2;
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

%struct.t1 = type { i32, i32 }

@g1 = dso_local local_unnamed_addr global %struct.t1 zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!14, !15, !16, !17}
!llvm.ident = !{!18}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g1", scope: !2, file: !3, line: 7, type: !6, isLocal: false, isDefinition: true, annotations: !11)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 13.0.0 (https://github.com/llvm/llvm-project.git 825661b8e31d0b29d78178df1e518949dfec9f9a)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/tmp/home/yhs/work/tests/llvm/btf_tag")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t1", file: !3, line: 3, size: 64, elements: !7, annotations: !11)
!7 = !{!8, !10}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "a1", scope: !6, file: !3, line: 4, baseType: !9, size: 32)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "a2", scope: !6, file: !3, line: 5, baseType: !9, size: 32, offset: 32, annotations: !11)
!11 = !{!12, !13}
!12 = !{!"btf_decl_tag", !"tag1"}
!13 = !{!"btf_decl_tag", !"tag2"}
!14 = !{i32 7, !"Dwarf Version", i32 4}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 4}
!17 = !{i32 7, !"frame-pointer", i32 2}
!18 = !{!"clang version 13.0.0 (https://github.com/llvm/llvm-project.git 825661b8e31d0b29d78178df1e518949dfec9f9a)"}

; CHECK:             .long   1                               # BTF_KIND_STRUCT(id = 1)
; CHECK-NEXT:        .long   67108866                        # 0x4000002
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   0                               # 0x0
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   32                              # 0x20
; CHECK-NEXT:        .long   10                              # BTF_KIND_DECL_TAG(id = 2)
; CHECK-NEXT:        .long   285212672                       # 0x11000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   4294967295
; CHECK-NEXT:        .long   15                              # BTF_KIND_DECL_TAG(id = 3)
; CHECK-NEXT:        .long   285212672                       # 0x11000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   4294967295
; CHECK-NEXT:        .long   20                              # BTF_KIND_INT(id = 4)
; CHECK-NEXT:        .long   16777216                        # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                        # 0x1000020
; CHECK-NEXT:        .long   10                              # BTF_KIND_DECL_TAG(id = 5)
; CHECK-NEXT:        .long   285212672                       # 0x11000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   15                              # BTF_KIND_DECL_TAG(id = 6)
; CHECK-NEXT:        .long   285212672                       # 0x11000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   24                              # BTF_KIND_VAR(id = 7)
; CHECK-NEXT:        .long   234881024                       # 0xe000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   10                              # BTF_KIND_DECL_TAG(id = 8)
; CHECK-NEXT:        .long   285212672                       # 0x11000000
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   4294967295
; CHECK-NEXT:        .long   15                              # BTF_KIND_DECL_TAG(id = 9)
; CHECK-NEXT:        .long   285212672                       # 0x11000000
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   4294967295

; CHECK:             .ascii  "t1"                            # string offset=1
; CHECK:             .ascii  "a1"                            # string offset=4
; CHECK:             .ascii  "a2"                            # string offset=7
; CHECK:             .ascii  "tag1"                          # string offset=10
; CHECK:             .ascii  "tag2"                          # string offset=15
; CHECK:             .ascii  "int"                           # string offset=20
; CHECK:             .ascii  "g1"                            # string offset=24
