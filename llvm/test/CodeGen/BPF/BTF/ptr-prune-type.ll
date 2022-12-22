; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   struct t {
;     int a;
;   };
;   struct t2 {
;     struct t *f1;
;   };
;   struct t2 __attribute__((section("prune_types"))) g;
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

%struct.t2 = type { ptr }
%struct.t = type { i32 }

@g = dso_local local_unnamed_addr global %struct.t2 zeroinitializer, section "prune_types", align 8, !dbg !0

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   88
; CHECK-NEXT:        .long   88
; CHECK-NEXT:        .long   23
; CHECK-NEXT:        .long   1                       # BTF_KIND_STRUCT(id = 1)
; CHECK-NEXT:        .long   67108865                # 0x4000001
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   0                       # 0x0
; CHECK-NEXT:        .long   0                       # BTF_KIND_PTR(id = 2)
; CHECK-NEXT:        .long   33554432                # 0x2000000
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   7                       # BTF_KIND_VAR(id = 3)
; CHECK-NEXT:        .long   234881024               # 0xe000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   9                       # BTF_KIND_DATASEC(id = 4)
; CHECK-NEXT:        .long   251658241               # 0xf000001
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   g
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   21                      # BTF_KIND_FWD(id = 5)
; CHECK-NEXT:        .long   117440512               # 0x7000000
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .ascii  "t2"                    # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "f1"                    # string offset=4
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   103                     # string offset=7
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "prune_types"           # string offset=9
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   116                     # string offset=21
; CHECK-NEXT:        .byte   0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!14, !15, !16}
!llvm.ident = !{!17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 7, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (trunk 364157) (llvm/trunk 364156)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/tmp/home/yhs/work/tests/llvm")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t2", file: !3, line: 4, size: 64, elements: !7)
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "f1", scope: !6, file: !3, line: 5, baseType: !9, size: 64)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t", file: !3, line: 1, size: 32, elements: !11)
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !10, file: !3, line: 2, baseType: !13, size: 32)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{i32 2, !"Dwarf Version", i32 4}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 4}
!17 = !{!"clang version 9.0.0 (trunk 364157) (llvm/trunk 364156)"}
