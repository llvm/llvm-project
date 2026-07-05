; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck %s
;
; Source:
;   enum AA { VAL = 100 };
;   typedef int (*func_t)(void);
;   struct s2 { int a[10]; };
;   int test() {
;     return __builtin_preserve_type_info(*(func_t *)0, 2) +
;            __builtin_preserve_type_info(*(struct s2 *)0, 2) +
;            __builtin_preserve_type_info(*(enum AA *)0, 2);
;   }
; Compiler flag to generate IR:
;   clang -target bpf -S -O2 -g -emit-llvm -Xclang -disable-llvm-passes t1.c

source_filename = "t1.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "bpf"

; Function Attrs: nounwind
define dso_local i32 @test() #0 !dbg !18 {
  %1 = call i32 @llvm.bpf.preserve.type.info(i32 0, i64 2), !dbg !20, !llvm.preserve.access.index !8
  %2 = call i32 @llvm.bpf.preserve.type.info(i32 1, i64 2), !dbg !21, !llvm.preserve.access.index !22
  %3 = add i32 %1, %2, !dbg !28
  %4 = call i32 @llvm.bpf.preserve.type.info(i32 2, i64 2), !dbg !29, !llvm.preserve.access.index !3
  %5 = add i32 %3, %4, !dbg !30
  ret i32 %5, !dbg !31
}

; CHECK:             r{{[0-9]+}} = 1
; CHECK:             r{{[0-9]+}} = 1
; CHECK:             r{{[0-9]+}} = 1
; CHECK:             exit

; CHECK:             .long   16                              # BTF_KIND_TYPEDEF(id = 4)
; CHECK:             .long   40                              # BTF_KIND_STRUCT(id = 7)
; CHECK:             .long   65                              # BTF_KIND_ENUM(id = 10)

; CHECK:             .ascii  ".text"                         # string offset=10
; CHECK:             .ascii  "func_t"                        # string offset=16
; CHECK:             .byte   48                              # string offset=23
; CHECK:             .ascii  "s2"                            # string offset=40
; CHECK:             .ascii  "AA"                            # string offset=65

; CHECK:             .long   16                              # FieldReloc
; CHECK-NEXT:        .long   10                              # Field reloc section string offset=10
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   23
; CHECK-NEXT:        .long   12
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   23
; CHECK-NEXT:        .long   12
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   10
; CHECK-NEXT:        .long   23
; CHECK-NEXT:        .long   12

; Function Attrs: nounwind readnone
declare i32 @llvm.bpf.preserve.type.info(i32, i64) #1

attributes #0 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14, !15, !16}
!llvm.ident = !{!17}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 15.0.0 (https://github.com/llvm/llvm-project.git 3d974661fd15612259d37f603ddf21df7ee0e428)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !7, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t1.c", directory: "/tmp/tmp1", checksumkind: CSK_MD5, checksum: "53350e4a8003565f949c897f1fce8567")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "AA", file: !1, line: 1, baseType: !4, size: 32, elements: !5)
!4 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!5 = !{!6}
!6 = !DIEnumerator(name: "VAL", value: 100)
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_typedef, name: "func_t", file: !1, line: 2, baseType: !9)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{i32 7, !"Dwarf Version", i32 5}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 4}
!16 = !{i32 7, !"frame-pointer", i32 2}
!17 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project.git 3d974661fd15612259d37f603ddf21df7ee0e428)"}
!18 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 4, type: !10, scopeLine: 4, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !19)
!19 = !{}
!20 = !DILocation(line: 5, column: 10, scope: !18)
!21 = !DILocation(line: 6, column: 10, scope: !18)
!22 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s2", file: !1, line: 3, size: 320, elements: !23)
!23 = !{!24}
!24 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !22, file: !1, line: 3, baseType: !25, size: 320)
!25 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 320, elements: !26)
!26 = !{!27}
!27 = !DISubrange(count: 10)
!28 = !DILocation(line: 5, column: 56, scope: !18)
!29 = !DILocation(line: 7, column: 10, scope: !18)
!30 = !DILocation(line: 6, column: 59, scope: !18)
!31 = !DILocation(line: 5, column: 3, scope: !18)
