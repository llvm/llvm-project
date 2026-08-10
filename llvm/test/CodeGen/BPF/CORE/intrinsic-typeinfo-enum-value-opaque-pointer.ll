; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck %s
;
; Source:
;   enum AA { VAL1 = -100, VAL2 = 0xffff8000 };
;   typedef enum { VAL10 = 0xffffFFFF80000000 } __BB;
;   int test() {
;     return __builtin_preserve_enum_value(*(enum AA *)VAL1, 0) +
;            __builtin_preserve_enum_value(*(enum AA *)VAL2, 1) +
;            __builtin_preserve_enum_value(*(__BB *)VAL10, 1);
;   }
; Compiler flag to generate IR:
;   clang -target bpf -S -O2 -g -emit-llvm -Xclang -disable-llvm-passes t1.c

target triple = "bpf"

@0 = private unnamed_addr constant [10 x i8] c"VAL1:-100\00", align 1
@1 = private unnamed_addr constant [16 x i8] c"VAL2:4294934528\00", align 1
@2 = private unnamed_addr constant [18 x i8] c"VAL10:-2147483648\00", align 1

; Function Attrs: nounwind
define dso_local i32 @test() #0 !dbg !19 {
entry:
  %0 = call i64 @llvm.bpf.preserve.enum.value(i32 0, ptr @0, i64 0), !dbg !24, !llvm.preserve.access.index !3
  %1 = call i64 @llvm.bpf.preserve.enum.value(i32 1, ptr @1, i64 1), !dbg !25, !llvm.preserve.access.index !3
  %add = add i64 %0, %1, !dbg !26
  %2 = call i64 @llvm.bpf.preserve.enum.value(i32 2, ptr @2, i64 1), !dbg !27, !llvm.preserve.access.index !13
  %add1 = add i64 %add, %2, !dbg !28
  %conv = trunc i64 %add1 to i32, !dbg !24
  ret i32 %conv, !dbg !29
}

; CHECK:             r{{[0-9]+}} = 1 ll
; CHECK:             r{{[0-9]+}} = 4294934528 ll
; CHECK:             r{{[0-9]+}} = -2147483648 ll
; CHECK:             exit

; CHECK:             .long   16                              # BTF_KIND_ENUM64(id = 4)
; CHECK:             .long   57                              # BTF_KIND_TYPEDEF(id = 5)

; CHECK:             .ascii  ".text"                         # string offset=10
; CHECK:             .ascii  "AA"                            # string offset=16
; CHECK:             .byte   48                              # string offset=29
; CHECK:             .byte   49                              # string offset=55
; CHECK:             .ascii  "__BB"                          # string offset=57

; CHECK:             .long   16                              # FieldReloc
; CHECK-NEXT:        .long   10                              # Field reloc section string offset=10
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   29
; CHECK-NEXT:        .long   10
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   55
; CHECK-NEXT:        .long   11
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   29
; CHECK-NEXT:        .long   11

; Function Attrs: nounwind readnone
declare i64 @llvm.bpf.preserve.enum.value(i32, ptr, i64) #1

attributes #0 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14, !15, !16, !17}
!llvm.ident = !{!18}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 15.0.0 (https://github.com/llvm/llvm-project.git 1218d7e1cf1284666cd7403ea021e40b3b40e92b)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !12, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t1.c", directory: "/tmp/home/yhs/tmp1", checksumkind: CSK_MD5, checksum: "e1a546573a450dae0abedfbf6bebcba9")
!2 = !{!3, !8}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "AA", file: !1, line: 1, baseType: !4, size: 64, elements: !5)
!4 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!5 = !{!6, !7}
!6 = !DIEnumerator(name: "VAL1", value: -100)
!7 = !DIEnumerator(name: "VAL2", value: 4294934528)
!8 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !1, line: 2, baseType: !9, size: 64, elements: !10)
!9 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!10 = !{!11}
!11 = !DIEnumerator(name: "VAL10", value: 18446744071562067968, isUnsigned: true)
!12 = !{!13}
!13 = !DIDerivedType(tag: DW_TAG_typedef, name: "__BB", file: !1, line: 2, baseType: !8)
!14 = !{i32 7, !"Dwarf Version", i32 5}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 4}
!17 = !{i32 7, !"frame-pointer", i32 2}
!18 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project.git 1218d7e1cf1284666cd7403ea021e40b3b40e92b)"}
!19 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 3, type: !20, scopeLine: 3, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !23)
!20 = !DISubroutineType(types: !21)
!21 = !{!22}
!22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!23 = !{}
!24 = !DILocation(line: 4, column: 10, scope: !19)
!25 = !DILocation(line: 5, column: 10, scope: !19)
!26 = !DILocation(line: 4, column: 61, scope: !19)
!27 = !DILocation(line: 6, column: 10, scope: !19)
!28 = !DILocation(line: 5, column: 61, scope: !19)
!29 = !DILocation(line: 4, column: 3, scope: !19)
