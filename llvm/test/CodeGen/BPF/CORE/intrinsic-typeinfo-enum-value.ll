; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck %s
; RUN: llc -march=bpfel -mattr=+alu32 -filetype=asm -o - %s | FileCheck %s
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
;   clang -target bpf -S -O2 -g -emit-llvm t1.c

@0 = private unnamed_addr constant [10 x i8] c"VAL1:-100\00", align 1
@1 = private unnamed_addr constant [16 x i8] c"VAL2:4294934528\00", align 1
@2 = private unnamed_addr constant [18 x i8] c"VAL10:-2147483648\00", align 1

; Function Attrs: nounwind readnone
define dso_local i32 @test() local_unnamed_addr #0 !dbg !18 {
entry:
  %0 = tail call i64 @llvm.bpf.preserve.enum.value(i32 0, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @0, i64 0, i64 0), i64 0), !dbg !23, !llvm.preserve.access.index !3
  %1 = tail call i64 @llvm.bpf.preserve.enum.value(i32 1, i8* getelementptr inbounds ([16 x i8], [16 x i8]* @1, i64 0, i64 0), i64 1), !dbg !24, !llvm.preserve.access.index !3
  %add = add i64 %1, %0, !dbg !25
  %2 = tail call i64 @llvm.bpf.preserve.enum.value(i32 2, i8* getelementptr inbounds ([18 x i8], [18 x i8]* @2, i64 0, i64 0), i64 1), !dbg !26, !llvm.preserve.access.index !13
  %add1 = add i64 %add, %2, !dbg !27
  %conv = trunc i64 %add1 to i32, !dbg !23
  ret i32 %conv, !dbg !28
}

; CHECK:             r{{[0-9]+}} = 1 ll
; CHECK:             r{{[0-9]+}} = 4294934528 ll
; CHECK:             r{{[0-9]+}} = -2147483648 ll
; CHECK:             exit

; CHECK:             .long   16                              # BTF_KIND_ENUM(id = 4)
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
declare i64 @llvm.bpf.preserve.enum.value(i32, i8*, i64) #1

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14, !15, !16}
!llvm.ident = !{!17}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0 (https://github.com/llvm/llvm-project.git d8b1394a0f4bbf57c254f69f8d3aa5381a89b5cd)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !12, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t1.c", directory: "/tmp/home/yhs/tmp1")
!2 = !{!3, !8}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "AA", file: !1, line: 1, baseType: !4, size: 64, elements: !5)
!4 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!5 = !{!6, !7}
!6 = !DIEnumerator(name: "VAL1", value: -100)
!7 = !DIEnumerator(name: "VAL2", value: 4294934528)
!8 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !1, line: 2, baseType: !9, size: 64, elements: !10)
!9 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!10 = !{!11}
!11 = !DIEnumerator(name: "VAL10", value: 18446744071562067968, isUnsigned: true)
!12 = !{!13}
!13 = !DIDerivedType(tag: DW_TAG_typedef, name: "__BB", file: !1, line: 2, baseType: !8)
!14 = !{i32 7, !"Dwarf Version", i32 4}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 4}
!17 = !{!"clang version 12.0.0 (https://github.com/llvm/llvm-project.git d8b1394a0f4bbf57c254f69f8d3aa5381a89b5cd)"}
!18 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 3, type: !19, scopeLine: 3, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !22)
!19 = !DISubroutineType(types: !20)
!20 = !{!21}
!21 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!22 = !{}
!23 = !DILocation(line: 4, column: 10, scope: !18)
!24 = !DILocation(line: 5, column: 10, scope: !18)
!25 = !DILocation(line: 4, column: 61, scope: !18)
!26 = !DILocation(line: 6, column: 10, scope: !18)
!27 = !DILocation(line: 5, column: 61, scope: !18)
!28 = !DILocation(line: 4, column: 3, scope: !18)
