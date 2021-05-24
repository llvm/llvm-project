; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck %s
;
; Source code:
;   #pragma clang attribute push (__attribute__((preserve_access_index)), apply_to = record)
;   typedef struct {
;     int a;
;   } __t;
;   #pragma clang attribute pop
;
;   int test(__t *arg) { return arg->a; }
; Compiler flag to generate IR:
;   clang -target bpf -S -O2 -g -emit-llvm -Xclang -disable-llvm-passes test.c

target triple = "bpf"

%struct.__t = type { i32 }

; Function Attrs: nounwind readonly
define dso_local i32 @test(%struct.__t* readonly %arg) local_unnamed_addr #0 !dbg !13 {
entry:
  call void @llvm.dbg.value(metadata %struct.__t* %arg, metadata !18, metadata !DIExpression()), !dbg !19
  %0 = tail call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.__ts(%struct.__t* %arg, i32 0, i32 0), !dbg !20, !llvm.preserve.access.index !4
  %1 = load i32, i32* %0, align 4, !dbg !20, !tbaa !21
  ret i32 %1, !dbg !26
}

; CHECK:             .long   1                       # BTF_KIND_TYPEDEF(id = 2)
; CHECK-NEXT:        .long   134217728               # 0x8000000
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   0                       # BTF_KIND_STRUCT(id = 3)
; CHECK-NEXT:        .long   67108865                # 0x4000001
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   0                       # 0x0
;
; CHECK:             .ascii  "__t"                   # string offset=1
; CHECK:             .byte   97                      # string offset=5
; CHECK:             .ascii  ".text"                 # string offset=20
; CHECK:             .ascii  "0:0"                   # string offset=26
;
; CHECK:             .long   16                      # FieldReloc
; CHECK-NEXT:        .long   20                      # Field reloc section string offset=20
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   26
; CHECK-NEXT:        .long   0

; Function Attrs: nounwind readnone
declare i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.__ts(%struct.__t*, i32, i32) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readnone speculatable}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git 5125d1c934efa69ffc1902ce3b8f2f288653a92f)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/core_bug")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_typedef, name: "__t", file: !1, line: 4, baseType: !5)
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !1, line: 2, size: 32, elements: !6)
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !5, file: !1, line: 3, baseType: !8, size: 32)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{i32 7, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 5125d1c934efa69ffc1902ce3b8f2f288653a92f)"}
!13 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 7, type: !14, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !17)
!14 = !DISubroutineType(types: !15)
!15 = !{!8, !16}
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64)
!17 = !{!18}
!18 = !DILocalVariable(name: "arg", arg: 1, scope: !13, file: !1, line: 7, type: !16)
!19 = !DILocation(line: 0, scope: !13)
!20 = !DILocation(line: 7, column: 34, scope: !13)
!21 = !{!22, !23, i64 0}
!22 = !{!"", !23, i64 0}
!23 = !{!"int", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C/C++ TBAA"}
!26 = !DILocation(line: 7, column: 22, scope: !13)
