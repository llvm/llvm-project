; RUN: %llc_dwarf -O0 -filetype=obj -o %t.o %s
; RUN: llvm-dwarfdump -debug-info %t.o | FileCheck %s
; REQUIRES: object-emission
target datalayout = "e-m:o-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7-apple-unknown-macho"

; generated from (-triple thumbv7-apple-unknown-macho -Os):
;   void f(_Complex double c) { c = 0; }

; Function Attrs: nounwind readnone
define arm_aapcscc void @f([2 x i64] %c.coerce) #0 !dbg !4 {
entry:
  tail call void @llvm.dbg.declare(metadata { double, double }* undef, metadata !14, metadata !15), !dbg !16
  ; The target has no native double type.
  ; SROA split the complex value into two i64 values.
  ; CHECK: DW_TAG_formal_parameter
  ; CHECK-NEXT:  DW_AT_location [DW_FORM_block1]	(DW_OP_constu 0x0, DW_OP_piece 0x8)
  ; CHECK-NEXT:  DW_AT_name {{.*}} "c"
  tail call void @llvm.dbg.value(metadata i64 0, metadata !14, metadata !17), !dbg !16
  ; Manually removed to disable location list emission:
  ; tail call void @llvm.dbg.value(metadata i64 0, metadata !14, metadata !18), !dbg !16
  ret void, !dbg !19
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10, !11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 (trunk 259998) (llvm/trunk 259999)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{}
!4 = distinct !DISubprogram(name: "f", scope: !5, file: !5, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!5 = !DIFile(filename: "test.c", directory: "/")
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIBasicType(name: "complex", size: 128, align: 64, encoding: DW_ATE_complex_float)
!9 = !{i32 2, !"Dwarf Version", i32 2}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{i32 1, !"min_enum_size", i32 4}
!13 = !{!"clang version 3.9.0 (trunk 259998) (llvm/trunk 259999)"}
!14 = !DILocalVariable(name: "c", arg: 1, scope: !4, file: !5, line: 1, type: !8)
!15 = !DIExpression()
!16 = !DILocation(line: 1, column: 24, scope: !4)
!17 = !DIExpression(DW_OP_LLVM_fragment, 0, 64)
!18 = !DIExpression(DW_OP_LLVM_fragment, 64, 64)
!19 = !DILocation(line: 1, column: 36, scope: !4)
