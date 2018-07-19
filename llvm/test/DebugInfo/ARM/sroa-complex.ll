; RUN: opt -sroa -S -o - %s | FileCheck %s
; REQUIRES: object-emission
target datalayout = "e-m:o-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7-apple-unknown-macho"

; generated from (-triple thumbv7-apple-unknown-macho):
;   void f(_Complex double c) { c = 0; }

; Function Attrs: nounwind
define arm_aapcscc void @f([2 x i64] %c.coerce) #0 !dbg !4 {
entry:
  %c = alloca { double, double }, align 8
  %0 = bitcast { double, double }* %c to [2 x i64]*
  store [2 x i64] %c.coerce, [2 x i64]* %0, align 8
  call void @llvm.dbg.declare(metadata { double, double }* %c, metadata !14, metadata !15), !dbg !16
  %c.realp = getelementptr inbounds { double, double }, { double, double }* %c, i32 0, i32 0, !dbg !17
  %c.imagp = getelementptr inbounds { double, double }, { double, double }* %c, i32 0, i32 1, !dbg !17
  store double 0.000000e+00, double* %c.realp, align 8, !dbg !17
  ; SROA will split the complex double into two i64 values, because there is
  ; no native double data type available.
  ; Test that debug info for both values survives:
  ; CHECK: call void @llvm.dbg.value(metadata i64 0,
  ; CHECK-SAME:                      metadata ![[C:[^,]*]],
  ; CHECK-SAME:                      metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64))
  store double 0.000000e+00, double* %c.imagp, align 8, !dbg !17
  ; CHECK: call void @llvm.dbg.value(metadata i64 0,
  ; CHECK-SAME:                      metadata ![[C]],
  ; CHECK-SAME:                      metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64))
  ret void, !dbg !18
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

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
!17 = !DILocation(line: 1, column: 31, scope: !4)
!18 = !DILocation(line: 1, column: 36, scope: !4)
