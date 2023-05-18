; RUN: llc -filetype=obj -O0 < %s -mtriple sparc64-unknown-linux-gnu | llvm-dwarfdump - --debug-loc | FileCheck %s
; The undescribable 128-bit register should be split into two 64-bit registers.
; CHECK: ({{.*}}, {{.*}}): DW_OP_regx D0, DW_OP_piece 0x8, DW_OP_regx D1, DW_OP_piece 0x8

target datalayout = "E-m:e-i64:64-n32:64-S128"
target triple = "sparc64"

; Function Attrs: nounwind readnone
define void @fn1(fp128 %b) local_unnamed_addr !dbg !7 {
entry:
  tail call void @llvm.dbg.value(metadata fp128 %b, i64 0, metadata !13, metadata !18), !dbg !17
  tail call void @llvm.dbg.value(metadata fp128 0xL00000000000000000000000000000000, i64 0, metadata !13, metadata !19), !dbg !17
  ret void, !dbg !20
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "subreg.c", directory: ".")
!4 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "fn1", scope: !1, file: !1, line: 1, type: !8, unit: !0)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIBasicType(name: "long double", size: 128, encoding: DW_ATE_float)
!13 = !DILocalVariable(name: "a", scope: !7, file: !1, line: 1, type: !14)
!14 = !DIBasicType(name: "complex", size: 256, encoding: DW_ATE_complex_float)
!17 = !DILocation(line: 1, column: 48, scope: !7)
!18 = !DIExpression(DW_OP_LLVM_fragment, 0, 128)
!19 = !DIExpression(DW_OP_LLVM_fragment, 128, 128)
!20 = !DILocation(line: 1, column: 55, scope: !7)
