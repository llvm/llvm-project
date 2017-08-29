; RUN: llc -filetype=obj -O0 < %s -mtriple sparc64-unknown-linux-gnu | llvm-dwarfdump - | FileCheck %s
; The undescribable 128-bit register should be split into two 64-bit registers.
; CHECK: Location description: 90 4a 93 08 90 4b 93 08
;                              DW_OP_reg74 DW_OP_piece 8 DW_OP_reg75 DW_OP_piece 8 ...
source_filename = "/tmp/t.c"
target datalayout = "E-m:e-i64:64-n32:64-S128"
target triple = "sparc64"

@a = common local_unnamed_addr global fp128 0xL00000000000000000000000000000000, align 16, !dbg !0

; Function Attrs: nounwind
define signext i32 @fn1() local_unnamed_addr #0 !dbg !11 {
entry:
  tail call void @llvm.dbg.declare(metadata { fp128, fp128 }* undef, metadata !16, metadata !DIExpression()), !dbg !18
  store fp128 fmul (fp128 undef, fp128 0xL00000000000000003FFF000000000000), fp128* @a, align 16, !dbg !19, !tbaa !20
  tail call void @llvm.dbg.value(metadata fp128 fmul (fp128 undef, fp128 0xL00000000000000003FFF000000000000), metadata !16, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 128)), !dbg !18
  tail call void @llvm.dbg.value(metadata fp128 0xL00000000000000000000000000000000, metadata !16, metadata !DIExpression(DW_OP_LLVM_fragment, 128, 128)), !dbg !18
  ret i32 undef, !dbg !24
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare void @llvm.stackprotector(i8*, i8**) #2

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}

!0 = !DIGlobalVariableExpression(var: !1)
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 6.0.0 (trunk 311908) (llvm/trunk 311915)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "t.c", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "long double", size: 128, encoding: DW_ATE_float)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!11 = distinct !DISubprogram(name: "fn1", scope: !3, file: !3, line: 1, type: !12, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !2, variables: !15)
!12 = !DISubroutineType(types: !13)
!13 = !{!14}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !{!16}
!16 = !DILocalVariable(name: "b", scope: !11, file: !3, line: 1, type: !17)
!17 = !DIBasicType(name: "complex", size: 256, encoding: DW_ATE_complex_float)
!18 = !DILocation(line: 1, column: 45, scope: !11)
!19 = !DILocation(line: 1, column: 51, scope: !11)
!20 = !{!21, !21, i64 0}
!21 = !{!"long double", !22, i64 0}
!22 = !{!"omnipotent char", !23, i64 0}
!23 = !{!"Simple C/C++ TBAA"}
!24 = !DILocation(line: 1, column: 63, scope: !11)
