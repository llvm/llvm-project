; RUN: llc --filetype=obj --fast-isel=true < %s | llvm-dwarfdump -debug-info - | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @main() !dbg !5 {

  ; CHECK: 0x{{[0-9a-z]+}}: DW_TAG_variable
  ; CHECK-NEXT: DW_AT_location (DW_OP_constu 0xff, DW_OP_dup, DW_OP_constu 0x7, DW_OP_shr, DW_OP_lit0, DW_OP_not, DW_OP_mul, DW_OP_constu 0x8, DW_OP_shl, DW_OP_or, DW_OP_stack_value)
  ; CHECK-NEXT: DW_AT_name ("sext_i8")
  ; CHECK-NEXT: DW_AT_decl_file
  ; CHECK-NEXT: DW_AT_decl_line
  ; CHECK-NEXT: DW_AT_type (0x{{[0-9a-z]+}} "i32")
  tail call void @llvm.dbg.value(metadata i8 -1, metadata !10, metadata !DIExpression(DIOpArg(0, i8), DIOpSExt(i32))), !dbg !15

  ; CHECK: 0x{{[0-9a-z]+}}: DW_TAG_variable
  ; CHECK-NEXT: DW_AT_location (DW_OP_constu 0xff, DW_OP_constu 0xff, DW_OP_and, DW_OP_stack_value)
  ; CHECK-NEXT: DW_AT_name ("zext_i8")
  ; CHECK-NEXT: DW_AT_decl_file
  ; CHECK-NEXT: DW_AT_decl_line
  ; CHECK-NEXT: DW_AT_type (0x{{[0-9a-z]+}} "i32")
  tail call void @llvm.dbg.value(metadata i8 -1, metadata !11, metadata !DIExpression(DIOpArg(0, i8), DIOpZExt(i32))), !dbg !15

  ; CHECK: 0x{{[0-9a-z]+}}: DW_TAG_variable
  ; CHECK-NEXT: DW_AT_location (DW_OP_constu 0xfffffffffffffff6, DW_OP_constu 0xffffffff, DW_OP_and, DW_OP_stack_value)
  ; CHECK-NEXT: DW_AT_name ("trunc_i64")
  ; CHECK-NEXT: DW_AT_decl_file
  ; CHECK-NEXT: DW_AT_decl_line
  ; CHECK-NEXT: DW_AT_type (0x{{[0-9a-z]+}} "i32")
  tail call void @llvm.dbg.value(metadata i64 -10, metadata !12, metadata !DIExpression(DIOpArg(0, i64), DIOpConvert(i32))), !dbg !15

  ; CHECK: 0x{{[0-9a-z]+}}: DW_TAG_variable
  ; CHECK-NEXT: DW_AT_location (DW_OP_constu 0xff, DW_OP_dup, DW_OP_constu 0x7, DW_OP_shr, DW_OP_lit0, DW_OP_not, DW_OP_mul, DW_OP_constu 0x8, DW_OP_shl, DW_OP_or, DW_OP_lit1, DW_OP_plus, DW_OP_stack_value)
  ; CHECK-NEXT: DW_AT_name ("add_const")
  ; CHECK-NEXT: DW_AT_decl_file
  ; CHECK-NEXT: DW_AT_decl_line
  ; CHECK-NEXT: DW_AT_type (0x{{[0-9a-z]+}} "i32")
  tail call void @llvm.dbg.value(metadata i8 -1, metadata !13, metadata !DIExpression(DIOpArg(0, i8), DIOpSExt(i32), DIOpConstant(i32 1), DIOpAdd())), !dbg !15

  ; CHECK: 0x{{[0-9a-z]+}}: DW_TAG_variable
  ; CHECK-NEXT: DW_AT_location (DW_OP_constu 0x2a, DW_OP_stack_value)
  ; CHECK-NEXT: DW_AT_name ("noop_convert")
  ; CHECK-NEXT: DW_AT_decl_file
  ; CHECK-NEXT: DW_AT_decl_line
  ; CHECK-NEXT: DW_AT_type (0x{{[0-9a-z]+}} "i32")
  tail call void @llvm.dbg.value(metadata i32 42, metadata !14, metadata !DIExpression(DIOpArg(0, i32), DIOpConvert(i32))), !dbg !15

  ret void, !dbg !15
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 8}
!3 = !{i32 7}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test", linkageName: "test", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!10, !11, !12, !13}
!9 = !DIBasicType(name: "i32", size: 32, encoding: DW_ATE_signed)
!10 = !DILocalVariable(name: "sext_i8", scope: !5, file: !1, line: 1, type: !9)
!11 = !DILocalVariable(name: "zext_i8", scope: !5, file: !1, line: 2, type: !9)
!12 = !DILocalVariable(name: "trunc_i64", scope: !5, file: !1, line: 3, type: !9)
!13 = !DILocalVariable(name: "add_const", scope: !5, file: !1, line: 4, type: !9)
!14 = !DILocalVariable(name: "noop_convert", scope: !5, file: !1, line: 4, type: !9)
!15 = !DILocation(line: 1, column: 1, scope: !5)
