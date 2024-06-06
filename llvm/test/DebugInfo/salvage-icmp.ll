; RUN: opt %s -passes=dce -S | FileCheck %s

; Tests the results of salvaging variadic dbg.values that use the same SSA value
; multiple times.

; CHECK: call void @llvm.dbg.value(metadata i32 %a,
; CHECK-SAME: ![[VAR_C:[0-9]+]],
; CHECK-SAME: !DIExpression(DW_OP_lit0, DW_OP_ne, DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_lit0, DW_OP_eq, DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_constu, 1, DW_OP_gt, DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_consts, 18446744073709551615, DW_OP_gt, DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_constu, 2, DW_OP_ge, DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_consts, 18446744073709551614, DW_OP_ge, DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_constu, 3, DW_OP_lt, DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_consts, 18446744073709551613, DW_OP_lt, DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_constu, 4, DW_OP_le, DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_consts, 18446744073709551612, DW_OP_le, DW_OP_stack_value))

; CHECK: call void @llvm.dbg.value(metadata !DIArgList(i32 %a, i32 %a, i32 %a, i32 %b, i32 %a, i32 %b, i32 %b, i32 %a, i32 %a, i32 %b, i32 %b),
; CHECK-SAME: ![[VAR_C:[0-9]+]],
; CHECK-SAME: !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 10, DW_OP_ne, DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_LLVM_arg, 9, DW_OP_eq, DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_LLVM_arg, 8, DW_OP_gt, DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_LLVM_arg, 7, DW_OP_gt, DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_LLVM_arg, 6, DW_OP_ge, DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_LLVM_arg, 5, DW_OP_ge, DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_LLVM_arg, 4, DW_OP_lt, DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_LLVM_arg, 3, DW_OP_lt, DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_LLVM_arg, 2, DW_OP_le, DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_LLVM_arg, 1, DW_OP_le, DW_OP_stack_value))

; CHECK: ![[VAR_C]] = !DILocalVariable(name: "c"

define i32 @"?multiply@@YAHHH@Z"(i32 %a, i32 %b) !dbg !8 {
entry:
  %icmp1 = icmp ne i32 %a, 0, !dbg !15
  %icmp1.1 = zext i1 %icmp1 to i32
  %icmp2 = icmp eq i32 %icmp1.1, 0, !dbg !15
  %icmp2.1 = zext i1 %icmp2 to i32
  %icmp3 = icmp ugt i32 %icmp2.1, 1, !dbg !15
  %icmp3.1 = zext i1 %icmp3 to i32
  %icmp4 = icmp sgt i32 %icmp3.1, -1, !dbg !15
  %icmp4.1 = zext i1 %icmp4 to i32
  %icmp5 = icmp uge i32 %icmp4.1, 2, !dbg !15
  %icmp5.1 = zext i1 %icmp5 to i32
  %icmp6 = icmp sge i32 %icmp5.1, -2, !dbg !15
  %icmp6.1 = zext i1 %icmp6 to i32
  %icmp7 = icmp ult i32 %icmp6.1, 3, !dbg !15
  %icmp7.1 = zext i1 %icmp7 to i32
  %icmp8 = icmp slt i32 %icmp7.1, -3, !dbg !15
  %icmp8.1 = zext i1 %icmp8 to i32
  %icmp9 = icmp ule i32 %icmp8.1, 4, !dbg !15
  %icmp9.1 = zext i1 %icmp9 to i32
  %icmp10 = icmp sle i32 %icmp9.1, -4, !dbg !15
  call void @llvm.dbg.value(metadata i1 %icmp10, metadata !16, metadata !DIExpression()), !dbg !13
  %icmp11 = icmp ne i32 %a, %b, !dbg !15
  %icmp11.1 = zext i1 %icmp11 to i32
  %icmp12 = icmp eq i32 %icmp11.1, %b, !dbg !15
  %icmp12.1 = zext i1 %icmp12 to i32
  %icmp13 = icmp ugt i32 %icmp12.1, %a, !dbg !15
  %icmp13.1 = zext i1 %icmp13 to i32
  %icmp14 = icmp sgt i32 %icmp13.1, %a, !dbg !15
  %icmp14.1 = zext i1 %icmp14 to i32
  %icmp15 = icmp uge i32 %icmp14.1, %b, !dbg !15
  %icmp15.1 = zext i1 %icmp15 to i32
  %icmp16 = icmp sge i32 %icmp15.1, %b, !dbg !15
  %icmp16.1 = zext i1 %icmp16 to i32
  %icmp17 = icmp ult i32 %icmp16.1, %a, !dbg !15
  %icmp17.1 = zext i1 %icmp17 to i32
  %icmp18 = icmp slt i32 %icmp17.1, %b, !dbg !15
  %icmp18.1 = zext i1 %icmp18 to i32
  %icmp19 = icmp ule i32 %icmp18.1, %a, !dbg !15
  %icmp19.1 = zext i1 %icmp19 to i32
  %icmp20 = icmp sle i32 %icmp19.1, %a, !dbg !15
  call void @llvm.dbg.value(metadata i1 %icmp20, metadata !16, metadata !DIExpression()), !dbg !13
  %mul = mul nsw i32 %a, %b, !dbg !17
  ret i32 %mul, !dbg !17
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 11.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 11.0.0"}
!8 = distinct !DISubprogram(name: "multiply", linkageName: "?multiply@@YAHHH@Z", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocalVariable(name: "b", arg: 2, scope: !8, file: !1, line: 1, type: !11)
!13 = !DILocation(line: 0, scope: !8)
!14 = !DILocalVariable(name: "a", arg: 1, scope: !8, file: !1, line: 1, type: !11)
!15 = !DILocation(line: 2, scope: !8)
!16 = !DILocalVariable(name: "c", scope: !8, file: !1, line: 2, type: !11)
!17 = !DILocation(line: 3, scope: !8)
