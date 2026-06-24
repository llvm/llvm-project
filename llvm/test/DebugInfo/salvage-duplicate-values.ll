; RUN: opt %s -passes=dce -S | FileCheck %s

; Tests the results of salvaging variadic dbg.values that use the same SSA value
; multiple times.

; CHECK: #dbg_value(!DIArgList(i32 %a, i32 %a),
; CHECK-SAME: ![[VAR_C:[0-9]+]],
; CHECK-SAME: !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_plus_uconst, 5, DW_OP_LLVM_arg, 1, DW_OP_plus_uconst, 5, DW_OP_plus, DW_OP_stack_value),

; CHECK: #dbg_value(!DIArgList(i32 %a, i32 %a, i32 %b, i32 %b),
; CHECK-SAME: ![[VAR_C]],
; CHECK-SAME: !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 2, DW_OP_plus, DW_OP_LLVM_arg, 1, DW_OP_LLVM_arg, 3, DW_OP_plus, DW_OP_plus, DW_OP_stack_value),

; CHECK: ![[VAR_C]] = !DILocalVariable(name: "c"

define i32 @"?multiply@@YAHHH@Z"(i32 %a, i32 %b) !dbg !8 {
entry:
  %add1 = add nsw i32 %a, 5, !dbg !15
  %add2 = add nsw i32 %a, %b, !dbg !15
  call void @llvm.dbg.value(metadata !DIArgList(i32 %add1, i32 %add1), metadata !16, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value)), !dbg !13
  call void @llvm.dbg.value(metadata !DIArgList(i32 %add2, i32 %add2), metadata !16, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value)), !dbg !13
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
