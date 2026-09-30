; RUN: opt -passes=hotcoldsplit -hotcoldsplit-threshold=-1 -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx11.0.0"

declare void @sink(i32) cold
declare void @use(i32)

define void @all_inputs(i32 %a, i32 %b, i1 %c) !dbg !6 {
entry:
  %x = add i32 %a, 1, !dbg !11
  %y = add i32 %b, 2, !dbg !11
    #dbg_value(!DIArgList(i32 %x, i32 %y), !9, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value), !11)
  br i1 %c, label %cold, label %exit, !dbg !11

cold:
  %s = add i32 %x, %y, !dbg !11
  call void @sink(i32 %s), !dbg !11
  br label %exit, !dbg !11

exit:
  ret void, !dbg !11
}

define void @some_inputs(i32 %a, i32 %b, i1 %c) !dbg !12 {
entry:
  %x = add i32 %a, 1, !dbg !14
  %y = add i32 %b, 2, !dbg !14
    #dbg_value(!DIArgList(i32 %x, i32 %y), !13, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value), !14)
  br i1 %c, label %cold, label %exit, !dbg !14

cold:
  %m = mul i32 %x, 3, !dbg !14
  call void @sink(i32 %m), !dbg !14
  br label %exit, !dbg !14

exit:
  call void @use(i32 %y), !dbg !14
  ret void, !dbg !14
}

define void @single_location(i32 %a, i1 %c) !dbg !15 {
entry:
  %x = add i32 %a, 1, !dbg !17
    #dbg_value(i32 %x, !16, !DIExpression(DW_OP_plus_uconst, 4, DW_OP_stack_value), !17)
  br i1 %c, label %cold, label %exit, !dbg !17

cold:
  %m = mul i32 %x, 3, !dbg !17
  call void @sink(i32 %m), !dbg !17
  br label %exit, !dbg !17

exit:
  ret void, !dbg !17
}

; Every location is available, so the record is copied whole.
; CHECK-LABEL: define internal void @all_inputs.cold.1(i32 %x, i32 %y)
; CHECK: #dbg_value(!DIArgList(i32 %x, i32 %y), ![[#]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value)
; CHECK: ret void

; %y is not passed in, so the record cannot be described and is dropped.
; CHECK-LABEL: define internal void @some_inputs.cold.1(i32 %x)
; CHECK-NOT: #dbg_value
; CHECK: ret void

; CHECK-LABEL: define internal void @single_location.cold.1(i32 %x)
; CHECK: #dbg_value(i32 %x, ![[#]], !DIExpression(DW_OP_plus_uconst, 4, DW_OP_stack_value)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "all_inputs", scope: !1, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!9 = !DILocalVariable(name: "v", scope: !6, file: !1, line: 2, type: !10)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 2, column: 1, scope: !6)
!12 = distinct !DISubprogram(name: "some_inputs", scope: !1, file: !1, line: 5, type: !7, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!13 = !DILocalVariable(name: "w", scope: !12, file: !1, line: 6, type: !10)
!14 = !DILocation(line: 6, column: 1, scope: !12)
!15 = distinct !DISubprogram(name: "single_location", scope: !1, file: !1, line: 9, type: !7, scopeLine: 9, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!16 = !DILocalVariable(name: "u", scope: !15, file: !1, line: 10, type: !10)
!17 = !DILocation(line: 10, column: 1, scope: !15)
