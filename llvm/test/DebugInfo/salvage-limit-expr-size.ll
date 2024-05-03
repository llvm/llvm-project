; RUN: opt %s -passes=dce -S | FileCheck %s
; RUN: opt %s -passes=dce -S --try-experimental-debuginfo-iterators | FileCheck %s

;; Tests that a DIExpression will only be salvaged up to a certain length, and
;; will produce an undef value if an expression would need to exceed that length.

define i32 @"?multiply@@YAHHH@Z"(i32 %a, i32 %b) !dbg !8 {
entry:
  %add.1 = add nsw i32 %a, 5, !dbg !14
  %add.2 = add nsw i32 %a, %b, !dbg !14
  ;; These expressions should salvage successfully, up to exactly 128 elements.
  ; CHECK: call void @llvm.dbg.value(metadata i32 %a, metadata ![[VAR_C:[0-9]+]]
  ; CHECK-NEXT: call void @llvm.dbg.value(metadata !DIArgList(i32 %a, i32 %b), metadata ![[VAR_C]]
  call void @llvm.dbg.value(metadata i32 %add.1, metadata !12, metadata !DIExpression(DW_OP_constu, 1, DW_OP_plus, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !13
  call void @llvm.dbg.value(metadata i32 %add.2, metadata !12, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !13
  ;; These expressions should be set undef, as they would salvage up to exactly 129 elements.
  ; CHECK-NEXT: call void @llvm.dbg.value(metadata i32 poison, metadata ![[VAR_C]]
  ; CHECK-NEXT: call void @llvm.dbg.value(metadata i32 poison, metadata ![[VAR_C]]
  call void @llvm.dbg.value(metadata i32 %add.1, metadata !12, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !13
  call void @llvm.dbg.value(metadata i32 %add.2, metadata !12, metadata !DIExpression(DW_OP_constu, 1, DW_OP_plus, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !13
  %mul = mul nsw i32 %a, %b, !dbg !15
  ret i32 %mul, !dbg !15
}

; CHECK: ![[VAR_C]] = !DILocalVariable(name: "c"

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
!12 = !DILocalVariable(name: "c", scope: !8, file: !1, line: 2, type: !11)
!13 = !DILocation(line: 0, scope: !8)
!14 = !DILocation(line: 2, scope: !8)
!15 = !DILocation(line: 3, scope: !8)
