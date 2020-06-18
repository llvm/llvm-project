; RUN: llc < %s | FileCheck %s

; Input C code:

;  int i = input();  // Nested case
;  int j = input();  // Trivial def-use.
;  output(i, j);

; The ll below generates 330 lines of .S, so relevant parts that the
; WebAssemblyDebugFixup pass affects:

; CHECK: 	call	input
; CHECK: .Ltmp0:
; CHECK: 	call	input
; CHECK: .Ltmp1:
; CHECK: 	call	output
; CHECK: .Ltmp2:

; This defines variable "i" which is live on the stack between Ltmp0 and Ltmp2,
; 2 = TI_OPERAND_STACK and 0 = stack offset.

; CHECK: 	.section	.debug_loc,"",@
; CHECK: .Ldebug_loc0:
; CHECK: 	.int32	.Ltmp0-.Lfunc_begin0
; CHECK: 	.int32	.Ltmp2-.Lfunc_begin0
; CHECK: 	.int16	4                       # Loc expr size
; CHECK: 	.int8	237                       # DW_OP_WASM_location
; CHECK: 	.int8	2                         # 2
; CHECK: 	.int8	0                         # 0
; CHECK: 	.int8	159                       # DW_OP_stack_value

; This defines variable "j" which is live on the stack between Ltmp1 and Ltmp2,
; 2 = TI_OPERAND_STACK and 1 = stack offset.

; CHECK: .Ldebug_loc1:
; CHECK: 	.int32	.Ltmp1-.Lfunc_begin0
; CHECK: 	.int32	.Ltmp2-.Lfunc_begin0
; CHECK: 	.int16	4                       # Loc expr size
; CHECK: 	.int8	237                       # DW_OP_WASM_location
; CHECK: 	.int8	2                         # 2
; CHECK: 	.int8	1                         # 1
; CHECK: 	.int8	159                       # DW_OP_stack_value




source_filename = "stackified.c"
target triple = "wasm32-unknown-unknown"

define void @foo() !dbg !12 {
entry:
  %call = call i32 @input(), !dbg !18
  call void @llvm.dbg.value(metadata i32 %call, metadata !16, metadata !DIExpression()), !dbg !19
  %call1 = call i32 @input(), !dbg !20
  call void @llvm.dbg.value(metadata i32 %call1, metadata !17, metadata !DIExpression()), !dbg !19
  call void @output(i32 %call, i32 %call1), !dbg !21
  ret void, !dbg !22
}

declare i32 @input()

declare !dbg !4 void @output(i32, i32)

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git ed7aaf832444411ce93aa0443425ce401f5c7a8e)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "stackified.c", directory: "C:\\stuff\\llvm-project")
!2 = !{}
!3 = !{!4}
!4 = !DISubprogram(name: "output", scope: !1, file: !1, line: 2, type: !5, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 7, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git ed7aaf832444411ce93aa0443425ce401f5c7a8e)"}
!12 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !13, scopeLine: 3, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{null}
!15 = !{!16, !17}
!16 = !DILocalVariable(name: "i", scope: !12, file: !1, line: 4, type: !7)
!17 = !DILocalVariable(name: "j", scope: !12, file: !1, line: 5, type: !7)
!18 = !DILocation(line: 4, column: 11, scope: !12)
!19 = !DILocation(line: 0, scope: !12)
!20 = !DILocation(line: 5, column: 11, scope: !12)
!21 = !DILocation(line: 6, column: 3, scope: !12)
!22 = !DILocation(line: 7, column: 1, scope: !12)
