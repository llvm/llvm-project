; RUN: llc -debug-entry-values -filetype=asm -o - %s | FileCheck %s
; RUN: llc -force-instr-ref-livedebugvalues=1 -debug-entry-values -filetype=asm -o - %s | FileCheck %s

; Verify that the size operands of the DW_OP_GNU_entry_value operations are
; correct for the multi-byte DW_OP_regx expressions.

; Based on the following C reproducer:
;
; extern void clobber();
; double global;
; int f(double a, double b) {
;   global = a + b;
;   clobber();
;   return 1;
; }

; This test checks the assembly output rather than the output from
; llvm-dwarfdump, as the latter printed the DW_OP_regx D0 correctly, even
; though the entry value's size operand did not fully cover that:

; DW_OP_GNU_entry_value(DW_OP_regx D0), DW_OP_stack_value
;
; whereas readelf interpreted it as an DW_OP_GNU_entry_value covering one byte,
; resulting in garbage data:
;
; DW_OP_GNU_entry_value: (DW_OP_regx: 0 (r0)); DW_OP_breg16 (r16): 2; DW_OP_stack_value

; CHECK:      .byte   243       @ DW_OP_GNU_entry_value
; CHECK-NEXT: .byte   3         @ 3
; CHECK-NEXT: .byte   144       @ DW_OP_regx
; CHECK-NEXT: .byte   128       @ 256
; CHECK-NEXT: .byte   2         @
; CHECK-NEXT: .byte   159       @ DW_OP_stack_value

; CHECK:      .byte   243       @ DW_OP_GNU_entry_value
; CHECK-NEXT: .byte   3         @ 3
; CHECK-NEXT: .byte   144       @ DW_OP_regx
; CHECK-NEXT: .byte   129       @ 257
; CHECK-NEXT: .byte   2         @
; CHECK-NEXT: .byte   159       @ DW_OP_stack_value

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7-unknown-unknown"

@global = common global double 0.000000e+00, align 8, !dbg !0

; Function Attrs: nounwind
define arm_aapcs_vfpcc i32 @f(double %a, double %b) #0 !dbg !12 {
entry:
  call void @llvm.dbg.value(metadata double %a, metadata !17, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.value(metadata double %b, metadata !18, metadata !DIExpression()), !dbg !19
  %add = fadd double %a, %b, !dbg !20
  store double %add, ptr @global, align 8, !dbg !20
  tail call arm_aapcs_vfpcc void @clobber(), !dbg !21
  ret i32 1, !dbg !22
}

declare arm_aapcs_vfpcc void @clobber()

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "global", scope: !2, file: !3, line: 4, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 10.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "entry-value-multi-byte-expr.c", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 1, !"min_enum_size", i32 4}
!11 = !{!"clang version 10.0.0"}
!12 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 6, type: !13, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !16)
!13 = !DISubroutineType(types: !14)
!14 = !{!15, !6, !6}
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !{!17, !18}
!17 = !DILocalVariable(name: "a", arg: 1, scope: !12, file: !3, line: 6, type: !6)
!18 = !DILocalVariable(name: "b", arg: 2, scope: !12, file: !3, line: 6, type: !6)
!19 = !DILocation(line: 0, scope: !12)
!20 = !DILocation(line: 7, scope: !12)
!21 = !DILocation(line: 8, scope: !12)
!22 = !DILocation(line: 9, scope: !12)
