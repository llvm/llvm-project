; RUN: llc -O1 -filetype=asm -mtriple x86_64-unknown-linux-gnu -mcpu=x86-64 \
; RUN:    -o - %s -stop-after=livedebugvars \
; RUN:    -experimental-debug-variable-locations=false \
; RUN: | FileCheck %s --check-prefixes=CHECK,VARLOCS
; RUN: llc -O1 -filetype=asm -mtriple x86_64-unknown-linux-gnu -mcpu=x86-64 \
; RUN:    -o - %s -stop-after=livedebugvars \
; RUN:    -experimental-debug-variable-locations=true \
; RUN: | FileCheck %s --check-prefixes=CHECK,INSTRREF

; CHECK:         $eax = MOV32rm
; INSTRREF-SAME:      debug-instr-number 1
; INSTRREF:      DBG_INSTR_REF {{.+}}, dbg-instr-ref(1, 0)
; VARLOCS:         DBG_VALUE $eax
; INSTRREF:        DBG_VALUE_LIST {{.+}} $eax
; CHECK:         $eax = SHL32rCL killed renamable $eax,
; INSTRREF-SAME:      debug-instr-number 2
; INSTRREF:      DBG_INSTR_REF {{.+}}, dbg-instr-ref(2, 0)
; VARLOCS:       DBG_VALUE $eax
; INSTRREF:      DBG_VALUE_LIST {{.+}} $eax
; VARLOCS:       DBG_VALUE $rsp, 0, !{{[0-9]+}}, !DIExpression(DW_OP_constu, 4, DW_OP_minus)
; INSTRREF:      DBG_VALUE_LIST !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_constu, 4, DW_OP_minus, DW_OP_deref), $rsp
; VARLOCS:       DBG_VALUE $eax
; CHECK:         $eax = SHL32rCL killed renamable $eax,
; INSTRREF-SAME:      debug-instr-number 3
; INSTRREF:      DBG_INSTR_REF {{.+}}, dbg-instr-ref(3, 0)
; VARLOCS:       DBG_VALUE $eax
; INSTRREF:      DBG_VALUE_LIST {{.+}} $eax
; CHECK:         RET64 $eax

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@var = local_unnamed_addr global i32 8, !dbg !0
@sc = local_unnamed_addr global i32 1, !dbg !6

define i32 @main() local_unnamed_addr !dbg !14 {
entry:
  %0 = load i32, ptr @var
  tail call void @llvm.dbg.value(metadata i32 %0, metadata !18, metadata !DIExpression()), !dbg !20
  %1 = load i32, ptr @sc
  %shl = shl i32 %0, %1
  tail call void @llvm.dbg.value(metadata i32 %shl, metadata !18, metadata !DIExpression()), !dbg !20
  tail call void asm sideeffect "", "~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{dirflag},~{fpsr},~{flags}"(), !srcloc !25
  %2 = load i32, ptr @sc
  %shl2 = shl i32 %shl, %2
  tail call void @llvm.dbg.value(metadata i32 %shl2, metadata !18, metadata !DIExpression()), !dbg !20
  store i32 %shl2, ptr @var
  ret i32 %shl2, !dbg !20
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "var", scope: !2, file: !3, line: 10, type: !9, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 6.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "bar.c", directory: ".")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "sc", scope: !2, file: !3, line: 11, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!14 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 12, type: !15, isLocal: false, isDefinition: true, scopeLine: 12, isOptimized: true, unit: !2, retainedNodes: !17)
!15 = !DISubroutineType(types: !16)
!16 = !{!9}
!17 = !{!18}
!18 = !DILocalVariable(name: "bazinga", scope: !14, file: !3, line: 13, type: !9)
!20 = !DILocation(line: 13, column: 7, scope: !14)
!25 = !{i32 -2147471481}
