; RUN: llc -mtriple=riscv64 -mattr=+v -filetype=obj %s -o %t64.o
; RUN: llvm-dwarfdump --debug-info %t64.o | FileCheck --check-prefixes=CHECK,RV64 %s
; RUN: llc -mtriple=riscv32 -mattr=+v -filetype=obj %s -o %t32.o
; RUN: llvm-dwarfdump --debug-info %t32.o | FileCheck --check-prefixes=CHECK,RV32 %s

; DW_AT_frame_base must be expressed in terms of the CFA, not the live frame
; register. A register-based frame base (DW_OP_reg8 / DW_OP_reg2) stops working
; as soon as the epilogue reloads s0 or restores sp, so DW_OP_fbreg locals read
; from the caller's frame when the debugger stops on the last instructions of a
; function. The CFA is recovered from the CFI and stays valid throughout.
;
; The constant added to the CFA mirrors what emitPrologue puts into the CFI:
;   * with a frame pointer: ".cfi_def_cfa s0, VarArgsSaveSize"
;   * without:              ".cfi_def_cfa_offset StackSize"

; CHECK:      DW_AT_frame_base (DW_OP_call_frame_cfa)
; CHECK-NEXT: DW_AT_name ("withfp")
; RV64:       DW_AT_location (DW_OP_fbreg -20)
; RV32:       DW_AT_location (DW_OP_fbreg -12)

; CHECK:      DW_AT_frame_base (DW_OP_call_frame_cfa, DW_OP_consts -16, DW_OP_plus)
; CHECK-NEXT: DW_AT_name ("nofp")
; RV64:       DW_AT_location (DW_OP_fbreg +4)
; RV32:       DW_AT_location (DW_OP_fbreg +8)

; RV64:       DW_AT_frame_base (DW_OP_call_frame_cfa, DW_OP_consts -64, DW_OP_plus)
; RV32:       DW_AT_frame_base (DW_OP_call_frame_cfa, DW_OP_consts -32, DW_OP_plus)
; CHECK-NEXT: DW_AT_name ("va")

; CHECK:      DW_AT_frame_base (DW_OP_reg2 X2)
; CHECK-NEXT: DW_AT_name ("rvv")

declare void @sink(ptr)

; Frame pointer: s0 == CFA.
define i32 @withfp() #0 !dbg !10 {
  %a = alloca i32
    #dbg_declare(ptr %a, !11, !DIExpression(), !12)
  call void @sink(ptr %a), !dbg !12
  %v = load i32, ptr %a, !dbg !12
  ret i32 %v, !dbg !12
}

; No frame pointer: sp == CFA - StackSize.
define i32 @nofp() #1 !dbg !20 {
  %a = alloca i32
    #dbg_declare(ptr %a, !21, !DIExpression(), !22)
  call void @sink(ptr %a), !dbg !22
  %v = load i32, ptr %a, !dbg !22
  ret i32 %v, !dbg !22
}

; Frame pointer with varargs: s0 == CFA - VarArgsSaveSize.
define i32 @va(i32 %n, ...) #0 !dbg !30 {
  %a = alloca i32
    #dbg_declare(ptr %a, !31, !DIExpression(), !32)
  call void @sink(ptr %a), !dbg !32
  %v = load i32, ptr %a, !dbg !32
  ret i32 %v, !dbg !32
}

; No frame pointer with a scalable vector stack object: the CFA is an
; expression containing vlenb, so the register form is kept.
define i32 @rvv() #1 !dbg !40 {
  %vec = alloca <vscale x 2 x i32>
  %a = alloca i32
    #dbg_declare(ptr %a, !41, !DIExpression(), !42)
  call void @sink(ptr %vec), !dbg !42
  call void @sink(ptr %a), !dbg !42
  %v = load i32, ptr %a, !dbg !42
  ret i32 %v, !dbg !42
}

attributes #0 = { noinline optnone "frame-pointer"="all" }
attributes #1 = { noinline optnone "frame-pointer"="none" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "frame-base.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !5)
!5 = !{!6}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !DISubroutineType(types: !8)
!8 = !{!6, !6, null}

!10 = distinct !DISubprogram(name: "withfp", file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DILocalVariable(name: "a", scope: !10, file: !1, line: 1, type: !6)
!12 = !DILocation(line: 1, scope: !10)

!20 = distinct !DISubprogram(name: "nofp", file: !1, line: 2, type: !4, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0)
!21 = !DILocalVariable(name: "a", scope: !20, file: !1, line: 2, type: !6)
!22 = !DILocation(line: 2, scope: !20)

!30 = distinct !DISubprogram(name: "va", file: !1, line: 3, type: !7, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !0)
!31 = !DILocalVariable(name: "a", scope: !30, file: !1, line: 3, type: !6)
!32 = !DILocation(line: 3, scope: !30)

!40 = distinct !DISubprogram(name: "rvv", file: !1, line: 4, type: !4, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !0)
!41 = !DILocalVariable(name: "a", scope: !40, file: !1, line: 4, type: !6)
!42 = !DILocation(line: 4, scope: !40)
