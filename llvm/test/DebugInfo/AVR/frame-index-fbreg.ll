; RUN: llc -mtriple=avr -O0 -filetype=obj -o %t %s
; RUN: llvm-dwarfdump --debug-info %t | FileCheck %s
; RUN: llvm-dwarfdump --verify %t
;
; RUN: llc -mtriple=avr -O0 -dwarf-version=4 -filetype=obj -o %t4 %s
; RUN: llvm-dwarfdump --debug-info %t4 | FileCheck %s --check-prefix=V4
;
; The DW_OP_fbreg offsets must match the Y-relative addressing used by the
; generated code (see the ASM checks below).
; RUN: llc -mtriple=avr -O0 -o - %s | FileCheck %s --check-prefix=ASM

; Generated with:
;
;   clang --target=avr -O0 -g -S -emit-llvm
;
; from the following source:
;
;   int add(int a, int b) {
;     int c = a + b;
;     return c;
;   }
;
; At -O0 every variable lives in a stack slot, so its debug location is
; described by a frame index (Loc::MMI). Emitting such a location requires
; AVRFrameLowering::getFrameIndexReference() to report back the register the
; frame index is addressed through -- the Y pointer (R29R28). When that
; out-parameter was left unset, DwarfExpression::addMachineReg() saw a
; non-physical register and hit the llvm_unreachable in
; TargetRegisterInfo::getDwarfRegNumForVirtReg(), crashing the compiler.

; CHECK:      DW_TAG_subprogram
; CHECK:        DW_AT_frame_base{{.*}}(DW_OP_reg28 R29R28)
; CHECK:        DW_AT_name{{.*}}("add")

; CHECK:      DW_TAG_formal_parameter
; CHECK:        DW_AT_location{{.*}}(DW_OP_fbreg +5)
; CHECK:        DW_AT_name{{.*}}("a")

; CHECK:      DW_TAG_formal_parameter
; CHECK:        DW_AT_location{{.*}}(DW_OP_fbreg +3)
; CHECK:        DW_AT_name{{.*}}("b")

; CHECK:      DW_TAG_variable
; CHECK:        DW_AT_location{{.*}}(DW_OP_fbreg +1)
; CHECK:        DW_AT_name{{.*}}("c")

; V4: DW_AT_frame_base{{.*}}(DW_OP_reg28 R29R28)
; V4: DW_AT_location{{.*}}(DW_OP_fbreg +5)
; V4: DW_AT_location{{.*}}(DW_OP_fbreg +3)
; V4: DW_AT_location{{.*}}(DW_OP_fbreg +1)

; The prologue points Y (r29:r28) at the frame and allocates 6 bytes.
; ASM:      in r28, 61
; ASM:      sbiw r28, 6

; The stack slot of each variable, matching the DW_OP_fbreg offsets above.
; ASM:      std Y+5, r24
; ASM:      std Y+3, r22
; ASM:      std Y+1, r24
; ASM:      ldd r24, Y+1

target datalayout = "e-P1-p:16:8-i8:8-i16:8-i32:8-i64:8-f32:8-f64:8-n8:16-a:8"
target triple = "avr"

define dso_local i16 @add(i16 noundef %a, i16 noundef %b) addrspace(1) #0 !dbg !7 {
entry:
  %a.addr = alloca i16, align 1
  %b.addr = alloca i16, align 1
  %c = alloca i16, align 1
  store i16 %a, ptr %a.addr, align 1
    #dbg_declare(ptr %a.addr, !12, !DIExpression(), !13)
  store i16 %b, ptr %b.addr, align 1
    #dbg_declare(ptr %b.addr, !14, !DIExpression(), !15)
    #dbg_declare(ptr %c, !16, !DIExpression(), !17)
  %0 = load i16, ptr %a.addr, align 1, !dbg !18
  %1 = load i16, ptr %b.addr, align 1, !dbg !19
  %add = add nsw i16 %0, %1, !dbg !20
  store i16 %add, ptr %c, align 1, !dbg !17
  %2 = load i16, ptr %c, align 1, !dbg !21
  ret i16 %2, !dbg !22
}

attributes #0 = { noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "avr-dwarf.c", directory: "")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 2}
!5 = !{i32 7, !"frame-pointer", i32 2}
!7 = distinct !DISubprogram(name: "add", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10, !10}
!10 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
!11 = !{}
!12 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!13 = !DILocation(line: 1, column: 13, scope: !7)
!14 = !DILocalVariable(name: "b", arg: 2, scope: !7, file: !1, line: 1, type: !10)
!15 = !DILocation(line: 1, column: 20, scope: !7)
!16 = !DILocalVariable(name: "c", scope: !7, file: !1, line: 2, type: !10)
!17 = !DILocation(line: 2, column: 7, scope: !7)
!18 = !DILocation(line: 2, column: 11, scope: !7)
!19 = !DILocation(line: 2, column: 15, scope: !7)
!20 = !DILocation(line: 2, column: 13, scope: !7)
!21 = !DILocation(line: 3, column: 10, scope: !7)
!22 = !DILocation(line: 3, column: 3, scope: !7)
