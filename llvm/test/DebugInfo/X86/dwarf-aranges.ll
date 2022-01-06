; RUN: llc -generate-arange-section < %s | FileCheck %s

; -- header --
; CHECK: .short 2 # DWARF Arange version number
; CHECK-NEXT: .long .Lcu_begin0
; CHECK-NEXT: .byte 8 # Address Size (in bytes)
; CHECK-NEXT: .byte 0 # Segment Size (in bytes)
; -- alignment --
; CHECK-NEXT: .zero 4,255

; <data section> - it should have made one span covering all vars in this CU.
; CHECK-NEXT: .quad some_data
; CHECK-NEXT: .quad .Lsec_end0-some_data

; <other sections> - it should have made one span covering all vars in this CU.
; CHECK-NEXT: .quad some_other
; CHECK-NEXT: .quad .Lsec_end1-some_other

; <common symbols> - it should have made one span for each symbol.
; CHECK-NEXT: .quad some_bss
; CHECK-NEXT: .quad 4

; <text section> - it should have made one span covering all functions in this CU.
; CHECK-NEXT: .quad .Lfunc_begin0
; CHECK-NEXT: .quad .Lsec_end2-.Lfunc_begin0

; -- finish --
; CHECK-NEXT: # ARange terminator

; -- source code --
; Generated from: "clang -c -g -emit-llvm"
;
; int some_data = 4;
; int some_bss;
; int some_other __attribute__ ((section ("strange+section"))) = 5;
; 
; void some_code()
; {
;    some_bss += some_data + some_other;
; }

source_filename = "test/DebugInfo/X86/dwarf-aranges.ll"
target triple = "x86_64-unknown-linux-gnu"

@some_data = global i32 4, align 4, !dbg !0
@some_other = global i32 5, section "strange+section", align 4, !dbg !4
@some_bss = common global i32 0, align 4, !dbg !6

define void @some_code() !dbg !13 {
entry:
  %0 = load i32, i32* @some_data, align 4, !dbg !16
  %1 = load i32, i32* @some_other, align 4, !dbg !16
  %add = add nsw i32 %0, %1, !dbg !16
  %2 = load i32, i32* @some_bss, align 4, !dbg !16
  %add1 = add nsw i32 %2, %add, !dbg !16
  store i32 %add1, i32* @some_bss, align 4, !dbg !16
  ret void, !dbg !17
}

!llvm.dbg.cu = !{!8}
!llvm.module.flags = !{!11, !12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "some_data", scope: null, file: !2, line: 1, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "test.c", directory: "/home/kayamon")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!5 = !DIGlobalVariable(name: "some_other", scope: null, file: !2, line: 3, type: !3, isLocal: false, isDefinition: true)
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = !DIGlobalVariable(name: "some_bss", scope: null, file: !2, line: 2, type: !3, isLocal: false, isDefinition: true)
!8 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 3.4 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !9, retainedTypes: !9, globals: !10, imports: !9)
!9 = !{}
!10 = !{!0, !4, !6}
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 1, !"Debug Info Version", i32 3}
!13 = distinct !DISubprogram(name: "some_code", scope: !2, file: !2, line: 5, type: !14, isLocal: false, isDefinition: true, scopeLine: 6, virtualIndex: 6, isOptimized: false, unit: !8, retainedNodes: !9)
!14 = !DISubroutineType(types: !15)
!15 = !{null}
!16 = !DILocation(line: 7, scope: !13)
!17 = !DILocation(line: 8, scope: !13)

