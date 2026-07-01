; RUN: llc -mattr=+egpr -mtriple=x86_64-windows-msvc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mattr=+egpr -mtriple=x86_64-windows-msvc < %s -filetype=obj | llvm-readobj --codeview - | FileCheck %s --check-prefix=OBJ

; ASM: #DEBUG_VALUE: test:x <- $r16d
; OBJ:      DefRangeRegisterSym {
; OBJ-NEXT:   Kind: S_DEFRANGE_REGISTER (0x1141)
; OBJ-NEXT:   Register: R16D (0x388)
; OBJ-NEXT:   MayHaveNoName: 0
; OBJ-NEXT:   LocalVariableAddrRange {
; OBJ-NEXT:     OffsetStart: .text+0x1
; OBJ-NEXT:     ISectStart: 0x0
; OBJ-NEXT:     Range: 0x3
; OBJ-NEXT:   }
; OBJ-NEXT: }

; This test is to check CodeView register IDs for APX EGPR.
define i32 @test() nounwind !dbg !4 {
entry:
  %0 = call i32 asm sideeffect "nop", "={r16},~{dirflag},~{fpsr},~{flags}"(), !dbg !7
  #dbg_value(i32 %0, !8, !DIExpression(), !10)
  ret i32 0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: ".")
!2 = !{i32 2, !"CodeView", i32 1}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 3, scope: !4)
!8 = !DILocalVariable(name: "x", scope: !4, file: !1, line: 2, type: !9)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !4)
