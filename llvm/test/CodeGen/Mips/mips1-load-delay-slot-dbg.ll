; RUN: llc -mtriple=mipsel-linux-gnu -mcpu=mips1 -relocation-model=static < %s \
; RUN:   | FileCheck %s

; MIPS I has no load interlock. The delay slot filler must not move the lw
; into the bnez delay slot, because the branch target reads $2 right away.
; A debug instruction at the start of the target must not hide that read.

@g = global i32 0
@h = global i32 0

; CHECK-LABEL: f:
; CHECK:         lw $2, %lo(g)($1)
; CHECK-NEXT:    bnez $4, $BB0_2
; CHECK-NEXT:    nop
; CHECK:       $BB0_2:
; CHECK-NEXT:    #DEBUG_VALUE:
; CHECK-NEXT:    addu $1, $2, $2

define void @f(i32 %c, i32 %d) !dbg !5 {
entry:
  %x = load i32, ptr @g, align 4
  %t = icmp eq i32 %c, 0
  br i1 %t, label %a, label %b
a:
  store i32 %d, ptr @h
  br label %b
b:
    #dbg_value(i32 %d, !7, !DIExpression(), !9)
  %y = add i32 %x, %x
  store i32 %y, ptr @h
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !6, unit: !0, spFlags: DISPFlagDefinition, retainedNodes: !{!7})
!6 = !DISubroutineType(types: !{})
!7 = !DILocalVariable(name: "d", arg: 2, scope: !5, file: !1, line: 1, type: !8)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DILocation(line: 1, scope: !5)
