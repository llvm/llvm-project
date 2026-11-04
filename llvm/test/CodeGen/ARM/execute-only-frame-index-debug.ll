; RUN: llc %s -o - | FileCheck %s
; RUN: opt -S -strip-debug %s | llc -o - | FileCheck %s

target triple = "thumbv6m-none-eabi"

declare void @g(i32)

; A debug value for the stack-passed argument must not cause extra callee-saved
; registers to be spilled.
; CHECK-LABEL: f:
; CHECK: push {r4, r5}
; CHECK: pop {r4, r5}
define i32 @f(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) #0 !dbg !4 {
entry:
  #dbg_value(i32 %e, !8, !DIExpression(), !9)
  %s = add i32 %a, %b
  call void @g(i32 %s)
  %t = add i32 %s, %d
  ret i32 %t
}

; A real use of the same stack-passed argument must still reserve enough
; callee-saved registers for frame-index elimination.
; CHECK-LABEL: uses_stack_arg:
; CHECK: push {r4, r5, r6, r7}
; CHECK: pop {r4, r5, r6, r7}
define i32 @uses_stack_arg(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) #0 {
entry:
  %s = add i32 %a, %b
  call void @g(i32 %s)
  %t = add i32 %s, %e
  ret i32 %t
}

attributes #0 = { nounwind "frame-pointer"="all" "target-features"="+execute-only,+aapcs-frame-chain" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "x", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !5, unit: !0, spFlags: DISPFlagDefinition | DISPFlagOptimized)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !7, !7, !7, !7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DILocalVariable(name: "e", arg: 5, scope: !4, file: !1, line: 1, type: !7)
!9 = !DILocation(line: 1, scope: !4)
