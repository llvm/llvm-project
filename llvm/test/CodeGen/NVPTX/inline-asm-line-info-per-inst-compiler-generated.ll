; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_89 | FileCheck %s
; RUN: %if ptxas-sm_89 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_89 | %ptxas-verify -arch=sm_89 %}

; Compiler-synthesized multi-line PTX inline asm.
;
; A DSL frontend can lower one source-level construct -- an assert, a barrier,
; an intrinsic -- into a single block of inline PTX. The individual lines of
; that block have no source correspondence; they all belong to the one
; construct, here at test.cu:5:28.
;
; NVPTXAsmPrinter::emitInlineAsm otherwise splits the asm string on newlines
; and emits a .loc per PTX instruction, doing line++ for every physical line.
; E.g. for this block of inline ptx starting at line 5, yields
; .loc 1 8 .. .loc 1 12, attributing the block to unrelated source lines.
;
; isImplicitCode on the asm call's !dbg marks the block as compiler generated,
; so the backend emits only the enclosing .loc before it and leaves the block
; itself alone.

target triple = "nvptx64-nvidia-cuda"

; CHECK-LABEL: .entry synth_asm
; CHECK:       .loc 1 5 28
; CHECK:       // begin inline asm
; CHECK-NOT:   .loc 1 8 28
; CHECK-NOT:   .loc 1 9 28
; CHECK-NOT:   .loc 1 10 28
; CHECK-NOT:   .loc 1 11 28
; CHECK-NOT:   .loc 1 12 28
; CHECK:       // end inline asm

define ptx_kernel void @synth_asm(i1 %cond) !dbg !3 {
entry:
  br i1 %cond, label %exit, label %trap, !dbg !6

trap:
  call void asm sideeffect "{\0A  .reg .b32 %t0;\0A  .reg .b32 %t1;\0A  mov.b32 %t0, 1;\0A  add.s32 %t1, %t0, 2;\0A  mul.lo.s32 %t0, %t1, 3;\0A  sub.s32 %t1, %t0, 4;\0A  xor.b32 %t0, %t1, 5;\0A}", ""(), !dbg !7
  br label %exit

exit:
  ret void, !dbg !6
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "manual", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !DIFile(filename: "test.cu", directory: "/tmp")
!3 = distinct !DISubprogram(name: "synth_asm", scope: !2, file: !2, line: 4, type: !4, scopeLine: 4, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = !DILocation(line: 5, column: 28, scope: !3)
!7 = !DILocation(line: 5, column: 28, scope: !3, isImplicitCode: true)
