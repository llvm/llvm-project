; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 -asm-verbose=false | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-unknown"

; Compiled from the following CUDA code:
;
;   #pragma nounroll
;   for (int i = 0; i < 2; ++i)
;     output[i] = input[i];
define void @nounroll(ptr %input, ptr %output) {
; CHECK-LABEL: .visible .func nounroll(
entry:
  br label %for.body

for.body:
; CHECK: .pragma "nounroll"
  %i.06 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %idxprom = sext i32 %i.06 to i64
  %arrayidx = getelementptr inbounds float, ptr %input, i64 %idxprom
  %0 = load float, ptr %arrayidx, align 4
; CHECK: ld.b32
  %arrayidx2 = getelementptr inbounds float, ptr %output, i64 %idxprom
  store float %0, ptr %arrayidx2, align 4
; CHECK: st.b32
  %inc = add nuw nsw i32 %i.06, 1
  %exitcond = icmp eq i32 %inc, 2
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !0
; CHECK-NOT: ld.b32
; CHECK-NOT: st.b32

for.end:
  ret void
}

; Compiled from the following CUDA code:
;
;   #pragma unroll 1
;   for (int i = 0; i < 2; ++i)
;     output[i] = input[i];
define void @unroll1(ptr %input, ptr %output) {
; CHECK-LABEL: .visible .func unroll1(
entry:
  br label %for.body

for.body:
; CHECK: .pragma "nounroll"
  %i.06 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %idxprom = sext i32 %i.06 to i64
  %arrayidx = getelementptr inbounds float, ptr %input, i64 %idxprom
  %0 = load float, ptr %arrayidx, align 4
; CHECK: ld.b32
  %arrayidx2 = getelementptr inbounds float, ptr %output, i64 %idxprom
  store float %0, ptr %arrayidx2, align 4
; CHECK: st.b32
  %inc = add nuw nsw i32 %i.06, 1
  %exitcond = icmp eq i32 %inc, 2
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !2
; CHECK-NOT: ld.b32
; CHECK-NOT: st.b32

for.end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.unroll.count", i32 1}

; A lexical scope beginning in the loop header introduces a debug label.
; Keep nounroll after that label and before the first instruction. Leading empty
; inline assembly and debug values must not emit the pragma early.
define void @nounroll_debug(ptr %p, i32 %n) !dbg !7 {
; CHECK-LABEL: .visible .func nounroll_debug(
; CHECK-NOT: .pragma
; CHECK: [[LOOP:\$L__BB[0-9_]+]]:
; CHECK-NOT: .pragma
; CHECK: // begin inline asm
; CHECK-NEXT: // end inline asm
; CHECK-NOT: .pragma
; CHECK: $L__tmp{{[0-9]+}}:
; CHECK: .loc {{[0-9]+}} 3 0
; CHECK-NEXT: .pragma "nounroll";
; CHECK-NEXT: ld.volatile.b32
; CHECK-NOT: .pragma
; CHECK: bra [[LOOP]];
; CHECK-NOT: .pragma
; CHECK: ret;
entry:
  br label %loop, !dbg !10

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  call void asm sideeffect "", ""(), !dbg !10
    #dbg_value(i32 %i, !14, !DIExpression(), !11)
  %v = load volatile i32, ptr %p, align 4, !dbg !11
  %inc = add i32 %i, 1, !dbg !11
  %cmp = icmp ult i32 %inc, %n, !dbg !11
  br i1 %cmp, label %loop, label %exit, !dbg !11, !llvm.loop !0

exit:
  ret void, !dbg !12
}

; Inline assembly bypasses the target's emitInstruction hook. Also exercise
; llvm.loop.unroll.count = 1 with debug information.
define void @unroll1_debug_inlineasm(i32 %n) !dbg !17 {
; CHECK-LABEL: .visible .func unroll1_debug_inlineasm(
; CHECK-NOT: .pragma
; CHECK: [[ASM_LOOP:\$L__BB[0-9_]+]]:
; CHECK-NOT: .pragma
; CHECK: $L__tmp{{[0-9]+}}:
; CHECK: .loc {{[0-9]+}} 7 0
; CHECK-NEXT: // begin inline asm
; CHECK-NEXT: .pragma "nounroll";
; CHECK-NEXT: .loc {{[0-9]+}} 7 0
; CHECK-NEXT: bar.sync 0;
; CHECK-NOT: .pragma
; CHECK: bra [[ASM_LOOP]];
; CHECK-NOT: .pragma
; CHECK: ret;
entry:
  br label %loop, !dbg !19

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
    #dbg_value(i32 %i, !22, !DIExpression(), !20)
  call void asm sideeffect "bar.sync 0;", ""(), !dbg !20
  %inc = add i32 %i, 1, !dbg !20
  %cmp = icmp ult i32 %inc, %n, !dbg !20
  br i1 %cmp, label %loop, label %exit, !dbg !20, !llvm.loop !2

exit:
  ret void, !dbg !21
}

; INLINEASM_BR uses the same generic emission path as INLINEASM.
define void @nounroll_debug_inlineasm_br(i1 %done) !dbg !23 {
; CHECK-LABEL: .visible .func nounroll_debug_inlineasm_br(
; CHECK-NOT: .pragma
; CHECK: [[BR_LOOP:\$L__BB[0-9_]+]]:
; CHECK-NOT: .pragma
; CHECK: $L__tmp{{[0-9]+}}:
; CHECK: .loc {{[0-9]+}} 11 0
; CHECK-NEXT: // begin inline asm
; CHECK-NEXT: .pragma "nounroll";
; CHECK-NEXT: .loc {{[0-9]+}} 11 0
; CHECK-NEXT: bra
; CHECK-NOT: .pragma
; CHECK: bra [[BR_LOOP]];
; CHECK-NOT: .pragma
; CHECK: ret;
entry:
  br label %loop, !dbg !25

loop:
  callbr void asm sideeffect "bra $0;", "!i"()
          to label %latch [label %exit], !dbg !26

latch:
  br i1 %done, label %exit, label %loop, !dbg !26, !llvm.loop !0

exit:
  ret void, !dbg !27
}

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!5}
!4 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "test", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !DIFile(filename: "nounroll.c", directory: "/")
!7 = distinct !DISubprogram(name: "nounroll_debug", scope: !6, file: !6, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !9)
!8 = !DISubroutineType(types: !9)
!9 = !{}
!10 = !DILocation(line: 2, scope: !7)
!11 = !DILocation(line: 3, scope: !13)
!12 = !DILocation(line: 4, scope: !7)
!13 = distinct !DILexicalBlock(scope: !7, file: !6, line: 3)
!14 = !DILocalVariable(name: "i", scope: !13, file: !6, line: 3, type: !15)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = distinct !DISubprogram(name: "unroll1_debug_inlineasm", scope: !6, file: !6, line: 5, type: !8, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !9)
!18 = distinct !DILexicalBlock(scope: !17, file: !6, line: 7)
!19 = !DILocation(line: 6, scope: !17)
!20 = !DILocation(line: 7, scope: !18)
!21 = !DILocation(line: 8, scope: !17)
!22 = !DILocalVariable(name: "i", scope: !18, file: !6, line: 7, type: !15)
!23 = distinct !DISubprogram(name: "nounroll_debug_inlineasm_br", scope: !6, file: !6, line: 9, type: !8, scopeLine: 9, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !9)
!24 = distinct !DILexicalBlock(scope: !23, file: !6, line: 11)
!25 = !DILocation(line: 10, scope: !23)
!26 = !DILocation(line: 11, scope: !24)
!27 = !DILocation(line: 12, scope: !23)
