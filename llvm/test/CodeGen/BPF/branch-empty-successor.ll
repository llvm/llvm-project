; RUN: llc -mtriple=bpfel -mcpu=v4 -verify-machineinstrs -filetype=obj -o - %s \
; RUN:   | llvm-objdump --no-show-raw-insn -d - | FileCheck %s
; RUN: llc -mtriple=bpfel -mcpu=v2 -verify-machineinstrs -filetype=obj -o - %s \
; RUN:   | llvm-objdump --no-show-raw-insn -d - | FileCheck %s

; Test fix for GitHub issue #208984: conditional branches to basic blocks that
; lower to zero BPF instructions must not resolve to the wrong PC.

target triple = "bpf"

; Case 1: branch to bare unreachable block (minbug.ll from #208984).
define void @f(i1 %c) {
  br i1 %c, label %empty, label %fall
fall:
  store i32 0, ptr null, align 4
  unreachable
empty:
  unreachable
}

define void @next() { ret void }

; Case 2: branch to barrier-only block (C reproducer shape from #208984).
define void @g(i1 %cond) {
entry:
  br i1 %cond, label %skip, label %raise
raise:
  tail call void @raise_fn()
  unreachable
skip:
  tail call void asm sideeffect "", "~{memory}"()
  unreachable
}

declare void @raise_fn() #0
attributes #0 = { noreturn }

; Case 3: conditional branch target with debug info only (no BPF code).
define void @h(i1 %c) !dbg !6 {
entry:
  br i1 %c, label %skip, label %fall, !dbg !10
fall:
  store i32 0, ptr null, align 4, !dbg !10
  unreachable
skip:
  call void @llvm.dbg.value(metadata i1 %c, metadata !11, metadata !DIExpression()), !dbg !10
  unreachable
}

declare void @llvm.dbg.value(metadata, metadata, metadata) #1
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "branch-empty-successor.ll", directory: "/tmp")
!6 = distinct !DISubprogram(name: "h", scope: !1, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 1, column: 1, scope: !6)
!11 = !DILocalVariable(name: "c", arg: 1, scope: !6, file: !1, line: 1, type: !9)

; CHECK-LABEL: <f>:
; CHECK:       goto
; CHECK:       exit
; CHECK-NOT:   <next>
; CHECK-LABEL: <next>:
; CHECK:       exit

; CHECK-LABEL: <g>:
; CHECK:       goto
; CHECK:       exit
; CHECK-NOT:   call

; CHECK-LABEL: <h>:
; CHECK:       goto
; CHECK:       exit
