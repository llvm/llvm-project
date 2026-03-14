; RUN: llc < %s -O0 -stop-after=livedebugvalues -o - | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

define void @foo(ptr %p) !dbg !4 {
bb:
  %tmp = load i32, ptr %p, align 4, !dbg !7
  ; CHECK: $eax = MOV32rm killed {{.*}} $rdi, {{.*}} debug-location !7 :: (load (s32) from %ir.p)
  ; CHECK-NEXT: $rax = KILL killed renamable $eax, debug-location !7
  ; CHECK-NEXT: MOV64mr $rsp, 1, $noreg, -8, $noreg, $rax :: (store (s64) into %stack.0)
  ; CHECK-NEXT: SUB64ri32 renamable $rax, 3, implicit-def $eflags, debug-location !7

  switch i32 %tmp, label %bb7 [
    i32 0, label %bb1
    i32 1, label %bb2
    i32 2, label %bb3
    i32 3, label %bb4
  ], !dbg !8

bb1:                                              ; preds = %bb
  unreachable

bb2:                                              ; preds = %bb
  unreachable

bb3:                                              ; preds = %bb
  unreachable

bb4:                                              ; preds = %bb
  unreachable

bb7:                                              ; preds = %bb
  ret void, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 7.0.0 (trunk 330770) (llvm/trunk 330769)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "/Users/vsk/src/llvm.org-main/llvm/lib/Demangle/ItaniumDemangle.cpp", directory: "/Users/vsk/src/builds/llvm.org-main-RA-stage2")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"PIC Level", i32 2}
!4 = distinct !DISubprogram(name: "printLeft", scope: !1, file: !1, line: 1306, type: !5, isLocal: true, isDefinition: true, scopeLine: 1306, flags: DIFlagPrototyped, isOptimized: true, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 1307, column: 13, scope: !4)
!8 = !DILocation(line: 1307, column: 5, scope: !4)
!9 = !DILocation(line: 1327, column: 3, scope: !4)
