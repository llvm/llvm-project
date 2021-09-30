; Test that BDCE doesn't destroy llvm.dbg.value's argument.
; RUN: opt -passes=bdce %s -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: define void @f()
; CHECK-NEXT: entry:
; CHECK-NEXT: tail call void (...) @h()
; CHECK-NEXT: %[[CALL:.*]] = tail call i32 (...) @g()
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata i32 %[[CALL:.*]]

define void @f() !dbg !6 {
entry:
  tail call void (...) @h(), !dbg !9
  %call = tail call i32 (...) @g(), !dbg !10
  tail call void @llvm.dbg.value(metadata i32 %call, metadata !11, metadata !13), !dbg !14
  %patatino = xor i32 %call, %call
  tail call void (...) @h(), !dbg !15
  ret void, !dbg !16
}

declare void @h(...)
declare i32 @g(...)
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0 (trunk 288665) (llvm/trunk 288725)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "patatino.c", directory: "/home/davide/work/llvm/build-clang/bin")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 4.0.0 (trunk 288665) (llvm/trunk 288725)"}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 3, type: !7, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: false, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !DILocation(line: 4, column: 3, scope: !6)
!10 = !DILocation(line: 5, column: 11, scope: !6)
!11 = !DILocalVariable(name: "a", scope: !6, file: !1, line: 5, type: !12)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DIExpression()
!14 = !DILocation(line: 5, column: 7, scope: !6)
!15 = !DILocation(line: 6, column: 3, scope: !6)
!16 = !DILocation(line: 7, column: 1, scope: !6)
