; RUN: llc -O1 -o - %s | FileCheck %s

; Check that dbg.value(undef) calls are not simply discarded.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 42, metadata !11, metadata !DIExpression()), !dbg !13
  call void (...) @bar(), !dbg !14
  call void @llvm.dbg.value(metadata i32 undef, metadata !11, metadata !DIExpression()), !dbg !13
  call void (...) @bar(), !dbg !15
  ret void, !dbg !16
}

declare void @bar(...)

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 7.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 7.0.0"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: true, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "x", scope: !7, file: !1, line: 4, type: !12)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 4, column: 7, scope: !7)
!14 = !DILocation(line: 5, column: 3, scope: !7)
!15 = !DILocation(line: 7, column: 3, scope: !7)
!16 = !DILocation(line: 8, column: 1, scope: !7)

; Original C program:
;  void bar();
;
;  void foo() {
;    int x = 42;
;    bar();
;    x = 23;
;    bar();
;  }
;
; Then the dbg.value for the x = 23 setting has been replaced with dbg.value(undef)
; in this ll file to provoke the wanted behavior in LiveDebugVariables.

; Variable 'x' should thus be updated two times, first to 42 and then to undef.

; CHECK-LABEL: foo:
; CHECK:               #DEBUG_VALUE: foo:x <- 42
; CHECK:               callq   bar
; CHECK:               #DEBUG_VALUE: foo:x <- undef
; CHECK:               callq   bar
