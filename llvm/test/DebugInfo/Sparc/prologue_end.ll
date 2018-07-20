; RUN: llc -disable-fp-elim -O0 %s -mtriple sparc -o - | FileCheck %s

; int func(void);
; void prologue_end_test() {
;   func();
;   func();
; }

define void @prologue_end_test() nounwind uwtable !dbg !4 {
  ; CHECK: prologue_end_test:
  ; CHECK: .cfi_startproc
  ; CHECK: save %sp
  ; CHECK: .loc 1 3 3 prologue_end
  ; CHECK: call func
  ; CHECK: call func
entry:
  %call = call i32 @func(), !dbg !11
  %call1 = call i32 @func(), !dbg !12
  ret void, !dbg !13
}

declare i32 @func()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.7.0 (trunk 242129)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "foo.c", directory: "/tmp")
!2 = !{}
!4 = distinct !DISubprogram(name: "prologue_end_test", scope: !1, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{i32 2, !"Dwarf Version", i32 2}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"PIC Level", i32 2}
!10 = !{!"clang version 3.7.0 (trunk 242129)"}
!11 = !DILocation(line: 3, column: 3, scope: !4)
!12 = !DILocation(line: 4, column: 3, scope: !4)
!13 = !DILocation(line: 5, column: 1, scope: !4)
