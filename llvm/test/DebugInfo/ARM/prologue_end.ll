; RUN: llc -disable-fp-elim -O0 %s -mtriple armv7-apple-darwin -o - | FileCheck %s
; RUN: llc -disable-fp-elim -O0 %s -mtriple thumbv1-apple-darwin -o - | FileCheck %s

; int func(void);
; void prologue_end_test() {
;   func();
;   func();
; }

define void @prologue_end_test() nounwind uwtable !dbg !4 {
  ; CHECK: prologue_end_test:
  ; CHECK: .cfi_startproc
  ; CHECK: push {r7, lr}
  ; CHECK: {{mov r7, sp|add r7, sp}}
  ; CHECK: sub sp
  ; CHECK: .loc 1 3 3 prologue_end
  ; CHECK: bl {{_func|Ltmp}}
  ; CHECK: bl {{_func|Ltmp}}
entry:
  %call = call i32 @func(), !dbg !13
  %call1 = call i32 @func(), !dbg !14
  ret void, !dbg !15
}

declare i32 @func()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8, !9, !10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.7.0 (trunk 242129)", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !2, subprograms: !3)
!1 = !DIFile(filename: "foo.c", directory: "/tmp")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "prologue_end_test", scope: !1, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, variables: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{i32 2, !"Dwarf Version", i32 2}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 1, !"min_enum_size", i32 4}
!11 = !{i32 1, !"PIC Level", i32 2}
!12 = !{!"clang version 3.7.0 (trunk 242129)"}
!13 = !DILocation(line: 3, column: 3, scope: !4)
!14 = !DILocation(line: 4, column: 3, scope: !4)
!15 = !DILocation(line: 5, column: 1, scope: !4)
