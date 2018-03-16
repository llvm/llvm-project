; RUN: llc -disable-fp-elim -O0 %s -mtriple x86_64-unknown-linux-gnu -o - | FileCheck %s

; int callme(int);
; int isel_line_test2() {
;   callme(400);
;   return 0;
; }

define i32 @isel_line_test2() nounwind uwtable !dbg !5 {
  ; The stack adjustment should be part of the prologue.
  ; CHECK: isel_line_test2:
  ; CHECK: {{subq|leaq}} {{.*}}, %rsp
  ; CHECK: .loc 1 5 3 prologue_end
  ; CHECK: movl $400, %edi
  ; CHECK: callq callme
entry:
  %call = call i32 @callme(i32 400), !dbg !10
  ret i32 0, !dbg !12
}

declare i32 @callme(i32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.2 (trunk 164980) (llvm/trunk 164979)", isOptimized: false, emissionKind: FullDebug, file: !13, enums: !1, retainedTypes: !1, globals: !1, imports:  !1)
!1 = !{}
!5 = distinct !DISubprogram(name: "isel_line_test2", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !0, scopeLine: 4, file: !13, scope: !6, type: !7, variables: !1)
!6 = !DIFile(filename: "bar.c", directory: "/usr/local/google/home/echristo/tmp")
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 5, column: 3, scope: !11)
!11 = distinct !DILexicalBlock(line: 4, column: 1, file: !13, scope: !5)
!12 = !DILocation(line: 6, column: 3, scope: !11)
!13 = !DIFile(filename: "bar.c", directory: "/usr/local/google/home/echristo/tmp")
!14 = !{i32 1, !"Debug Info Version", i32 3}
