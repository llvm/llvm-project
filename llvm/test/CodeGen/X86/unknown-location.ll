; RUN: llc < %s -asm-verbose=false -mtriple=x86_64-apple-darwin10 -use-unknown-locations | FileCheck %s

; The divide instruction does not have a debug location. CodeGen should
; represent this in the debug information. This is done by setting line
; and column to 0

;      CHECK:         leal
; CHECK-NEXT:         .loc 1 0 0
;      CHECK:         cltd
; CHECK-NEXT:         idivl
; CHECK-NEXT:         .loc 1 4 3

define i32 @foo(i32 %w, i32 %x, i32 %y, i32 %z) nounwind !dbg !1 {
entry:
  %a = add  i32 %w, %x, !dbg !8
  %b = sdiv i32 %a, %y
  %c = add  i32 %b, %z, !dbg !8
  ret i32 %c, !dbg !8
}

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!12}

!0 = !DILocalVariable(name: "x", line: 1, arg: 2, scope: !1, file: !2, type: !6)
!1 = distinct !DISubprogram(name: "foo", linkageName: "foo", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !3, scopeLine: 1, file: !10, scope: !2, type: !4)
!2 = !DIFile(filename: "test.c", directory: "/dir")
!3 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "producer", isOptimized: false, emissionKind: FullDebug, file: !10, enums: !11, retainedTypes: !11)
!4 = !DISubroutineType(types: !5)
!5 = !{!6}
!6 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = distinct !DILexicalBlock(line: 1, column: 30, file: !10, scope: !1)
!8 = !DILocation(line: 4, column: 3, scope: !7)
!10 = !DIFile(filename: "test.c", directory: "/dir")
!11 = !{}
!12 = !{i32 1, !"Debug Info Version", i32 3}
