; RUN: llc -O0 -disable-sparc-delay-filler %s -mtriple sparcv9 -o - | FileCheck %s

; The stack adjustment in a leaf epilogue belongs to the return's line, so
; a breakpoint on that line stops before the frame is released.

; long leaf(long x) {
;   return x;
; }

define i64 @leaf(i64 %x) !dbg !4 {
  ; CHECK-LABEL: leaf:
  ; CHECK:       add %sp, -{{[0-9]+}}, %sp
  ; CHECK:       stx %o0,
  ; CHECK:       .loc 1 3 3 prologue_end
  ; CHECK-NEXT:  add %sp, {{[0-9]+}}, %sp
  ; CHECK-NEXT:  retl
entry:
  %x.addr = alloca i64, align 8
  store i64 %x, ptr %x.addr, align 8
  ret i64 %x, !dbg !11
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "foo.c", directory: "/tmp")
!4 = distinct !DISubprogram(name: "leaf", scope: !1, file: !1, line: 2, type: !5, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !DILocation(line: 3, column: 3, scope: !4)
