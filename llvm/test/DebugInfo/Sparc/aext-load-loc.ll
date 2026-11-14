; RUN: llc -O2 -disable-sparc-delay-filler %s -mtriple sparcv9 -o - | FileCheck %s

; An i32 return without signext/zeroext is any-extended to i64. When the
; DAG combiner folds that any_extend into the load feeding the return, the
; new load must keep the load's debug location rather than taking the
; return's. Otherwise a lexical block whose only instruction is the load is
; dropped from the DWARF.

; int get(int *p) {
;   ...
;   int v = p[1];
;   ...
;   return v;
; }

define i32 @get(ptr %p) !dbg !4 {
  ; CHECK-LABEL: get:
  ; CHECK:       .loc 1 8 7
  ; CHECK-NEXT:  ld [%o0+4], %o0
  ; CHECK:       .loc 1 10 2
  ; CHECK-NEXT:  retl
entry:
  %a = getelementptr inbounds i8, ptr %p, i64 4, !dbg !11
  %v = load i32, ptr %a, align 4, !dbg !11
  ret i32 %v, !dbg !12
}

; Same, but the load is already a zextload when the any_extend is folded.

define i32 @get_byte(ptr %p) !dbg !13 {
  ; CHECK-LABEL: get_byte:
  ; CHECK:       .loc 1 14 10
  ; CHECK-NEXT:  ldub [%o0], %o0
  ; CHECK:       .loc 1 15 3
  ; CHECK-NEXT:  retl
entry:
  %v = load i8, ptr %p, align 1, !dbg !14
  %z = zext i8 %v to i32, !dbg !14
  ret i32 %z, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "blk.c", directory: "/tmp")
!4 = distinct !DISubprogram(name: "get", scope: !1, file: !1, line: 7, type: !5, scopeLine: 7, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !DILocation(line: 8, column: 7, scope: !4)
!12 = !DILocation(line: 10, column: 2, scope: !4)
!13 = distinct !DISubprogram(name: "get_byte", scope: !1, file: !1, line: 13, type: !5, scopeLine: 13, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!14 = !DILocation(line: 14, column: 10, scope: !13)
!15 = !DILocation(line: 15, column: 3, scope: !13)
