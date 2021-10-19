;; A parameter list long enough to put one parameter on the stack, plus
;; at least one float parameter, triggered a corner case that messed up
;; setting prologue_end.
;;
;; Generated using -gmlt from this source:
;; void call_7i_1f (char c1, float f1, char c2, char c3, char c4, char c5, char c6, short s)
;; {
;;   c1 = 'a'; f1 = 0.1; c2 = 5; c3 = 6; c4 = 7;
;;   c5 = 's'; c6 = 'f'; s = 77;
;; }

; RUN: llc -mtriple x86_64-- -fast-isel < %s | FileCheck %s

define dso_local void @call_7i_1f(i8 signext %c1, float %f1, i8 signext %c2, i8 signext %c3, i8 signext %c4, i8 signext %c5, i8 signext %c6, i16 signext %s) !dbg !7 {
entry:
  %c1.addr = alloca i8, align 1
  %f1.addr = alloca float, align 4
  %c2.addr = alloca i8, align 1
  %c3.addr = alloca i8, align 1
  %c4.addr = alloca i8, align 1
  %c5.addr = alloca i8, align 1
  %c6.addr = alloca i8, align 1
  %s.addr = alloca i16, align 2
  store i8 %c1, i8* %c1.addr, align 1
  store float %f1, float* %f1.addr, align 4
  store i8 %c2, i8* %c2.addr, align 1
  store i8 %c3, i8* %c3.addr, align 1
  store i8 %c4, i8* %c4.addr, align 1
  store i8 %c5, i8* %c5.addr, align 1
  store i8 %c6, i8* %c6.addr, align 1
  store i16 %s, i16* %s.addr, align 2
  store i8 97, i8* %c1.addr, align 1, !dbg !9
  store float 0x3FB99999A0000000, float* %f1.addr, align 4, !dbg !10
  store i8 5, i8* %c2.addr, align 1, !dbg !11
  store i8 6, i8* %c3.addr, align 1, !dbg !12
  store i8 7, i8* %c4.addr, align 1, !dbg !13
  store i8 115, i8* %c5.addr, align 1, !dbg !14
  store i8 102, i8* %c6.addr, align 1, !dbg !15
  store i16 77, i16* %s.addr, align 2, !dbg !16
  ret void, !dbg !17
}

;; All incoming parameter registers should be homed as part of the prologue.
; CHECK-DAG: %dil
; CHECK-DAG: %xmm0
; CHECK-DAG: %sil
; CHECK-DAG: %dl
; CHECK-DAG: %cl
; CHECK-DAG: %r8b
; CHECK-DAG: %r9b
; CHECK:     prologue_end

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "fast-isel-prolog-dbgloc.c", directory: "/home/probinson/projects/scratch")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0"}
!7 = distinct !DISubprogram(name: "call_7i_1f", scope: !1, file: !1, line: 3, type: !8, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 5, column: 6, scope: !7)
!10 = !DILocation(line: 5, column: 16, scope: !7)
!11 = !DILocation(line: 5, column: 26, scope: !7)
!12 = !DILocation(line: 5, column: 34, scope: !7)
!13 = !DILocation(line: 5, column: 42, scope: !7)
!14 = !DILocation(line: 6, column: 6, scope: !7)
!15 = !DILocation(line: 6, column: 16, scope: !7)
!16 = !DILocation(line: 6, column: 25, scope: !7)
!17 = !DILocation(line: 7, column: 1, scope: !7)
