; RUN: opt -mtriple=aarch64-unknown-linux-gnu -S %s -passes=sroa -o - | FileCheck %s

; In this test we want to ensure that getMergedLocations uses common include
; location if incoming locations belong to different files.
; The location of phi instruction merged from locations of %mul3 and %mul10
; should be the location of do-loop lexical block from y.c.

; Generated with clang from
;
; main.c:
;   int foo(int a) {
;     int i = 0;
;     if ((a & 1) == 1) {
;       a -= 1;
;   #define A
;   #include "y.c"
;    } else {
;       a += 3;
;   #undef A
;   #include "y.c"
;    }
;     return i;
;   }
;
; y.c:
;   # 300 "y.c" 1
;   do {
;   #ifdef A
;   #include "z1.c"
;   #else
;   #include "z2.c"
;   #endif
;   } while (0);
;
; z1.c:
;   # 100 "z1.c" 1
;   i += a;
;   i -= 10*a;
;   i *= a*a;
;
; z2.c:
;   # 200 "z1.c" 1
;   i += a;
;   i -= 10*a;
;   i *= a*a;
;
; Preprocessed source:
;
; # 1 "main.c"
; int foo(int a) {
;   int i = 0;
;   if ((a & 1) == 1) {
;     a -= 1;
; # 300 "y.c" 1
; do {
; # 100 "z1.c" 1
; i += a;
; i -= 10*a;
; i *= a*a;
; # 303 "y.c" 2
; } while (0);
; # 7 "main.c" 2
;  } else {
;     a += 3;
; # 300 "y.c" 1
; do {
; # 200 "z2.c" 1
; i += a;
; i -= 10*a;
; i *= a*a;
; # 305 "y.c" 2
; } while (0);
; # 11 "main.c" 2
;  }
;   return i;
; }

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx15.0.0"

define i32 @foo() !dbg !3 {
; CHECK:    phi i32 {{.*}}, !dbg [[PHILOC:![0-9]+]]
;
entry:
  %i = alloca i32, align 4
  br i1 false, label %do.body, label %if.else

do.body:                                          ; preds = %entry
  store i32 1, ptr %i, align 4, !dbg !6
  br label %if.end

if.else:                                          ; preds = %entry
  store i32 0, ptr %i, align 4, !dbg !14
  br label %if.end

if.end:                                           ; preds = %if.else, %do.body
  %0 = load i32, ptr %i, align 4
  ret i32 0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 21.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!1 = !DIFile(filename: "main.c", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !5)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !DILocation(line: 102, column: 3, scope: !7)
!7 = !DILexicalBlockFile(scope: !9, file: !8, discriminator: 0)
!8 = !DIFile(filename: "z1.c", directory: "")
!9 = distinct !DILexicalBlock(scope: !11, file: !10, line: 300, column: 4)
!10 = !DIFile(filename: "y.c", directory: "")
!11 = !DILexicalBlockFile(scope: !12, file: !10, discriminator: 0)
!12 = distinct !DILexicalBlock(scope: !13, file: !1, line: 3, column: 21)
!13 = distinct !DILexicalBlock(scope: !3, file: !1, line: 3, column: 7)
!14 = !DILocation(line: 202, column: 3, scope: !15)
!15 = !DILexicalBlockFile(scope: !17, file: !16, discriminator: 0)
!16 = !DIFile(filename: "z2.c", directory: "")
!17 = distinct !DILexicalBlock(scope: !18, file: !10, line: 300, column: 4)
!18 = !DILexicalBlockFile(scope: !19, file: !10, discriminator: 0)
!19 = distinct !DILexicalBlock(scope: !13, file: !1, line: 7, column: 9)

; CHECK: [[FILE_MAIN:![0-9]+]] = !DIFile(filename: "main.c"
; CHECK: [[SP:![0-9]+]] = distinct !DISubprogram(name: "foo", scope: [[FILE_MAIN]], file: [[FILE_MAIN]], line: 1
; CHECK: [[PHILOC]] = !DILocation(line: 300, column: 4, scope: [[BLOCK_Y:![0-9]+]])
; CHECK-NEXT: [[BLOCK_Y]] = !DILexicalBlock(scope: [[BLOCK_MAIN:![0-9]+]], file: [[FILE_Y:![0-9]+]], line: 300, column: 4)
; CHECK-NEXT: [[FILE_Y]] = !DIFile(filename: "y.c"
; CHECK: [[BLOCK_MAIN]] = distinct !DILexicalBlock(scope: [[SP]], file: [[FILE_MAIN]], line: 3, column: 7)
