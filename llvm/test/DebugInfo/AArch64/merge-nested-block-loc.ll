; RUN: opt -mtriple=aarch64-unknown-linux-gnu -S %s -passes=sroa -o - | FileCheck %s

; In this test we want to ensure that the location of phi instruction merged from locations
; of %mul3 and %mul9 belongs to the correct scope (DILexicalBlockFile), so that line
; number of that location belongs to the corresponding file.

; Generated with clang from
; # 1 "1.c" 1
; # 1 "1.c" 2
; int foo(int a) {
;   int i = 0;
;   if ((a & 1) == 1) {
;     a -= 1;
; # 1 "m.c" 1
; # 40 "m.c"
; i += a;
; i -= 10*a;
; i *= a*a;
; # 6 "1.c" 2
;  } else {
;     a += 3;
; # 1 "m.c" 1
; # 40 "m.c"
; i += a;
; i -= 10*a;
; i *= a*a;
; # 9 "1.c" 2
;  }
;   return i;
; }

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

define i32 @foo() !dbg !3 {
; CHECK:    phi i32 {{.*}}, !dbg [[PHILOC:![0-9]+]]
;
entry:
  %i = alloca i32, align 4
  br i1 false, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %mul3 = mul i32 0, 0, !dbg !14
  store i32 %mul3, ptr %i, align 4, !dbg !14
  br label %if.end

if.else:                                          ; preds = %entry
  %mul9 = mul i32 0, 0, !dbg !15
  store i32 %mul9, ptr %i, align 4, !dbg !15
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %0 = load i32, ptr %i, align 4
  ret i32 0
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 21.0.0git", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "repro.c", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "foo", scope: !4, file: !4, line: 1, type: !5, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "1.c", directory: "")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 3, column: 10, scope: !8)
!8 = distinct !DILexicalBlock(scope: !3, file: !4, line: 3, column: 7)
!9 = !DILocation(line: 4, column: 7, scope: !10)
!10 = distinct !DILexicalBlock(scope: !8, file: !4, line: 3, column: 21)
!11 = !DILocation(line: 40, column: 3, scope: !12)
!12 = !DILexicalBlockFile(scope: !10, file: !13, discriminator: 0)
!13 = !DIFile(filename: "m.c", directory: "")
!14 = !DILocation(line: 42, column: 3, scope: !12)
!15 = !DILocation(line: 42, column: 3, scope: !16)
!16 = !DILexicalBlockFile(scope: !17, file: !13, discriminator: 0)
!17 = distinct !DILexicalBlock(scope: !8, file: !4, line: 6, column: 9)

; CHECK: [[SP:![0-9]+]] = distinct !DISubprogram(name: "foo", scope: [[FILE1:![0-9]+]], file: [[FILE1]], line: 1
; CHECK: [[FILE1]] = !DIFile(filename: "1.c", directory: "")
; CHECK: [[PHILOC]] = !DILocation(line: 42, column: 3, scope: [[LBF:![0-9]+]])
; CHECK: [[LBF]] = !DILexicalBlockFile(scope: [[LB1:![0-9]+]], file: [[FILE2:![0-9]+]], discriminator: 0)
; CHECK: [[FILE2]] = !DIFile(filename: "m.c", directory: "")
; CHECK: [[LB1]] = distinct !DILexicalBlock(scope: [[LB2:![0-9]+]], file: [[FILE1]], line: 3, column: 21)
; CHECK: [[LB2]] = distinct !DILexicalBlock(scope: [[SP]], file: [[FILE1]], line: 3, column: 7)
