; RUN: llc < %s -O0 -verify-machineinstrs | FileCheck %s

; Test verifies if .loc is properly set for loop and block instructions --
; the line number is not a 0 when DebugLoc is defined.

; Compiled from the dbg-loop-loc.c:
; int fib(int n) {
;   int i, t, a = 0, b = 1;
;   for (i = 0; i < n; i++) {
;     t = a + b; a = b; b = t;
;   }
;   return b;
; }

; ModuleID = 'dbg-loop-loc.bc'
source_filename = "dbg-loop-loc.c"
target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK: .file 1
; CHECK: .loc 1 1 0
; CHECK: .loc 1 3 15
; CHECK-NEXT: block
; CHECK-NEXT: loop
; CHECK: br 0
; CHECK-NOT: .loc 1 0
; CHECK: end_loop
; CHECK: end_block
; CHECK-NOT: .loc 1 0
; CHECK: end_function

; Function Attrs: noinline nounwind optnone
define hidden i32 @fib(i32 %n) #0 !dbg !7 {
entry:
  %n.addr = alloca i32, align 4
  %i = alloca i32, align 4
  %t = alloca i32, align 4
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %n.addr, metadata !11, metadata !DIExpression()), !dbg !12
  call void @llvm.dbg.declare(metadata ptr %i, metadata !13, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata ptr %t, metadata !15, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata ptr %a, metadata !17, metadata !DIExpression()), !dbg !18
  store i32 0, ptr %a, align 4, !dbg !18
  call void @llvm.dbg.declare(metadata ptr %b, metadata !19, metadata !DIExpression()), !dbg !20
  store i32 1, ptr %b, align 4, !dbg !20
  store i32 0, ptr %i, align 4, !dbg !21
  br label %for.cond, !dbg !23

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4, !dbg !24
  %1 = load i32, ptr %n.addr, align 4, !dbg !26
  %cmp = icmp slt i32 %0, %1, !dbg !27
  br i1 %cmp, label %for.body, label %for.end, !dbg !28

for.body:                                         ; preds = %for.cond
  %2 = load i32, ptr %a, align 4, !dbg !29
  %3 = load i32, ptr %b, align 4, !dbg !31
  %add = add nsw i32 %2, %3, !dbg !32
  store i32 %add, ptr %t, align 4, !dbg !33
  %4 = load i32, ptr %b, align 4, !dbg !34
  store i32 %4, ptr %a, align 4, !dbg !35
  %5 = load i32, ptr %t, align 4, !dbg !36
  store i32 %5, ptr %b, align 4, !dbg !37
  br label %for.inc, !dbg !38

for.inc:                                          ; preds = %for.body
  %6 = load i32, ptr %i, align 4, !dbg !39
  %inc = add nsw i32 %6, 1, !dbg !39
  store i32 %inc, ptr %i, align 4, !dbg !39
  br label %for.cond, !dbg !40, !llvm.loop !41

for.end:                                          ; preds = %for.cond
  %7 = load i32, ptr %b, align 4, !dbg !43
  ret i32 %7, !dbg !44
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 7.0.0 (trunk 326837)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "dbg-loop-loc.c", directory: "/Users/yury/llvmwasm")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 7.0.0 (trunk 326837)"}
!7 = distinct !DISubprogram(name: "fib", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocalVariable(name: "n", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!12 = !DILocation(line: 1, column: 13, scope: !7)
!13 = !DILocalVariable(name: "i", scope: !7, file: !1, line: 2, type: !10)
!14 = !DILocation(line: 2, column: 7, scope: !7)
!15 = !DILocalVariable(name: "t", scope: !7, file: !1, line: 2, type: !10)
!16 = !DILocation(line: 2, column: 10, scope: !7)
!17 = !DILocalVariable(name: "a", scope: !7, file: !1, line: 2, type: !10)
!18 = !DILocation(line: 2, column: 13, scope: !7)
!19 = !DILocalVariable(name: "b", scope: !7, file: !1, line: 2, type: !10)
!20 = !DILocation(line: 2, column: 20, scope: !7)
!21 = !DILocation(line: 3, column: 10, scope: !22)
!22 = distinct !DILexicalBlock(scope: !7, file: !1, line: 3, column: 3)
!23 = !DILocation(line: 3, column: 8, scope: !22)
!24 = !DILocation(line: 3, column: 15, scope: !25)
!25 = distinct !DILexicalBlock(scope: !22, file: !1, line: 3, column: 3)
!26 = !DILocation(line: 3, column: 19, scope: !25)
!27 = !DILocation(line: 3, column: 17, scope: !25)
!28 = !DILocation(line: 3, column: 3, scope: !22)
!29 = !DILocation(line: 4, column: 9, scope: !30)
!30 = distinct !DILexicalBlock(scope: !25, file: !1, line: 3, column: 27)
!31 = !DILocation(line: 4, column: 13, scope: !30)
!32 = !DILocation(line: 4, column: 11, scope: !30)
!33 = !DILocation(line: 4, column: 7, scope: !30)
!34 = !DILocation(line: 4, column: 20, scope: !30)
!35 = !DILocation(line: 4, column: 18, scope: !30)
!36 = !DILocation(line: 4, column: 27, scope: !30)
!37 = !DILocation(line: 4, column: 25, scope: !30)
!38 = !DILocation(line: 5, column: 3, scope: !30)
!39 = !DILocation(line: 3, column: 23, scope: !25)
!40 = !DILocation(line: 3, column: 3, scope: !25)
!41 = distinct !{!41, !28, !42}
!42 = !DILocation(line: 5, column: 3, scope: !22)
!43 = !DILocation(line: 6, column: 10, scope: !7)
!44 = !DILocation(line: 6, column: 3, scope: !7)
