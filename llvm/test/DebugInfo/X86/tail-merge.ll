; RUN: llc %s -mtriple=x86_64-unknown-unknown -use-unknown-locations=Enable -o - | FileCheck %s

; Generated with "clang -gline-tables-only -c -emit-llvm -o - | opt -sroa -S"
; from source:
;
; extern int foo(int);
; extern int bar(int);
;
; int test(int a, int b) {
;   if(b)
;     a += foo(a);
;   else
;     a += bar(a);
;   return a;
; }

; When tail-merging the debug location of the common tail should be removed.

; CHECK-LABEL: test:
; CHECK: movl	%edi, [[REG:%.*]]
; CHECK: testl	%esi, %esi
; CHECK: je	[[ELSE:.LBB[0-9]+_[0-9]+]]
; CHECK: .loc	1 6 10
; CHECK: callq	foo
; CHECK: jmp	[[TAIL:.LBB[0-9]+_[0-9]+]]
; CHECK: [[ELSE]]:
; CHECK: .loc	1 8 10
; CHECK: callq	bar
; CHECK: [[TAIL]]:
; CHECK: .loc	1 0
; CHECK: addl	[[REG]], %eax
; CHECK: .loc	1 9 3

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @test(i32 %a, i32 %b) !dbg !6 {
entry:
  %tobool = icmp ne i32 %b, 0, !dbg !8
  br i1 %tobool, label %if.then, label %if.else, !dbg !8

if.then:                                          ; preds = %entry
  %call = call i32 @foo(i32 %a), !dbg !9
  %add = add nsw i32 %a, %call, !dbg !10
  br label %if.end, !dbg !11

if.else:                                          ; preds = %entry
  %call1 = call i32 @bar(i32 %a), !dbg !12
  %add2 = add nsw i32 %a, %call1, !dbg !13
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %a.addr.0 = phi i32 [ %add, %if.then ], [ %add2, %if.else ]
  ret i32 %a.addr.0, !dbg !14
}

declare i32 @foo(i32)
declare i32 @bar(i32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 4, type: !7, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 5, column: 6, scope: !6)
!9 = !DILocation(line: 6, column: 10, scope: !6)
!10 = !DILocation(line: 6, column: 7, scope: !6)
!11 = !DILocation(line: 6, column: 5, scope: !6)
!12 = !DILocation(line: 8, column: 10, scope: !6)
!13 = !DILocation(line: 8, column: 7, scope: !6)
!14 = !DILocation(line: 9, column: 3, scope: !6)
