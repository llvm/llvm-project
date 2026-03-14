; RUN: llvm-as < %s -o %t
; RUN: llvm-dis < %t -o - | FileCheck %s
; Created at -O1 from:
; int sink(int);
; __attribute__((always_inline)) int f(int i) { return sink(i); }
; __attribute__((always_inline)) int g(int j) { return sink(j); }
; __attribute__((nodebug)) int nodebug(int k) { return f(k)+g(k); }
source_filename = "t.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

declare i32 @sink(i32) local_unnamed_addr

define i32 @nodebug(i32 %k) local_unnamed_addr #2 {
entry:
; This should not set off the FnArg Verifier. The two variables are in differrent scopes.
  tail call void @llvm.dbg.value(metadata i32 %k, i64 0, metadata !12, metadata !13) #4, !dbg !14
  %call.k = tail call i32 @sink(i32 %k) #4, !dbg !15
  tail call void @llvm.dbg.value(metadata i32 %k, i64 0, metadata !19, metadata !13) #4, !dbg !20
  %call.k3 = tail call i32 @sink(i32 %k) #4, !dbg !21
  %add = add nsw i32 %call.k3, %call.k
  ret i32 %add
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #3

attributes #2 = { nounwind ssp uwtable }
attributes #3 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 5.0.0 (trunk 297153) (llvm/trunk 297155)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 5.0.0 (trunk 297153) (llvm/trunk 297155)"}
!7 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 2, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
; CHECK: !DILocalVariable(name: "i", arg: 1
!12 = !DILocalVariable(name: "i", arg: 1, scope: !7, file: !1, line: 2, type: !10)
!13 = !DIExpression()
!14 = !DILocation(line: 2, column: 42, scope: !7)
!15 = !DILocation(line: 2, column: 54, scope: !7)
!16 = !DILocation(line: 2, column: 47, scope: !7)
!17 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !18)
!18 = !{!19}
; CHECK: !DILocalVariable(name: "j", arg: 1
!19 = !DILocalVariable(name: "j", arg: 1, scope: !17, file: !1, line: 3, type: !10)
!20 = !DILocation(line: 3, column: 42, scope: !17)
!21 = !DILocation(line: 3, column: 54, scope: !17)
!22 = !DILocation(line: 3, column: 47, scope: !17)
