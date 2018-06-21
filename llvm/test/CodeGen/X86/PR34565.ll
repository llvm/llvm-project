; RUN: llc -mtriple=x86_64-pc-linux -x86-cmov-converter=true -verify-machineinstrs < %s | FileCheck %s

; Test for PR34565, check that DBG instructions are ignored while optimizing
; X86 CMOV instructions.
; In this case, we check that there is no 'cmov' generated.

; CHECK-NOT: cmov

@main.buf = private unnamed_addr constant [10 x i64] [i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9], align 8

define i32 @main() #0 !dbg !5 {
entry:
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %a.010 = phi i32 [ 0, %entry ], [ %add.a.0, %while.body ]
  %b.09 = phi i32 [ 10, %entry ], [ %b.0.add, %while.body ]
  %add = add i32 %a.010, %b.09
  %call = tail call i32 @rand()
  %conv = sext i32 %call to i64
  %arrayidx = getelementptr inbounds [10 x i64], [10 x i64]* @main.buf, i32 0, i32 %add
  %0 = load i64, i64* %arrayidx, align 8
  %cmp1 = icmp ult i64 %0, %conv
  %b.0.add = select i1 %cmp1, i32 %b.09, i32 %add
  %add.a.0 = select i1 %cmp1, i32 %add, i32 %a.010
  tail call void @llvm.dbg.value(metadata i32 %add.a.0, metadata !10, metadata !DIExpression()), !dbg !13
  tail call void @llvm.dbg.value(metadata i32 %b.0.add, metadata !12, metadata !DIExpression()), !dbg !14
  tail call void @llvm.dbg.value(metadata i32 %add.a.0, metadata !10, metadata !DIExpression()), !dbg !13
  tail call void @llvm.dbg.value(metadata i32 %b.0.add, metadata !12, metadata !DIExpression()), !dbg !14
  %cmp = icmp ult i32 %add.a.0, %b.0.add
  br i1 %cmp, label %while.body, label %while.end

while.end:                                        ; preds = %while.body
  ret i32 0
}

declare i32 @rand()

declare void @llvm.dbg.value(metadata, metadata, metadata)

attributes #0 = { "target-cpu"="x86-64" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 (trunk)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "PR34565.c", directory: "\5C")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 3, type: !6, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !9)
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{!10, !12}
!10 = !DILocalVariable(name: "a", scope: !5, file: !1, line: 6, type: !11)
!11 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!12 = !DILocalVariable(name: "b", scope: !5, file: !1, line: 7, type: !11)
!13 = !DILocation(line: 6, column: 16, scope: !5)
!14 = !DILocation(line: 7, column: 16, scope: !5)
