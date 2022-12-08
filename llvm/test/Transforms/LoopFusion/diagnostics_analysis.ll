; RUN: opt -S -passes=loop-fusion -pass-remarks-analysis=loop-fusion -disable-output < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

@B = common global [1024 x i32] zeroinitializer, align 16

; CHECK: remark: diagnostics_analysis.c:6:3: [test]: Loop is not a candidate for fusion: Loop contains a volatile access
; CHECK: remark: diagnostics_analysis.c:10:3: [test]: Loop is not a candidate for fusion: Loop has unknown trip count
define void @test(i32* %A, i32 %n) !dbg !15 {
entry:
  %A.addr = alloca i32*, align 8
  %n.addr = alloca i32, align 4
  %i = alloca i32, align 4
  %i1 = alloca i32, align 4
  store i32* %A, i32** %A.addr, align 8
  store i32 %n, i32* %n.addr, align 4
  %0 = bitcast i32* %i to i8*
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %1 = load i32, i32* %i, align 4
  %2 = load i32, i32* %n.addr, align 4
  %cmp = icmp slt i32 %1, %2
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  %3 = bitcast i32* %i to i8*, !dbg !42
  br label %for.end

for.body:                                         ; preds = %for.cond
  %4 = load i32, i32* %i, align 4
  %sub = sub nsw i32 %4, 3
  %5 = load i32, i32* %i, align 4
  %add = add nsw i32 %5, 3
  %mul = mul nsw i32 %sub, %add
  %6 = load i32, i32* %i, align 4
  %rem = srem i32 %mul, %6
  %7 = load i32*, i32** %A.addr, align 8
  %8 = load i32, i32* %i, align 4
  %idxprom = sext i32 %8 to i64
  %arrayidx = getelementptr inbounds i32, i32* %7, i64 %idxprom
  store volatile i32 %rem, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %9 = load i32, i32* %i, align 4, !dbg !49
  %inc = add nsw i32 %9, 1, !dbg !49
  store i32 %inc, i32* %i, align 4, !dbg !49
  br label %for.cond, !dbg !42, !llvm.loop !50

for.end:                                          ; preds = %for.cond.cleanup
  %10 = bitcast i32* %i1 to i8*
  store i32 0, i32* %i1, align 4
  br label %for.cond2

for.cond2:                                        ; preds = %for.inc12, %for.end
  %11 = load i32, i32* %i1, align 4
  %12 = load i32, i32* %n.addr, align 4
  %cmp3 = icmp slt i32 %11, %12
  br i1 %cmp3, label %for.body5, label %for.cond.cleanup4

for.cond.cleanup4:                                ; preds = %for.cond2
  %13 = bitcast i32* %i1 to i8*
  br label %for.end14

for.body5:                                        ; preds = %for.cond2
  %14 = load i32, i32* %i1, align 4
  %sub6 = sub nsw i32 %14, 3
  %15 = load i32, i32* %i1, align 4
  %add7 = add nsw i32 %15, 3
  %mul8 = mul nsw i32 %sub6, %add7
  %16 = load i32, i32* %i1, align 4
  %rem9 = srem i32 %mul8, %16
  %17 = load i32, i32* %i1, align 4
  %idxprom10 = sext i32 %17 to i64
  %arrayidx11 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %idxprom10
  store i32 %rem9, i32* %arrayidx11, align 4
  br label %for.inc12

for.inc12:                                        ; preds = %for.body5
  %18 = load i32, i32* %i1, align 4
  %inc13 = add nsw i32 %18, 1
  store i32 %inc13, i32* %i1, align 4
  br label %for.cond2, !dbg !59, !llvm.loop !67

for.end14:                                        ; preds = %for.cond.cleanup4
  ret void
}

!llvm.module.flags = !{!10, !11, !13}
!llvm.ident = !{!14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "B", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (git@github.ibm.com:compiler/llvm-project.git c019c32c5a2b0ed4487a738337d35fd3f630ac0a)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: GNU)
!3 = !DIFile(filename: "diagnostics_analysis.c", directory: "/tmp")
!4 = !{}
!5 = !{!0}
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 32768, elements: !8)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{!9}
!9 = !DISubrange(count: 1024)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 7, !"PIC Level", i32 2}
!14 = !{!"clang version 9.0.0 (git@github.ibm.com:compiler/llvm-project.git c019c32c5a2b0ed4487a738337d35fd3f630ac0a)"}
!15 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 5, type: !16, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !20)
!16 = !DISubroutineType(types: !17)
!17 = !{null, !18, !7}
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !7)
!20 = !{!21, !22, !23, !25}
!21 = !DILocalVariable(name: "A", arg: 1, scope: !15, file: !3, line: 5, type: !18)
!22 = !DILocalVariable(name: "n", arg: 2, scope: !15, file: !3, line: 5, type: !7)
!23 = !DILocalVariable(name: "i", scope: !24, file: !3, line: 6, type: !7)
!24 = distinct !DILexicalBlock(scope: !15, file: !3, line: 6, column: 3)
!25 = !DILocalVariable(name: "i", scope: !26, file: !3, line: 10, type: !7)
!26 = distinct !DILexicalBlock(scope: !15, file: !3, line: 10, column: 3)
!38 = distinct !DILexicalBlock(scope: !24, file: !3, line: 6, column: 3)
!41 = !DILocation(line: 6, column: 3, scope: !24)
!42 = !DILocation(line: 6, column: 3, scope: !38)
!44 = distinct !DILexicalBlock(scope: !38, file: !3, line: 6, column: 31)
!49 = !DILocation(line: 6, column: 27, scope: !38)
!50 = distinct !{!50, !41, !51}
!51 = !DILocation(line: 8, column: 3, scope: !24)
!55 = distinct !DILexicalBlock(scope: !26, file: !3, line: 10, column: 3)
!58 = !DILocation(line: 10, column: 3, scope: !26)
!59 = !DILocation(line: 10, column: 3, scope: !55)
!67 = distinct !{!67, !58, !68}
!68 = !DILocation(line: 12, column: 3, scope: !26)
!69 = !DILocation(line: 13, column: 1, scope: !15)
