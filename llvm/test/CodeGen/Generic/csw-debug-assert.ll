; Check that compiler doesn't crash when calculateSpillWeightsAndHints is called with dbg instrs present
; REQUIRES: asserts
; REQUIRES: x86_64-linux
; RUN: llc -O1 -regalloc=pbqp < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: test:
define dso_local void @test(i32* %a) local_unnamed_addr #0 !dbg !7 {
entry:
  ; CHECK: DEBUG_VALUE: i <- 0
  call void @llvm.dbg.value(metadata i32* %a, metadata !14, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !17
  br label %for.cond, !dbg !17

for.cond:                                         ; preds = %for.body, %entry
  %a.addr.0 = phi i32* [ %a, %entry ], [ %incdec.ptr, %for.body ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ], !dbg !17
  call void @llvm.dbg.value(metadata i32 %i.0, metadata !15, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32* %a.addr.0, metadata !14, metadata !DIExpression()), !dbg !17
  %cmp = icmp slt i32 %i.0, 32, !dbg !17
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !17

for.cond.cleanup:                                 ; preds = %for.cond
  ret void, !dbg !17

for.body:                                         ; preds = %for.cond
  %incdec.ptr = getelementptr inbounds i32, i32* %a.addr.0, i32 1, !dbg !17
  call void @llvm.dbg.value(metadata i32* %incdec.ptr, metadata !14, metadata !DIExpression()), !dbg !17
  store i32 42, i32* %a.addr.0, align 4, !dbg !17
  %inc = add nsw i32 %i.0, 1, !dbg !17
  call void @llvm.dbg.value(metadata i32 %inc, metadata !15, metadata !DIExpression()), !dbg !17
  br label %for.cond, !dbg !17
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable optnone noinline }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git afaeb817468d2fdc0a315a7ff136db245e59a8eb)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "csw-debug-assert.c", directory: "dir")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git afaeb817468d2fdc0a315a7ff136db245e59a8eb)"}
!7 = distinct !DISubprogram(name: "test", scope: !8, file: !8, line: 5, type: !9, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!8 = !DIFile(filename: "csw-debug-assert.c", directory: "dir")
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{!14, !15}
!14 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !8, line: 5, type: !11)
!15 = !DILocalVariable(name: "i", scope: !16, file: !8, line: 6, type: !12)
!16 = distinct !DILexicalBlock(scope: !7, file: !8, line: 6, column: 5)
!17 = !DILocation(line: 0, scope: !7)
