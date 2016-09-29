; RUN: llc -mtriple=arm-eabi -stop-after=livedebugvalues %s -o - | FileCheck %s
;
; Check that the debug information for variables are propagated into the correct blocks.
;
; Generated from the C source:
;
; int func2(int,  int);
; void func(int a) {
;   int b = func2(10, 11);
;   if (a) {
;     int c = func2(12, 13);
;     for(int i = 1; i < a; i++) {
;       func2(i, i+b);
;     }
;     func2(b,c);
;   }
;   func2(b,a);
; }

; CHECK: [[VAR_A:![0-9]+]] = !DILocalVariable(name: "a",
; CHECK: [[VAR_B:![0-9]+]] = !DILocalVariable(name: "b",
; CHECK: [[VAR_C:![0-9]+]] = !DILocalVariable(name: "c",
; CHECK: [[VAR_I:![0-9]+]] = !DILocalVariable(name: "i",
;
; CHECK: bb.0.entry
; CHECK: DBG_VALUE debug-use %r0, debug-use _, [[VAR_A]]
; CHECK: DBG_VALUE debug-use [[REG_A:%r[0-9]+]], debug-use _, [[VAR_A]]
; CHECK: DBG_VALUE debug-use [[REG_B:%r[0-9]+]], debug-use _, [[VAR_B]]
;
; CHECK: bb.1.if.then
; CHECK: DBG_VALUE debug-use [[REG_B]], debug-use _, [[VAR_B]]
; CHECK: DBG_VALUE debug-use [[REG_A]], debug-use _, [[VAR_A]]
; CHECK: DBG_VALUE debug-use [[REG_C:%r[0-9]+]], debug-use _, [[VAR_C]]
; CHECK: DBG_VALUE 1, 0, [[VAR_I]]
;
; CHECK: bb.2.for.body
; CHECK: DBG_VALUE debug-use [[REG_I:%r[0-9]+]], debug-use _, [[VAR_I]]
; CHECK: DBG_VALUE debug-use [[REG_C]], debug-use _, [[VAR_C]]
; CHECK: DBG_VALUE debug-use [[REG_B]], debug-use _, [[VAR_B]]
; CHECK: DBG_VALUE debug-use [[REG_A]], debug-use _, [[VAR_A]]
; CHECK: DBG_VALUE debug-use [[REG_I]], debug-use _, [[VAR_I]]
;
; CHECK: bb.3.for.cond
; CHECK: DBG_VALUE debug-use [[REG_C]], debug-use _, [[VAR_C]]
; CHECK: DBG_VALUE debug-use [[REG_B]], debug-use _, [[VAR_B]]
; CHECK: DBG_VALUE debug-use [[REG_A]], debug-use _, [[VAR_A]]
; CHECK: DBG_VALUE debug-use [[REG_I]], debug-use _, [[VAR_I]]
;
; CHECK: bb.4.for.cond.cleanup
; CHECK: DBG_VALUE debug-use [[REG_C]], debug-use _, [[VAR_C]]
; CHECK: DBG_VALUE debug-use [[REG_B]], debug-use _, [[VAR_B]]
; CHECK: DBG_VALUE debug-use [[REG_A]], debug-use _, [[VAR_A]]
;
; CHECK: bb.5.if.end
; CHECK: DBG_VALUE debug-use [[REG_B]], debug-use _, [[VAR_B]]
; CHECK: DBG_VALUE debug-use [[REG_A]], debug-use _, [[VAR_A]]
  ; ModuleID = '/data/kwalker/work/OpenSource-llvm/llvm/test/CodeGen/ARM/dbg-range-extension.ll'
  source_filename = "/data/kwalker/work/OpenSource-llvm/llvm/test/CodeGen/ARM/dbg-range-extension.ll"
  target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
  target triple = "arm---eabi"
  
  ; Function Attrs: minsize nounwind optsize
  define void @func(i32 %a) local_unnamed_addr #0 !dbg !8 {
  entry:
    tail call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !13, metadata !20), !dbg !21
    %call = tail call i32 @func2(i32 10, i32 11) #0, !dbg !22
    tail call void @llvm.dbg.value(metadata i32 %call, i64 0, metadata !14, metadata !20), !dbg !23
    %tobool = icmp eq i32 %a, 0, !dbg !24
    br i1 %tobool, label %if.end, label %if.then, !dbg !25
  
  if.then:                                          ; preds = %entry
    %call1 = tail call i32 @func2(i32 12, i32 13) #0, !dbg !26
    tail call void @llvm.dbg.value(metadata i32 %call1, i64 0, metadata !15, metadata !20), !dbg !27
    tail call void @llvm.dbg.value(metadata i32 1, i64 0, metadata !18, metadata !20), !dbg !28
    br label %for.cond, !dbg !29
  
  for.cond:                                         ; preds = %for.body, %if.then
    %i.0 = phi i32 [ 1, %if.then ], [ %inc, %for.body ]
    tail call void @llvm.dbg.value(metadata i32 %i.0, i64 0, metadata !18, metadata !20), !dbg !28
    %cmp = icmp slt i32 %i.0, %a, !dbg !30
    br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !33
  
  for.cond.cleanup:                                 ; preds = %for.cond
    %call3 = tail call i32 @func2(i32 %call, i32 %call1) #0, !dbg !34
    br label %if.end, !dbg !35
  
  for.body:                                         ; preds = %for.cond
    %0 = add i32 %call, %i.0, !dbg !36
    %call2 = tail call i32 @func2(i32 %i.0, i32 %0) #0, !dbg !36
    %inc = add nuw nsw i32 %i.0, 1, !dbg !38
    tail call void @llvm.dbg.value(metadata i32 %inc, i64 0, metadata !18, metadata !20), !dbg !28
    br label %for.cond, !dbg !40, !llvm.loop !41
  
  if.end:                                           ; preds = %for.cond.cleanup, %entry
    %call4 = tail call i32 @func2(i32 %call, i32 %a) #0, !dbg !43
    ret void, !dbg !44
  }
  
  ; Function Attrs: minsize optsize
  declare i32 @func2(i32, i32) local_unnamed_addr #1
  
  ; Function Attrs: nounwind readnone
  declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2
  
  ; Function Attrs: nounwind
  declare void @llvm.stackprotector(i8*, i8**) #3
  
  attributes #0 = { minsize nounwind optsize }
  attributes #1 = { minsize optsize }
  attributes #2 = { nounwind readnone }
  attributes #3 = { nounwind }
  
  !llvm.dbg.cu = !{!0}
  !llvm.module.flags = !{!3, !4, !5, !6}
  !llvm.ident = !{!7}
  
  !0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
  !1 = !DIFile(filename: "loop.c", directory: "/tmp")
  !2 = !{}
  !3 = !{i32 2, !"Dwarf Version", i32 4}
  !4 = !{i32 2, !"Debug Info Version", i32 3}
  !5 = !{i32 1, !"wchar_size", i32 4}
  !6 = !{i32 1, !"min_enum_size", i32 4}
  !7 = !{!"clang version 4.0.0 (http://llvm.org/git/clang.git b8f10df3679b36f51e1de7c4351b82d297825089) (http://llvm.org/git/llvm.git c2a5d16d1e3b8c49f5bbb1ff87a76ac4f88edb89)"}
  !8 = distinct !DISubprogram(name: "func", scope: !1, file: !1, line: 2, type: !9, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !12)
  !9 = !DISubroutineType(types: !10)
  !10 = !{null, !11}
  !11 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
  !12 = !{!13, !14, !15, !18}
  !13 = !DILocalVariable(name: "a", arg: 1, scope: !8, file: !1, line: 2, type: !11)
  !14 = !DILocalVariable(name: "b", scope: !8, file: !1, line: 3, type: !11)
  !15 = !DILocalVariable(name: "c", scope: !16, file: !1, line: 5, type: !11)
  !16 = distinct !DILexicalBlock(scope: !17, file: !1, line: 4, column: 9)
  !17 = distinct !DILexicalBlock(scope: !8, file: !1, line: 4, column: 6)
  !18 = !DILocalVariable(name: "i", scope: !19, file: !1, line: 6, type: !11)
  !19 = distinct !DILexicalBlock(scope: !16, file: !1, line: 6, column: 3)
  !20 = !DIExpression()
  !21 = !DILocation(line: 2, column: 15, scope: !8)
  !22 = !DILocation(line: 3, column: 17, scope: !8)
  !23 = !DILocation(line: 3, column: 13, scope: !8)
  !24 = !DILocation(line: 4, column: 6, scope: !17)
  !25 = !DILocation(line: 4, column: 6, scope: !8)
  !26 = !DILocation(line: 5, column: 11, scope: !16)
  !27 = !DILocation(line: 5, column: 7, scope: !16)
  !28 = !DILocation(line: 6, column: 11, scope: !19)
  !29 = !DILocation(line: 6, column: 7, scope: !19)
  !30 = !DILocation(line: 6, column: 20, scope: !31)
  !31 = !DILexicalBlockFile(scope: !32, file: !1, discriminator: 1)
  !32 = distinct !DILexicalBlock(scope: !19, file: !1, line: 6, column: 3)
  !33 = !DILocation(line: 6, column: 3, scope: !31)
  !34 = !DILocation(line: 9, column: 3, scope: !16)
  !35 = !DILocation(line: 10, column: 2, scope: !16)
  !36 = !DILocation(line: 7, column: 4, scope: !37)
  !37 = distinct !DILexicalBlock(scope: !32, file: !1, line: 6, column: 30)
  !38 = !DILocation(line: 6, column: 26, scope: !39)
  !39 = !DILexicalBlockFile(scope: !32, file: !1, discriminator: 3)
  !40 = !DILocation(line: 6, column: 3, scope: !39)
  !41 = distinct !{!41, !42}
  !42 = !DILocation(line: 6, column: 3, scope: !16)
  !43 = !DILocation(line: 11, column: 2, scope: !8)
  !44 = !DILocation(line: 12, column: 1, scope: !8)
