; RUN: llc < %s

source_filename = "test/CodeGen/ARM/2010-06-25-Thumb2ITInvalidIterator.ll"
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin3.0.0-iphoneos"

@length = common global i32 0, align 4, !dbg !0

; Function Attrs: nounwind optsize
define void @x0(ptr nocapture %buf, i32 %nbytes) #0 {
entry:
  tail call void @llvm.dbg.value(metadata ptr %buf, metadata !8, metadata !14), !dbg !15
  tail call void @llvm.dbg.value(metadata i32 %nbytes, metadata !16, metadata !14), !dbg !18
  %tmp = load i32, ptr @length, !dbg !19
  %cmp = icmp eq i32 %tmp, -1, !dbg !19
  %cmp.not = xor i1 %cmp, true
  %cmp3 = icmp ult i32 %tmp, %nbytes, !dbg !19
  %or.cond = and i1 %cmp.not, %cmp3
  tail call void @llvm.dbg.value(metadata i32 %tmp, metadata !16, metadata !14), !dbg !19
  %nbytes.addr.0 = select i1 %or.cond, i32 %tmp, i32 %nbytes
  tail call void @llvm.dbg.value(metadata i32 0, metadata !21, metadata !14), !dbg !22
  br label %while.cond, !dbg !23

while.cond:                                       ; preds = %while.body, %entry

  %0 = phi i32 [ 0, %entry ], [ %inc, %while.body ]
  %buf.addr.0 = getelementptr i8, ptr %buf, i32 %0
  %cmp7 = icmp ult i32 %0, %nbytes.addr.0, !dbg !23
  br i1 %cmp7, label %land.rhs, label %while.end, !dbg !23

land.rhs:                                         ; preds = %while.cond
  %call = tail call i32 @x1() #0, !dbg !23
  %cmp9 = icmp eq i32 %call, -1, !dbg !23
  br i1 %cmp9, label %while.end, label %while.body, !dbg !23

while.body:                                       ; preds = %land.rhs
  %conv = trunc i32 %call to i8, !dbg !24
  store i8 %conv, ptr %buf.addr.0, !dbg !24
  %inc = add i32 %0, 1, !dbg !26
  br label %while.cond, !dbg !27

while.end:                                        ; preds = %land.rhs, %while.cond
  ret void, !dbg !28
}

; Function Attrs: optsize
declare i32 @x1() #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind optsize }
attributes #1 = { optsize }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!6, !7}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "length", linkageName: "length", scope: !2, file: !2, line: 1, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "t.c", directory: "/private/tmp")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang 2.0", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, globals: !5)
!5 = !{!0}
!6 = !{i32 2, !"Dwarf Version", i32 2}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !DILocalVariable(name: "buf", arg: 1, scope: !9, file: !2, line: 4, type: !12)
!9 = distinct !DISubprogram(name: "x0", linkageName: "x0", scope: null, file: !2, line: 5, type: !10, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !4)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, scope: !2, file: !2, baseType: !13, size: 32, align: 32)
!13 = !DIBasicType(name: "unsigned char", size: 8, align: 8, encoding: DW_ATE_unsigned_char)
!14 = !DIExpression()
!15 = !DILocation(line: 4, column: 24, scope: !9)
!16 = !DILocalVariable(name: "nbytes", arg: 2, scope: !9, file: !2, line: 4, type: !17)
!17 = !DIBasicType(name: "unsigned long", size: 32, align: 32, encoding: DW_ATE_unsigned)
!18 = !DILocation(line: 4, column: 43, scope: !9)
!19 = !DILocation(line: 9, column: 2, scope: !20)
!20 = distinct !DILexicalBlock(scope: !9, file: !2, line: 5, column: 1)
!21 = !DILocalVariable(name: "nread", scope: !20, file: !2, line: 6, type: !17)
!22 = !DILocation(line: 10, column: 2, scope: !20)
!23 = !DILocation(line: 11, column: 2, scope: !20)
!24 = !DILocation(line: 12, column: 3, scope: !25)
!25 = distinct !DILexicalBlock(scope: !20, file: !2, line: 11, column: 45)
!26 = !DILocation(line: 13, column: 3, scope: !25)
!27 = !DILocation(line: 14, column: 2, scope: !25)
!28 = !DILocation(line: 15, column: 1, scope: !20)

