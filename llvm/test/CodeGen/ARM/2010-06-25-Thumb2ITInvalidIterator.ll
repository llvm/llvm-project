; RUN: llc < %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin3.0.0-iphoneos"

@length = common global i32 0, align 4, !dbg !14            ; <i32*> [#uses=1]

define void @x0(i8* nocapture %buf, i32 %nbytes) nounwind optsize {
entry:
  tail call void @llvm.dbg.value(metadata i8* %buf, i64 0, metadata !0, metadata !DIExpression()), !dbg !15
  tail call void @llvm.dbg.value(metadata i32 %nbytes, i64 0, metadata !8, metadata !DIExpression()), !dbg !16
  %tmp = load i32, i32* @length, !dbg !17              ; <i32> [#uses=3]
  %cmp = icmp eq i32 %tmp, -1, !dbg !17           ; <i1> [#uses=1]
  %cmp.not = xor i1 %cmp, true                    ; <i1> [#uses=1]
  %cmp3 = icmp ult i32 %tmp, %nbytes, !dbg !17    ; <i1> [#uses=1]
  %or.cond = and i1 %cmp.not, %cmp3               ; <i1> [#uses=1]
  tail call void @llvm.dbg.value(metadata i32 %tmp, i64 0, metadata !8, metadata !DIExpression()), !dbg !17
  %nbytes.addr.0 = select i1 %or.cond, i32 %tmp, i32 %nbytes ; <i32> [#uses=1]
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !10, metadata !DIExpression()), !dbg !19
  br label %while.cond, !dbg !20

while.cond:                                       ; preds = %while.body, %entry
  %0 = phi i32 [ 0, %entry ], [ %inc, %while.body ] ; <i32> [#uses=3]
  %buf.addr.0 = getelementptr i8, i8* %buf, i32 %0    ; <i8*> [#uses=1]
  %cmp7 = icmp ult i32 %0, %nbytes.addr.0, !dbg !20 ; <i1> [#uses=1]
  br i1 %cmp7, label %land.rhs, label %while.end, !dbg !20

land.rhs:                                         ; preds = %while.cond
  %call = tail call i32 @x1() nounwind optsize, !dbg !20 ; <i32> [#uses=2]
  %cmp9 = icmp eq i32 %call, -1, !dbg !20         ; <i1> [#uses=1]
  br i1 %cmp9, label %while.end, label %while.body, !dbg !20

while.body:                                       ; preds = %land.rhs
  %conv = trunc i32 %call to i8, !dbg !21         ; <i8> [#uses=1]
  store i8 %conv, i8* %buf.addr.0, !dbg !21
  %inc = add i32 %0, 1, !dbg !23                  ; <i32> [#uses=1]
  br label %while.cond, !dbg !24

while.end:                                        ; preds = %land.rhs, %while.cond
  ret void, !dbg !25
}

declare i32 @x1() optsize

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!3}

!0 = !DILocalVariable(name: "buf", line: 4, arg: 1, scope: !1, file: !2, type: !6)
!1 = distinct !DISubprogram(name: "x0", linkageName: "x0", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !3, file: !26, scope: null, type: !4)
!2 = !DIFile(filename: "t.c", directory: "/private/tmp")
!3 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang 2.0", isOptimized: true, file: !26, globals: !{!14})
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, file: !26, scope: !2, baseType: !7)
!7 = !DIBasicType(tag: DW_TAG_base_type, name: "unsigned char", size: 8, align: 8, encoding: DW_ATE_unsigned_char)
!8 = !DILocalVariable(name: "nbytes", line: 4, arg: 2, scope: !1, file: !2, type: !9)
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "unsigned long", size: 32, align: 32, encoding: DW_ATE_unsigned)
!10 = !DILocalVariable(name: "nread", line: 6, scope: !11, file: !2, type: !9)
!11 = distinct !DILexicalBlock(line: 5, column: 1, file: !26, scope: !1)
!12 = !DILocalVariable(name: "c", line: 7, scope: !11, file: !2, type: !13)
!13 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = !DIGlobalVariable(name: "length", linkageName: "length", line: 1, isLocal: false, isDefinition: true, scope: !2, file: !2, type: !13)
!15 = !DILocation(line: 4, column: 24, scope: !1)
!16 = !DILocation(line: 4, column: 43, scope: !1)
!17 = !DILocation(line: 9, column: 2, scope: !11)
!18 = !{i32 0}
!19 = !DILocation(line: 10, column: 2, scope: !11)
!20 = !DILocation(line: 11, column: 2, scope: !11)
!21 = !DILocation(line: 12, column: 3, scope: !22)
!22 = distinct !DILexicalBlock(line: 11, column: 45, file: !26, scope: !11)
!23 = !DILocation(line: 13, column: 3, scope: !22)
!24 = !DILocation(line: 14, column: 2, scope: !22)
!25 = !DILocation(line: 15, column: 1, scope: !11)
!26 = !DIFile(filename: "t.c", directory: "/private/tmp")
