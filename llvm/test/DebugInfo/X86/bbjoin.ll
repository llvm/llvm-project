; RUN: llc -mtriple=x86_64-apple-macosx10.9.0 %s -stop-after=livedebugvars \
; RUN:   -o - | FileCheck %s
; Generated from:
; void g(int *);
; int f() {
;   int x = 23;
;   g(&x);
;   if (x == 42)
;     ++x;
;   return x; // check that x is not a constant here.
; }
; CHECK: ![[X:.*]] = !DILocalVariable(name: "x",
; CHECK: bb.0.entry:
; CHECK:   DBG_VALUE 23, 0, ![[X]],
; CHECK:   DBG_VALUE %rsp, 4, ![[X]]
; CHECK: bb.1.if.then:
; CHECK:   DBG_VALUE 43, 0, ![[X]],
; CHECK: bb.2.if.end:
; CHECK-NOT:  DBG_VALUE 23, 0, ![[X]],
; CHECK:   RETQ %eax

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

; Function Attrs: nounwind ssp uwtable
define i32 @f() #0 !dbg !4 {
entry:
  %x = alloca i32, align 4
  %0 = bitcast i32* %x to i8*, !dbg !14
  call void @llvm.lifetime.start(i64 4, i8* %0) #4, !dbg !14
  tail call void @llvm.dbg.value(metadata i32 23, i64 0, metadata !9, metadata !15), !dbg !16
  store i32 23, i32* %x, align 4, !dbg !16, !tbaa !17
  tail call void @llvm.dbg.value(metadata i32* %x, i64 0, metadata !9, metadata !DIExpression(DW_OP_deref)), !dbg !16
  call void @g(i32* nonnull %x) #4, !dbg !21
  call void @llvm.dbg.value(metadata i32* %x, i64 0, metadata !9, metadata !DIExpression(DW_OP_deref)), !dbg !16
  %1 = load i32, i32* %x, align 4, !dbg !22, !tbaa !17
  %cmp = icmp eq i32 %1, 42, !dbg !24
  br i1 %cmp, label %if.then, label %if.end, !dbg !25

if.then:                                          ; preds = %entry
  call void @llvm.dbg.value(metadata i32 43, i64 0, metadata !9, metadata !15), !dbg !16
  store i32 43, i32* %x, align 4, !dbg !26, !tbaa !17
  br label %if.end, !dbg !26

if.end:                                           ; preds = %if.then, %entry
  %2 = phi i32 [ 43, %if.then ], [ %1, %entry ], !dbg !27
  call void @llvm.dbg.value(metadata i32* %x, i64 0, metadata !9, metadata !DIExpression(DW_OP_deref)), !dbg !16
  call void @llvm.lifetime.end(i64 4, i8* %0) #4, !dbg !28
  ret i32 %2, !dbg !29
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #1

declare void @g(i32*) #4

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #3

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { argmemonly nounwind }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 (trunk 255890) (llvm/trunk 255919)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "constant.c", directory: "")
!2 = !{}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: true, unit: !0, variables: !8)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{!9}
!9 = !DILocalVariable(name: "x", scope: !4, file: !1, line: 3, type: !7)
!10 = !{i32 2, !"Dwarf Version", i32 2}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"PIC Level", i32 2}
!13 = !{!"clang version 3.8.0 (trunk 255890) (llvm/trunk 255919)"}
!14 = !DILocation(line: 3, column: 3, scope: !4)
!15 = !DIExpression()
!16 = !DILocation(line: 3, column: 7, scope: !4)
!17 = !{!18, !18, i64 0}
!18 = !{!"int", !19, i64 0}
!19 = !{!"omnipotent char", !20, i64 0}
!20 = !{!"Simple C/C++ TBAA"}
!21 = !DILocation(line: 4, column: 3, scope: !4)
!22 = !DILocation(line: 5, column: 7, scope: !23)
!23 = distinct !DILexicalBlock(scope: !4, file: !1, line: 5, column: 7)
!24 = !DILocation(line: 5, column: 9, scope: !23)
!25 = !DILocation(line: 5, column: 7, scope: !4)
!26 = !DILocation(line: 6, column: 5, scope: !23)
!27 = !DILocation(line: 7, column: 10, scope: !4)
!28 = !DILocation(line: 8, column: 1, scope: !4)
!29 = !DILocation(line: 7, column: 3, scope: !4)
