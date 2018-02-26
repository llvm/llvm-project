; RUN: opt -mem2reg -S < %s | FileCheck %s

; ModuleID = 'newvars.c'
source_filename = "newvars.c"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

; Function Attrs: nounwind uwtable
define i32 @if_else(i32 %cond, i32 %a, i32 %b) !dbg !8 {
entry:
  %x = alloca i32, align 4
  call void @llvm.dbg.addr(metadata i32* %x, metadata !16, metadata !DIExpression()), !dbg !26
  store i32 %a, i32* %x, align 4, !dbg !26, !tbaa !17
  %tobool = icmp ne i32 %cond, 0, !dbg !28
  br i1 %tobool, label %if.then, label %if.else, !dbg !30

if.then:                                          ; preds = %entry
  store i32 0, i32* %x, align 4, !dbg !31, !tbaa !17
  br label %if.end, !dbg !33

if.else:                                          ; preds = %entry
  store i32 %b, i32* %x, align 4, !dbg !36, !tbaa !17
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %rv = load i32, i32* %x, align 4, !dbg !37, !tbaa !17
  ret i32 %rv, !dbg !39
}

; CHECK-LABEL: define i32 @if_else({{.*}})
; CHECK: entry:
; CHECK-NOT:   alloca i32
; CHECK:   call void @llvm.dbg.value(metadata i32 %a, metadata ![[X_LOCAL:[0-9]+]], metadata !DIExpression())
; CHECK: if.then:                                          ; preds = %entry
; CHECK:   call void @llvm.dbg.value(metadata i32 0, metadata ![[X_LOCAL]], metadata !DIExpression())
; CHECK: if.else:                                          ; preds = %entry
; CHECK:   call void @llvm.dbg.value(metadata i32 %b, metadata ![[X_LOCAL]], metadata !DIExpression())
; CHECK: if.end:                                           ; preds = %if.else, %if.then
; CHECK:   %[[PHI:[^ ]*]] = phi i32 [ 0, %if.then ], [ %b, %if.else ]
; CHECK:   call void @llvm.dbg.value(metadata i32 %[[PHI]], metadata ![[X_LOCAL]], metadata !DIExpression())
; CHECK:   ret i32

; CHECK: ![[X_LOCAL]] = !DILocalVariable(name: "x", {{.*}})

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.addr(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "newvars.c", directory: "C:\5Csrc\5Cllvm-project\5Cbuild")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 6.0.0 "}
!8 = distinct !DISubprogram(name: "if_else", scope: !1, file: !1, line: 1, type: !9, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11, !11, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13, !14, !15, !16}
!13 = !DILocalVariable(name: "b", arg: 3, scope: !8, file: !1, line: 1, type: !11)
!14 = !DILocalVariable(name: "a", arg: 2, scope: !8, file: !1, line: 1, type: !11)
!15 = !DILocalVariable(name: "cond", arg: 1, scope: !8, file: !1, line: 1, type: !11)
!16 = !DILocalVariable(name: "x", scope: !8, file: !1, line: 2, type: !11)
!17 = !{!18, !18, i64 0}
!18 = !{!"int", !19, i64 0}
!19 = !{!"omnipotent char", !20, i64 0}
!20 = !{!"Simple C/C++ TBAA"}
!22 = !DILocation(line: 1, column: 34, scope: !8)
!23 = !DILocation(line: 1, column: 27, scope: !8)
!24 = !DILocation(line: 1, column: 17, scope: !8)
!25 = !DILocation(line: 2, column: 3, scope: !8)
!26 = !DILocation(line: 2, column: 7, scope: !8)
!27 = !DILocation(line: 2, column: 11, scope: !8)
!28 = !DILocation(line: 3, column: 7, scope: !29)
!29 = distinct !DILexicalBlock(scope: !8, file: !1, line: 3, column: 7)
!30 = !DILocation(line: 3, column: 7, scope: !8)
!31 = !DILocation(line: 4, column: 7, scope: !32)
!32 = distinct !DILexicalBlock(scope: !29, file: !1, line: 3, column: 13)
!33 = !DILocation(line: 5, column: 3, scope: !32)
!34 = !DILocation(line: 6, column: 9, scope: !35)
!35 = distinct !DILexicalBlock(scope: !29, file: !1, line: 5, column: 10)
!36 = !DILocation(line: 6, column: 7, scope: !35)
!37 = !DILocation(line: 8, column: 10, scope: !8)
!38 = !DILocation(line: 9, column: 1, scope: !8)
!39 = !DILocation(line: 8, column: 3, scope: !8)
