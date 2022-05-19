; RUN: opt -passes=deadargelim -S < %s | FileCheck %s
;test.c
;int s;
;
;void f2(int k) __attribute__((noinline)) {
; s++;
; k = s;
;}
;
;void f() __attribute__((noinline)) {
; f2(4);
;}
;
;int main()
;{
; f();
; return 0;
;}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@s = common dso_local local_unnamed_addr global i32 0, align 4, !dbg !0

; Function Attrs: noinline nounwind uwtable
define dso_local void @f2(i32 %k) local_unnamed_addr !dbg !11 {
entry:
; CHECK: call void @llvm.dbg.value(metadata i32 poison, metadata !15, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.value(metadata i32 %k, metadata !15, metadata !DIExpression()), !dbg !16
  %0 = load i32, i32* @s, align 4, !dbg !17
  %inc = add nsw i32 %0, 1, !dbg !17
  store i32 %inc, i32* @s, align 4, !dbg !17
  call void @llvm.dbg.value(metadata i32* @s, metadata !15, metadata !DIExpression(DW_OP_deref)), !dbg !16
  ret void, !dbg !18
}

; Function Attrs: noinline nounwind uwtable
define dso_local void @f() local_unnamed_addr !dbg !19 {
entry:
; CHECK: tail call void @f2(i32 poison), !dbg !22
  tail call void @f2(i32 4), !dbg !22
  ret void, !dbg !23
}

; Function Attrs: nounwind uwtable
define dso_local i32 @main() local_unnamed_addr !dbg !24 {
entry:
  tail call void @f(), !dbg !27
  ret i32 0, !dbg !28
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "s", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 8.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "test.c", directory: "/dir")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 7.0.0"}
!11 = distinct !DISubprogram(name: "f2", scope: !3, file: !3, line: 3, type: !12, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !14)
!12 = !DISubroutineType(types: !13)
!13 = !{null, !6}
!14 = !{!15}
!15 = !DILocalVariable(name: "k", arg: 1, scope: !11, file: !3, line: 3, type: !6)
!16 = !DILocation(line: 3, column: 13, scope: !11)
!17 = !DILocation(line: 4, column: 3, scope: !11)
!18 = !DILocation(line: 6, column: 1, scope: !11)
!19 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 8, type: !20, isLocal: false, isDefinition: true, scopeLine: 8, isOptimized: true, unit: !2, retainedNodes: !4)
!20 = !DISubroutineType(types: !21)
!21 = !{null}
!22 = !DILocation(line: 9, column: 2, scope: !19)
!23 = !DILocation(line: 10, column: 1, scope: !19)
!24 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 12, type: !25, isLocal: false, isDefinition: true, scopeLine: 12, isOptimized: true, unit: !2, retainedNodes: !4)
!25 = !DISubroutineType(types: !26)
!26 = !{!6}
!27 = !DILocation(line: 13, column: 2, scope: !24)
!28 = !DILocation(line: 14, column: 2, scope: !24)
