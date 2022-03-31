; RUN: opt -passes=instcombine < %s -S | FileCheck %s

; This tests dbg.declare lowering for CallInst users of an alloca. The
; resulting dbg.value expressions should add a deref to the declare's expression.

; Hand-reduced from this example (-g -Og -mllvm -disable-llvm-optzns -gno-column-info):

; static volatile int sink;
; static void escape(int &c) { sink = c; }
; static bool empty ( int p1 ) { return p1 == 0; }
; int main() {
;   int d1 = 42;
;   while (!empty(d1))
;     escape(d1);
;   return 0;
; }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZL4sink = internal global i32 0, align 4, !dbg !0

; CHECK-LABEL: @main
define dso_local i32 @main() !dbg !12 {
entry:
  %retval = alloca i32, align 4
  %d1 = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  %0 = bitcast i32* %d1 to i8*, !dbg !17
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #4, !dbg !17
; CHECK: dbg.value(metadata i32 42, metadata [[METADATA_IDX1:![0-9]+]], metadata !DIExpression())
; CHECK-NEXT: store
  call void @llvm.dbg.declare(metadata i32* %d1, metadata !16, metadata !DIExpression()), !dbg !17
  store i32 42, i32* %d1, align 4, !dbg !17
  br label %while.cond, !dbg !22

while.cond:                                       ; preds = %while.body, %entry
; CHECK: dbg.value(metadata i32 %1, metadata [[METADATA_IDX1]], metadata !DIExpression())
; CHECK-NEXT: call zeroext i1 @_ZL5emptyi
  %1 = load i32, i32* %d1, align 4, !dbg !22
  %call = call zeroext i1 @_ZL5emptyi(i32 %1), !dbg !22
  %lnot = xor i1 %call, true, !dbg !22
  br i1 %lnot, label %while.body, label %while.end, !dbg !22

while.body:                                       ; preds = %while.cond
; CHECK: dbg.value(metadata i32* %d1, metadata [[METADATA_IDX1]], metadata !DIExpression(DW_OP_deref))
; CHECK-NEXT: call void @_ZL6escapeRi
  call void @_ZL6escapeRi(i32* dereferenceable(4) %d1), !dbg !23
  br label %while.cond, !dbg !22, !llvm.loop !24

while.end:                                        ; preds = %while.cond
  %2 = bitcast i32* %d1 to i8*, !dbg !25
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %2) #4, !dbg !25
  ret i32 0, !dbg !26
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)

declare void @llvm.dbg.declare(metadata, metadata, metadata)

define internal zeroext i1 @_ZL5emptyi(i32 %p1) !dbg !27 {
  ret i1 false
}

define internal void @_ZL6escapeRi(i32* dereferenceable(4) %c) #3 !dbg !34 {
  ret void
}

declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

; CHECK: DIGlobalVariableExpression
; CHECK: [[METADATA_IDX1]] = !DILocalVariable(name: "d1"

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "sink", linkageName: "_ZL4sink", scope: !2, file: !3, line: 2, type: !6, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "lower-dbg-declare.cpp", directory: "")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !7)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{!"clang"}
!12 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 5, type: !13, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{!7}
!15 = !{!16}
!16 = !DILocalVariable(name: "d1", scope: !12, file: !3, line: 6, type: !7)
!17 = !DILocation(line: 6, scope: !12)
!22 = !DILocation(line: 7, scope: !12)
!23 = !DILocation(line: 8, scope: !12)
!24 = distinct !{!24, !22, !23}
!25 = !DILocation(line: 10, scope: !12)
!26 = !DILocation(line: 9, scope: !12)
!27 = distinct !DISubprogram(name: "empty", linkageName: "_ZL5emptyi", scope: !3, file: !3, line: 4, type: !28, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !31)
!28 = !DISubroutineType(types: !29)
!29 = !{!30, !7}
!30 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!31 = !{!32}
!32 = !DILocalVariable(name: "p1", arg: 1, scope: !27, file: !3, line: 4, type: !7)
!33 = !DILocation(line: 4, scope: !27)
!34 = distinct !DISubprogram(name: "escape", linkageName: "_ZL6escapeRi", scope: !3, file: !3, line: 3, type: !35, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !38)
!35 = !DISubroutineType(types: !36)
!36 = !{null, !37}
!37 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !7, size: 64)
!38 = !{!39}
!39 = !DILocalVariable(name: "c", arg: 1, scope: !34, file: !3, line: 3, type: !37)
!42 = !DILocation(line: 3, scope: !34)
