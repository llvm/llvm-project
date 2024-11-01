; RUN: opt -passes=instcombine -S %s -o - | FileCheck %s

; In pr40648 one of two dbg.values used to describe the variable bumble was
; being dropped by instcombine. Test that both dbg.values survive instcombine.

; $ clang -O0 -Xclang -disable-O0-optnone -g bees.c -emit-llvm -S -o - \
;   | opt -opt-bisect-limit=10 -O2 -o -
; $ cat bees.c
; struct bees {
;  int a;
;  int b;
; }

; int global = 0;

; struct bees foo(struct bees bumble)
; {
;   global = 1;
;   int temp = bumble.a + bumble.b;
;   return bumble;
; }

; CHECK: define dso_local i64 @foo
; CHECK: @llvm.dbg.value(metadata i64 {{.*}}, metadata ![[BEE:[0-9]+]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32)),
; CHECK: @llvm.dbg.value(metadata i64 {{.*}}, metadata ![[BEE]], metadata !DIExpression({{.*}}, DW_OP_LLVM_fragment, 32, 32)),

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

@global = dso_local local_unnamed_addr global i32 0, align 4, !dbg !0

define dso_local i64 @foo(i64 %bumble.coerce) local_unnamed_addr !dbg !11 {
entry:
  %bumble.sroa.0.0.extract.trunc = trunc i64 %bumble.coerce to i32
  call void @llvm.dbg.value(metadata i32 %bumble.sroa.0.0.extract.trunc, metadata !18, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32)), !dbg !19
  %bumble.sroa.3.0.extract.shift = lshr i64 %bumble.coerce, 32
  %bumble.sroa.3.0.extract.trunc = trunc i64 %bumble.sroa.3.0.extract.shift to i32
  call void @llvm.dbg.value(metadata i32 %bumble.sroa.3.0.extract.trunc, metadata !18, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32)), !dbg !19
  store i32 1, i32* @global, align 4, !dbg !20
  call void @llvm.dbg.value(metadata i32 undef, metadata !21, metadata !DIExpression()), !dbg !19
  %retval.sroa.2.0.insert.ext = zext i32 %bumble.sroa.3.0.extract.trunc to i64, !dbg !22
  %retval.sroa.2.0.insert.shift = shl i64 %retval.sroa.2.0.insert.ext, 32, !dbg !22
  %retval.sroa.0.0.insert.ext = zext i32 %bumble.sroa.0.0.extract.trunc to i64, !dbg !22
  %retval.sroa.0.0.insert.insert = or i64 %retval.sroa.2.0.insert.shift, %retval.sroa.0.0.insert.ext, !dbg !22
  ret i64 %retval.sroa.0.0.insert.insert, !dbg !22
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "global", scope: !2, file: !3, line: 6, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 10.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "bees.c", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 10.0.0"}
!11 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 8, type: !12, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!12 = !DISubroutineType(types: !13)
!13 = !{!14, !14}
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bees", file: !3, line: 1, size: 64, elements: !15)
!15 = !{!16, !17}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !14, file: !3, line: 2, baseType: !6, size: 32)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !14, file: !3, line: 3, baseType: !6, size: 32, offset: 32)
!18 = !DILocalVariable(name: "bumble", arg: 1, scope: !11, file: !3, line: 8, type: !14)
!19 = !DILocation(line: 0, scope: !11)
!20 = !DILocation(line: 10, column: 10, scope: !11)
!21 = !DILocalVariable(name: "temp", scope: !11, file: !3, line: 11, type: !6)
!22 = !DILocation(line: 12, column: 3, scope: !11)
