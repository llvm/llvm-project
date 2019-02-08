; RUN: llc -O0 %s -o - -filetype=obj \
; RUN:   | llvm-dwarfdump -statistics - | FileCheck %s
; CHECK: "version":2

; int GlobalConst = 42;
; int Global;
;
; struct S {
;   static const int constant = 24;
; } s;
;
; int __attribute__((always_inline)) square(int i) { return i * i; }
; int cube(int i) {
;   int squared = square(i);
;   return squared*i;
; }

; GlobalConst,Global,s,s.constant,square::i,cube::i,cube::squared
; CHECK: "unique source variables":7
; +1 extra inline i.
; CHECK: "source variables":8
; -1 square::i
; CHECK: "variables with location":7
; CHECK: "scope bytes total":[[BYTES:[0-9]+]]
; Because of the dbg.value in the middle of the function, the pc range coverage
; must be below 100%.
; CHECK-NOT: "scope bytes covered":0
; CHECK-NOT "scope bytes covered":[[BYTES]]
; CHECK: "scope bytes covered":
; CHECK: "total function size":[[FUNCSIZE:[0-9]+]]
; CHECK: "total inlined function size":[[INLINESIZE:[0-9]+]]


; ModuleID = '/tmp/quality.cpp'
source_filename = "/tmp/quality.cpp"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

%struct.S = type { i8 }

@GlobalConst = global i32 42, align 4, !dbg !0
@Global = global i32 0, align 4, !dbg !6
@s = global %struct.S zeroinitializer, align 1, !dbg !9

; Function Attrs: alwaysinline nounwind ssp uwtable
define i32 @_Z6squarei(i32 %i) #0 !dbg !20 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  ; Modified to loose debug info for i here.
  call void @llvm.dbg.declare(metadata i32* undef, metadata !23, metadata !24), !dbg !25
  %0 = load i32, i32* %i.addr, align 4, !dbg !26
  %1 = load i32, i32* %i.addr, align 4, !dbg !27
  %mul = mul nsw i32 %0, %1, !dbg !28
  ret i32 %mul, !dbg !29
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone ssp uwtable
define i32 @_Z4cubei(i32 %i) #2 !dbg !30 {
entry:
  %i.addr.i = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %i.addr.i, metadata !23, metadata !24), !dbg !31
  %i.addr = alloca i32, align 4
  %squared = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %i.addr, metadata !33, metadata !24), !dbg !34
  %0 = load i32, i32* %i.addr, align 4, !dbg !37
  store i32 %0, i32* %i.addr.i, align 4
  %1 = load i32, i32* %i.addr.i, align 4, !dbg !38
  %2 = load i32, i32* %i.addr.i, align 4, !dbg !39
  %mul.i = mul nsw i32 %1, %2, !dbg !40
  ; Modified to cover only about 50% of the lexical scope.
  call void @llvm.dbg.value(metadata i32 %mul.i, metadata !35, metadata !24), !dbg !36
  store i32 %mul.i, i32* %squared, align 4, !dbg !36
  %3 = load i32, i32* %squared, align 4, !dbg !41
  call void @llvm.dbg.value(metadata i32 %3, metadata !35, metadata !24), !dbg !36
  %4 = load i32, i32* %i.addr, align 4, !dbg !42
  %mul = mul nsw i32 %3, %4, !dbg !43
  ret i32 %mul, !dbg !44
}

attributes #0 = { alwaysinline nounwind ssp uwtable }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { noinline nounwind optnone ssp uwtable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!15, !16, !17, !18}
!llvm.ident = !{!19}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "GlobalConst", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 6.0.0 (trunk 310529) (llvm/trunk 310534)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "/tmp/quality.cpp", directory: "/Volumes/Data/llvm")
!4 = !{}
!5 = !{!0, !6, !9}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "Global", scope: !2, file: !3, line: 2, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = distinct !DIGlobalVariable(name: "s", scope: !2, file: !3, line: 6, type: !11, isLocal: false, isDefinition: true)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !3, line: 4, size: 8, elements: !12, identifier: "_ZTS1S")
!12 = !{!13}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "constant", scope: !11, file: !3, line: 5, baseType: !14, flags: DIFlagStaticMember, extraData: i32 24)
!14 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !8)
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{i32 1, !"wchar_size", i32 4}
!18 = !{i32 7, !"PIC Level", i32 2}
!19 = !{!"clang version 6.0.0 (trunk 310529) (llvm/trunk 310534)"}
!20 = distinct !DISubprogram(name: "square", linkageName: "_Z6squarei", scope: !3, file: !3, line: 8, type: !21, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!21 = !DISubroutineType(types: !22)
!22 = !{!8, !8}
!23 = !DILocalVariable(name: "i", arg: 1, scope: !20, file: !3, line: 8, type: !8)
!24 = !DIExpression()
!25 = !DILocation(line: 8, column: 47, scope: !20)
!26 = !DILocation(line: 8, column: 59, scope: !20)
!27 = !DILocation(line: 8, column: 63, scope: !20)
!28 = !DILocation(line: 8, column: 61, scope: !20)
!29 = !DILocation(line: 8, column: 52, scope: !20)
!30 = distinct !DISubprogram(name: "cube", linkageName: "_Z4cubei", scope: !3, file: !3, line: 9, type: !21, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!31 = !DILocation(line: 8, column: 47, scope: !20, inlinedAt: !32)
!32 = distinct !DILocation(line: 10, column: 17, scope: !30)
!33 = !DILocalVariable(name: "i", arg: 1, scope: !30, file: !3, line: 9, type: !8)
!34 = !DILocation(line: 9, column: 14, scope: !30)
!35 = !DILocalVariable(name: "squared", scope: !30, file: !3, line: 10, type: !8)
!36 = !DILocation(line: 10, column: 7, scope: !30)
!37 = !DILocation(line: 10, column: 24, scope: !30)
!38 = !DILocation(line: 8, column: 59, scope: !20, inlinedAt: !32)
!39 = !DILocation(line: 8, column: 63, scope: !20, inlinedAt: !32)
!40 = !DILocation(line: 8, column: 61, scope: !20, inlinedAt: !32)
!41 = !DILocation(line: 11, column: 10, scope: !30)
!42 = !DILocation(line: 11, column: 18, scope: !30)
!43 = !DILocation(line: 11, column: 17, scope: !30)
!44 = !DILocation(line: 11, column: 3, scope: !30)
