; RUN: llc -O0 %s -o - -filetype=obj \
; RUN:   | llvm-dwarfdump -statistics - | FileCheck %s

; This test serves as a baseline / sanity-check for stats-inlining-multi-cu.ll
; The results for both tests should be identical.

; CHECK:      "#functions": 4,
; CHECK:      "#inlined functions": 2,
; CHECK:      "#unique source variables": 4,
; CHECK-NEXT: "#source variables": 6,
; CHECK-NEXT: "#source variables with location": 6,

;header.h:
;extern "C" int getchar();
;template<typename T> T __attribute__((always_inline)) inlined() {
;  if (getchar()=='a') {
;    int i = getchar();
;    return i;
;  } else {
;    int i = 'a';
;    return i;
;  }
;}
;ab.cpp
;#include <header.h>
;int b();
;int a() {
;  int a = inlined<int>();
;  return a+1;
;}
; 
;int b() {
;  int b = inlined<int>();
;  return b+1;
;}
;int main() {
;  return a() + b();
;}


; ModuleID = 'a.cpp'
source_filename = "a.cpp"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; Function Attrs: noinline optnone ssp uwtable
define i32 @_Z1av() #0 !dbg !8 {
entry:
  %retval.i = alloca i32, align 4
  %i.i = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %i.i, metadata !12, metadata !DIExpression()), !dbg !19
  %i2.i = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %i2.i, metadata !21, metadata !DIExpression()), !dbg !23
  %a = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %a, metadata !24, metadata !DIExpression()), !dbg !25
  %call.i = call i32 @getchar(), !dbg !26
  %cmp.i = icmp eq i32 %call.i, 97, !dbg !26
  br i1 %cmp.i, label %if.then.i, label %if.else.i, !dbg !27

if.then.i:                                        ; preds = %entry
  %call1.i = call i32 @getchar(), !dbg !19
  store i32 %call1.i, i32* %i.i, align 4, !dbg !19
  %0 = load i32, i32* %i.i, align 4, !dbg !28
  store i32 %0, i32* %retval.i, align 4, !dbg !28
  br label %_Z7inlinedIiET_v.exit, !dbg !28

if.else.i:                                        ; preds = %entry
  store i32 97, i32* %i2.i, align 4, !dbg !23
  %1 = load i32, i32* %i2.i, align 4, !dbg !29
  store i32 %1, i32* %retval.i, align 4, !dbg !29
  br label %_Z7inlinedIiET_v.exit, !dbg !29

_Z7inlinedIiET_v.exit:                            ; preds = %if.then.i, %if.else.i
  %2 = load i32, i32* %retval.i, align 4, !dbg !30
  store i32 %2, i32* %a, align 4, !dbg !25
  %3 = load i32, i32* %a, align 4, !dbg !31
  %add = add nsw i32 %3, 1, !dbg !31
  ret i32 %add, !dbg !31
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline optnone ssp uwtable
define i32 @_Z1bv() #0 !dbg !32 {
entry:
  %retval.i = alloca i32, align 4
  %i.i = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %i.i, metadata !12, metadata !DIExpression()), !dbg !33
  %i2.i = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %i2.i, metadata !21, metadata !DIExpression()), !dbg !35
  %b = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %b, metadata !36, metadata !DIExpression()), !dbg !37
  %call.i = call i32 @getchar(), !dbg !38
  %cmp.i = icmp eq i32 %call.i, 97, !dbg !38
  br i1 %cmp.i, label %if.then.i, label %if.else.i, !dbg !39

if.then.i:                                        ; preds = %entry
  %call1.i = call i32 @getchar(), !dbg !33
  store i32 %call1.i, i32* %i.i, align 4, !dbg !33
  %0 = load i32, i32* %i.i, align 4, !dbg !40
  store i32 %0, i32* %retval.i, align 4, !dbg !40
  br label %_Z7inlinedIiET_v.exit, !dbg !40

if.else.i:                                        ; preds = %entry
  store i32 97, i32* %i2.i, align 4, !dbg !35
  %1 = load i32, i32* %i2.i, align 4, !dbg !41
  store i32 %1, i32* %retval.i, align 4, !dbg !41
  br label %_Z7inlinedIiET_v.exit, !dbg !41

_Z7inlinedIiET_v.exit:                            ; preds = %if.then.i, %if.else.i
  %2 = load i32, i32* %retval.i, align 4, !dbg !42
  store i32 %2, i32* %b, align 4, !dbg !37
  %3 = load i32, i32* %b, align 4, !dbg !43
  %add = add nsw i32 %3, 1, !dbg !43
  ret i32 %add, !dbg !43
}

; Function Attrs: noinline norecurse optnone ssp uwtable
define i32 @main() #2 !dbg !44 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  %call = call i32 @_Z1av(), !dbg !45
  %call1 = call i32 @_Z1bv(), !dbg !45
  %add = add nsw i32 %call, %call1, !dbg !45
  ret i32 %add, !dbg !45
}

declare i32 @getchar()

attributes #0 = { noinline optnone ssp uwtable }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { noinline norecurse optnone ssp uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0 (trunk 340541) (llvm/trunk 340540)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "a.cpp", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 8.0.0 (trunk 340541) (llvm/trunk 340540)"}
!8 = distinct !DISubprogram(name: "a", linkageName: "_Z1av", scope: !1, file: !1, line: 3, type: !9, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocalVariable(name: "i", scope: !13, file: !14, line: 4, type: !11)
!13 = distinct !DILexicalBlock(scope: !15, file: !14, line: 3)
!14 = !DIFile(filename: "./header.h", directory: "/tmp")
!15 = distinct !DILexicalBlock(scope: !16, file: !14, line: 3)
!16 = distinct !DISubprogram(name: "inlined<int>", linkageName: "_Z7inlinedIiET_v", scope: !14, file: !14, line: 2, type: !9, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, templateParams: !17, retainedNodes: !2)
!17 = !{!18}
!18 = !DITemplateTypeParameter(name: "T", type: !11)
!19 = !DILocation(line: 4, scope: !13, inlinedAt: !20)
!20 = distinct !DILocation(line: 4, scope: !8)
!21 = !DILocalVariable(name: "i", scope: !22, file: !14, line: 7, type: !11)
!22 = distinct !DILexicalBlock(scope: !15, file: !14, line: 6)
!23 = !DILocation(line: 7, scope: !22, inlinedAt: !20)
!24 = !DILocalVariable(name: "a", scope: !8, file: !1, line: 4, type: !11)
!25 = !DILocation(line: 4, scope: !8)
!26 = !DILocation(line: 3, scope: !15, inlinedAt: !20)
!27 = !DILocation(line: 3, scope: !16, inlinedAt: !20)
!28 = !DILocation(line: 5, scope: !13, inlinedAt: !20)
!29 = !DILocation(line: 8, scope: !22, inlinedAt: !20)
!30 = !DILocation(line: 10, scope: !16, inlinedAt: !20)
!31 = !DILocation(line: 5, scope: !8)
!32 = distinct !DISubprogram(name: "b", linkageName: "_Z1bv", scope: !1, file: !1, line: 8, type: !9, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!33 = !DILocation(line: 4, scope: !13, inlinedAt: !34)
!34 = distinct !DILocation(line: 9, scope: !32)
!35 = !DILocation(line: 7, scope: !22, inlinedAt: !34)
!36 = !DILocalVariable(name: "b", scope: !32, file: !1, line: 9, type: !11)
!37 = !DILocation(line: 9, scope: !32)
!38 = !DILocation(line: 3, scope: !15, inlinedAt: !34)
!39 = !DILocation(line: 3, scope: !16, inlinedAt: !34)
!40 = !DILocation(line: 5, scope: !13, inlinedAt: !34)
!41 = !DILocation(line: 8, scope: !22, inlinedAt: !34)
!42 = !DILocation(line: 10, scope: !16, inlinedAt: !34)
!43 = !DILocation(line: 10, scope: !32)
!44 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 12, type: !9, isLocal: false, isDefinition: true, scopeLine: 12, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!45 = !DILocation(line: 13, scope: !44)
