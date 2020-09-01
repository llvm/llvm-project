; RUN: llc -O0 %s -o - -filetype=obj \
; RUN:   | llvm-dwarfdump -statistics - | FileCheck %s

; Test that abstract origins in multiple CUs are uniqued.

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
;b.cpp:
;#include <header.h>
;int b() {
;  int b = inlined<int>();
;  return b+1;
;}
;a.cpp
;#include <header.h>
;int b();
;int a() {
;  int a = inlined<int>();
;  return a+1;
;}
; 
;int main() {
;  return a() + b();
;}

; ModuleID = 'linked.ll'
source_filename = "llvm-link"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; Function Attrs: noinline optnone ssp uwtable
define i32 @_Z1av() #0 !dbg !10 {
entry:
  %retval.i = alloca i32, align 4
  %i.i = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %i.i, metadata !14, metadata !DIExpression()), !dbg !21
  %i2.i = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %i2.i, metadata !23, metadata !DIExpression()), !dbg !25
  %a = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %a, metadata !26, metadata !DIExpression()), !dbg !27
  %call.i = call i32 @getchar(), !dbg !28
  %cmp.i = icmp eq i32 %call.i, 97, !dbg !28
  br i1 %cmp.i, label %if.then.i, label %if.else.i, !dbg !29

if.then.i:                                        ; preds = %entry
  %call1.i = call i32 @getchar(), !dbg !21
  store i32 %call1.i, i32* %i.i, align 4, !dbg !21
  %0 = load i32, i32* %i.i, align 4, !dbg !30
  store i32 %0, i32* %retval.i, align 4, !dbg !30
  br label %_Z7inlinedIiET_v.exit, !dbg !30

if.else.i:                                        ; preds = %entry
  store i32 97, i32* %i2.i, align 4, !dbg !25
  %1 = load i32, i32* %i2.i, align 4, !dbg !31
  store i32 %1, i32* %retval.i, align 4, !dbg !31
  br label %_Z7inlinedIiET_v.exit, !dbg !31

_Z7inlinedIiET_v.exit:                            ; preds = %if.else.i, %if.then.i
  %2 = load i32, i32* %retval.i, align 4, !dbg !32
  store i32 %2, i32* %a, align 4, !dbg !27
  %3 = load i32, i32* %a, align 4, !dbg !33
  %add = add nsw i32 %3, 1, !dbg !33
  ret i32 %add, !dbg !33
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i32 @getchar()

; Function Attrs: noinline norecurse optnone ssp uwtable
define i32 @main() #3 !dbg !34 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  %call = call i32 @_Z1av(), !dbg !35
  %call1 = call i32 @_Z1bv(), !dbg !35
  %add = add nsw i32 %call, %call1, !dbg !35
  ret i32 %add, !dbg !35
}

; Function Attrs: noinline optnone ssp uwtable
define i32 @_Z1bv() #0 !dbg !36 {
entry:
  %retval.i = alloca i32, align 4
  %i.i = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %i.i, metadata !37, metadata !DIExpression()), !dbg !41
  %i2.i = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %i2.i, metadata !43, metadata !DIExpression()), !dbg !45
  %b = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %b, metadata !46, metadata !DIExpression()), !dbg !47
  %call.i = call i32 @getchar(), !dbg !48
  %cmp.i = icmp eq i32 %call.i, 97, !dbg !48
  br i1 %cmp.i, label %if.then.i, label %if.else.i, !dbg !49

if.then.i:                                        ; preds = %entry
  %call1.i = call i32 @getchar(), !dbg !41
  store i32 %call1.i, i32* %i.i, align 4, !dbg !41
  %0 = load i32, i32* %i.i, align 4, !dbg !50
  store i32 %0, i32* %retval.i, align 4, !dbg !50
  br label %_Z7inlinedIiET_v.exit, !dbg !50

if.else.i:                                        ; preds = %entry
  store i32 97, i32* %i2.i, align 4, !dbg !45
  %1 = load i32, i32* %i2.i, align 4, !dbg !51
  store i32 %1, i32* %retval.i, align 4, !dbg !51
  br label %_Z7inlinedIiET_v.exit, !dbg !51

_Z7inlinedIiET_v.exit:                            ; preds = %if.else.i, %if.then.i
  %2 = load i32, i32* %retval.i, align 4, !dbg !52
  store i32 %2, i32* %b, align 4, !dbg !47
  %3 = load i32, i32* %b, align 4, !dbg !53
  %add = add nsw i32 %3, 1, !dbg !53
  ret i32 %add, !dbg !53
}

attributes #0 = { noinline optnone ssp uwtable }
attributes #1 = { nounwind readnone speculatable }
attributes #3 = { noinline norecurse optnone ssp uwtable }

!llvm.dbg.cu = !{!0, !3}
!llvm.ident = !{!5, !5}
!llvm.module.flags = !{!6, !7, !8, !9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0 (trunk 340541) (llvm/trunk 340540)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "a.cpp", directory: "/tmp")
!2 = !{}
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "clang version 8.0.0 (trunk 340541) (llvm/trunk 340540)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!4 = !DIFile(filename: "b.cpp", directory: "/tmp")
!5 = !{!"clang version 8.0.0 (trunk 340541) (llvm/trunk 340540)"}
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 7, !"PIC Level", i32 2}
!10 = distinct !DISubprogram(name: "a", linkageName: "_Z1av", scope: !1, file: !1, line: 3, type: !11, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DILocalVariable(name: "i", scope: !15, file: !16, line: 4, type: !13)
!15 = distinct !DILexicalBlock(scope: !17, file: !16, line: 3)
!16 = !DIFile(filename: "./header.h", directory: "/tmp")
!17 = distinct !DILexicalBlock(scope: !18, file: !16, line: 3)
!18 = distinct !DISubprogram(name: "inlined<int>", linkageName: "_Z7inlinedIiET_v", scope: !16, file: !16, line: 2, type: !11, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, templateParams: !19, retainedNodes: !2)
!19 = !{!20}
!20 = !DITemplateTypeParameter(name: "T", type: !13)
!21 = !DILocation(line: 4, scope: !15, inlinedAt: !22)
!22 = distinct !DILocation(line: 4, scope: !10)
!23 = !DILocalVariable(name: "i", scope: !24, file: !16, line: 7, type: !13)
!24 = distinct !DILexicalBlock(scope: !17, file: !16, line: 6)
!25 = !DILocation(line: 7, scope: !24, inlinedAt: !22)
!26 = !DILocalVariable(name: "a", scope: !10, file: !1, line: 4, type: !13)
!27 = !DILocation(line: 4, scope: !10)
!28 = !DILocation(line: 3, scope: !17, inlinedAt: !22)
!29 = !DILocation(line: 3, scope: !18, inlinedAt: !22)
!30 = !DILocation(line: 5, scope: !15, inlinedAt: !22)
!31 = !DILocation(line: 8, scope: !24, inlinedAt: !22)
!32 = !DILocation(line: 10, scope: !18, inlinedAt: !22)
!33 = !DILocation(line: 5, scope: !10)
!34 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 12, type: !11, isLocal: false, isDefinition: true, scopeLine: 12, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!35 = !DILocation(line: 13, scope: !34)
!36 = distinct !DISubprogram(name: "b", linkageName: "_Z1bv", scope: !4, file: !4, line: 2, type: !11, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !3, retainedNodes: !2)
!37 = !DILocalVariable(name: "i", scope: !38, file: !16, line: 4, type: !13)
!38 = distinct !DILexicalBlock(scope: !39, file: !16, line: 3)
!39 = distinct !DILexicalBlock(scope: !40, file: !16, line: 3)
!40 = distinct !DISubprogram(name: "inlined<int>", linkageName: "_Z7inlinedIiET_v", scope: !16, file: !16, line: 2, type: !11, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !3, templateParams: !19, retainedNodes: !2)
!41 = !DILocation(line: 4, scope: !38, inlinedAt: !42)
!42 = distinct !DILocation(line: 3, scope: !36)
!43 = !DILocalVariable(name: "i", scope: !44, file: !16, line: 7, type: !13)
!44 = distinct !DILexicalBlock(scope: !39, file: !16, line: 6)
!45 = !DILocation(line: 7, scope: !44, inlinedAt: !42)
!46 = !DILocalVariable(name: "b", scope: !36, file: !4, line: 3, type: !13)
!47 = !DILocation(line: 3, scope: !36)
!48 = !DILocation(line: 3, scope: !39, inlinedAt: !42)
!49 = !DILocation(line: 3, scope: !40, inlinedAt: !42)
!50 = !DILocation(line: 5, scope: !38, inlinedAt: !42)
!51 = !DILocation(line: 8, scope: !44, inlinedAt: !42)
!52 = !DILocation(line: 10, scope: !40, inlinedAt: !42)
!53 = !DILocation(line: 4, scope: !36)
