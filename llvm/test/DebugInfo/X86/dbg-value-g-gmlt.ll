; RUN: llc < %s -filetype=obj | llvm-dwarfdump - --debug-info | FileCheck %s
;
; IR module created as follows:
;   clang -emit-llvm -S -O2 foo.cpp -o foo.ll -g
;   clang -emit-llvm -S -O2 bar.cpp -o bar.ll -gmlt
;   llvm-link foo.ll bar.ll -S -o linked.ll
;   opt -std-link-opts linked.ll -S -o opt.ll
; --- foo.cpp ---
; void f();
; void foo(int param) {
;   if (param) f();
; }
; --- bar.cpp ---
; void foo(int);
; void bar() {
;   foo(0);
; }
; ---
; The point is that bar() is compiled -gmlt and calls foo() with a constant 0.
; foo() is compiled -g and gets inlined into bar(); foo's body is then
; optimized away, leaving only a dbg.value call describing the inlined copy
; of 'param', which should be benign.
; That is, the compile-unit for bar.cpp should have nothing in it.
; PR31437.

; foo.cpp's unit comes first; skip past it, second unit should be empty.
; CHECK:     DW_TAG_compile_unit
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     DW_AT_name {{.*}} "foo.cpp"
; CHECK:     DW_TAG_compile_unit
; CHECK-NOT: DW_TAG

; ModuleID = 'linked.ll'
source_filename = "llvm-link"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: uwtable
define void @_Z3fooi(i32 %param) local_unnamed_addr #0 !dbg !8 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %param, metadata !13, metadata !14), !dbg !15
  %tobool = icmp eq i32 %param, 0, !dbg !16
  br i1 %tobool, label %if.end, label %if.then, !dbg !18

if.then:                                          ; preds = %entry
  tail call void @_Z1fv(), !dbg !19
  br label %if.end, !dbg !19

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !21
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

declare void @_Z1fv() local_unnamed_addr #2

; Function Attrs: nounwind readnone uwtable
define void @_Z3barv() local_unnamed_addr #3 !dbg !22 {
entry:
  tail call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !14), !dbg !24
  ret void, !dbg !26
}

attributes #0 = { uwtable }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone uwtable }

!llvm.dbg.cu = !{!0, !3}
!llvm.ident = !{!5, !5}
!llvm.module.flags = !{!6, !7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 5.0.0 (trunk 293745)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "foo.cpp", directory: "/home/probinson/projects/scratch/pr31437")
!2 = !{}
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "clang version 5.0.0 (trunk 293745)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!4 = !DIFile(filename: "bar.cpp", directory: "/home/probinson/projects/scratch/pr31437")
!5 = !{!"clang version 5.0.0 (trunk 293745)"}
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !1, file: !1, line: 2, type: !9, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DILocalVariable(name: "param", arg: 1, scope: !8, file: !1, line: 2, type: !11)
!14 = !DIExpression()
!15 = !DILocation(line: 2, column: 14, scope: !8)
!16 = !DILocation(line: 3, column: 7, scope: !17)
!17 = distinct !DILexicalBlock(scope: !8, file: !1, line: 3, column: 7)
!18 = !DILocation(line: 3, column: 7, scope: !8)
!19 = !DILocation(line: 3, column: 14, scope: !20)
!20 = !DILexicalBlockFile(scope: !17, file: !1, discriminator: 1)
!21 = !DILocation(line: 4, column: 1, scope: !8)
!22 = distinct !DISubprogram(name: "bar", scope: !4, file: !4, line: 2, type: !23, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !3, variables: !2)
!23 = !DISubroutineType(types: !2)
!24 = !DILocation(line: 2, column: 14, scope: !8, inlinedAt: !25)
!25 = distinct !DILocation(line: 3, column: 3, scope: !22)
!26 = !DILocation(line: 4, column: 1, scope: !22)
