; RUN: opt < %s -S -simplifycfg -hoist-common-insts=true | FileCheck %s

; SimplifyCFG can hoist any common code in the 'then' and 'else' blocks to
; the 'if' basic block.
;
; For the special case, when hoisting the terminator instruction, its debug
; location keep references to its basic block, causing the debug information
; to become ambiguous. It causes the debugger to display unreached lines.

; Check that hoisted instructions get unknown-location line numbers -- there
; is no correct line number for code that has been common'd in this way.

; IR generated with:
; clang -S -g -gno-column-info -O2 -emit-llvm pr39187.cpp -o pr39187-g.ll -mllvm -opt-bisect-limit=10

; // pr39187.cpp
; int main() {
;   volatile int foo = 0;
;
;   int beards = 0;
;   bool cond = foo == 4;
;   int bar = 0;
;   if (cond)
;     beards = 8;
;   else
;     beards = 4;
;
;   volatile bool face = cond;
;
;   return face ? beards : 0;
; }

; CHECK-LABEL: entry
; CHECK:  %foo = alloca i32, align 4
; CHECK:  %face = alloca i8, align 1
; CHECK:  %foo.0..sroa_cast = bitcast i32* %foo to i8*
; CHECK:  store volatile i32 0, i32* %foo, align 4
; CHECK:  %foo.0. = load volatile i32, i32* %foo, align 4, !dbg !16
; CHECK:  %cmp = icmp eq i32 %foo.0., 4, !dbg !16
; CHECK:  %frombool = zext i1 %cmp to i8, !dbg !16
; CHECK:  call void @llvm.dbg.value(metadata i8 %frombool, metadata !13, metadata !DIExpression()), !dbg !16
; CHECK:  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !17
; CHECK:  %. = select i1 %cmp, i32 8, i32 4, !dbg ![[MERGEDLOC:[0-9]+]]
; CHECK:  ![[MERGEDLOC]] = !DILocation(line: 0, scope: !7)

; ModuleID = 'pr39187.cpp'
source_filename = "pr39187.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: norecurse nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #0 !dbg !7 {
entry:
  %foo = alloca i32, align 4
  %face = alloca i8, align 1
  %foo.0..sroa_cast = bitcast i32* %foo to i8*
  store volatile i32 0, i32* %foo, align 4
  %foo.0. = load volatile i32, i32* %foo, align 4, !dbg !26
  %cmp = icmp eq i32 %foo.0., 4, !dbg !26
  %frombool = zext i1 %cmp to i8, !dbg !26
  call void @llvm.dbg.value(metadata i8 %frombool, metadata !15, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 0, metadata !17, metadata !DIExpression()), !dbg !27
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  call void @llvm.dbg.value(metadata i32 8, metadata !14, metadata !DIExpression()), !dbg !25
  br label %if.end, !dbg !25

if.else:                                          ; preds = %entry
  call void @llvm.dbg.value(metadata i32 4, metadata !14, metadata !DIExpression()), !dbg !27
  br label %if.end, !dbg !27

if.end:                                           ; preds = %if.else, %if.then
  %beards.0 = phi i32 [ 8, %if.then ], [ 4, %if.else ]
  store volatile i8 %frombool, i8* %face, align 1
  %face.0. = load volatile i8, i8* %face, align 1
  %0 = and i8 %face.0., 1
  %tobool3 = icmp eq i8 %0, 0
  %cond4 = select i1 %tobool3, i32 0, i32 %beards.0
  ret i32 %cond4
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0 (trunk 346301)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "pr39187.cpp", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0 (trunk 346301)"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!14, !15, !17}
!14 = !DILocalVariable(name: "beards", scope: !7, file: !1, line: 4, type: !10)
!15 = !DILocalVariable(name: "cond", scope: !7, file: !1, line: 5, type: !16)
!16 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!17 = !DILocalVariable(name: "bar", scope: !7, file: !1, line: 6, type: !10)
!25 = !DILocation(line: 4, scope: !7)
!26 = !DILocation(line: 5, scope: !7)
!27 = !DILocation(line: 6, scope: !7)
