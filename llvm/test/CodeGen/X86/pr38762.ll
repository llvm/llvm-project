; RUN: opt < %s -S -simplifycfg | FileCheck %s

; Note: This patch is a complement to pr38763.
;
; When SimplifyCFG changes the PHI node into a select instruction, the debug
; information becomes ambiguous. It causes the debugger to display unreached
; lines and invalid variable values.
;
; When in the debugger, on the line "if (read1 > 3)", and we step from the
; 'if' condition, onto the addition, then back to the 'if' again, which is
; misleading because that addition doesn't really "happen" (it's speculated).

; IR generated with:
; clang -S -g -gno-column-info -O2 -emit-llvm pr38762.cpp -o pr38762.ll -mllvm -opt-bisect-limit=10

; // pr38762.cpp
; int main() {
;   volatile int foo = 0;
;   int read1 = foo;
;   int brains = foo;
; 
;   if (read1 > 3) {
;     brains *= 2;
;     brains += 1;
;   }
; 
;   return brains;
; }

; Change the debug locations associated with the PHI nodes being promoted, to
; the debug locations from the insertion point in the dominant block.

; CHECK-LABEL: entry
; CHECK:  %cmp = icmp sgt i32 %foo.0., 3, !dbg !14
; CHECK:  %mul = shl nsw i32 %foo.0.5, 1, !dbg !16
; CHECK-NOT:  call void @llvm.dbg.value(metadata i32 %mul, metadata !15, metadata !DIExpression()), !dbg !25
; CHECK:  %add = or i32 %mul, 1, !dbg !16
; CHECK-NOT:  call void @llvm.dbg.value(metadata i32 %add, metadata !15, metadata !DIExpression()), !dbg !25
; CHECK:  %brains.0 = select i1 %cmp, i32 %add, i32 %foo.0.5, !dbg !16

; ModuleID = 'pr38762.cpp'
source_filename = "pr38762.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: norecurse nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #0 !dbg !7 {
entry:
  %foo = alloca i32, align 4
  %foo.0..sroa_cast = bitcast i32* %foo to i8*
  store volatile i32 0, i32* %foo, align 4
  %foo.0. = load volatile i32, i32* %foo, align 4
  %foo.0.5 = load volatile i32, i32* %foo, align 4
  call void @llvm.dbg.value(metadata i32 %foo.0.5, metadata !15, metadata !DIExpression()), !dbg !25
  %cmp = icmp sgt i32 %foo.0., 3, !dbg !26
  br i1 %cmp, label %if.then, label %if.end, !dbg !28

if.then:                                          ; preds = %entry
  %mul = shl nsw i32 %foo.0.5, 1, !dbg !29
  call void @llvm.dbg.value(metadata i32 %mul, metadata !15, metadata !DIExpression()), !dbg !25
  %add = or i32 %mul, 1, !dbg !31
  call void @llvm.dbg.value(metadata i32 %add, metadata !15, metadata !DIExpression()), !dbg !25
  br label %if.end, !dbg !32

if.end:                                           ; preds = %if.then, %entry
  %brains.0 = phi i32 [ %add, %if.then ], [ %foo.0.5, %entry ], !dbg !33
  call void @llvm.dbg.value(metadata i32 %brains.0, metadata !15, metadata !DIExpression()), !dbg !25
  ret i32 %brains.0, !dbg !35
}

declare void @llvm.dbg.value(metadata, metadata, metadata) #2

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0 (trunk 343753)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "pr38762.cpp", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0 (trunk 343753)"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!15}
!13 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !10)
!15 = !DILocalVariable(name: "brains", scope: !7, file: !1, line: 4, type: !10)
!25 = !DILocation(line: 4, scope: !7)
!26 = !DILocation(line: 6, scope: !27)
!27 = distinct !DILexicalBlock(scope: !7, file: !1, line: 6)
!28 = !DILocation(line: 6, scope: !7)
!29 = !DILocation(line: 7, scope: !30)
!30 = distinct !DILexicalBlock(scope: !27, file: !1, line: 6)
!31 = !DILocation(line: 8, scope: !30)
!32 = !DILocation(line: 9, scope: !30)
!33 = !DILocation(line: 0, scope: !7)
!34 = !DILocation(line: 12, scope: !7)
!35 = !DILocation(line: 11, scope: !7)
