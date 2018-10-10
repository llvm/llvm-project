; RUN: opt < %s -S -simplifycfg | FileCheck %s

; When SimplifyCFG changes the PHI node into a select instruction, the debug
; information becomes ambiguous. It causes the debugger to display wrong
; variable value.
;
; When in the debugger, on the line "if (read == 4)", the value of "result"
; is reported as '2', where it should be zero.

; IR generated with:
; clang -S -g -O2 -emit-llvm pr38763.cpp -o pr38763.ll -mllvm -opt-bisect-limit=10

; // pr38763.cpp
; int main() {
;   volatile int foo = 4;
;   int read = foo;
;   int read1 = foo;
; 
;   int result = 0;
;   if (read == 4) {
;     result = read1 + 2;
;   } else {
;     result = read1 - 2;
;   }
; 
;   return result;
; }

; Remove the '@llvm.dbg.value' associated with 'result' for the true/false
; branches, as they becomes ambiguous.

; CHECK-LABEL: entry
; CHECK:  %cmp = icmp eq i32 %foo.0., 4, !dbg !14
; CHECK:  %add = add nsw i32 %foo.0.4, 2, !dbg !16
; CHECK-NOT: @llvm.dbg.value(metadata i32 %add
; CHECK:  %sub = add nsw i32 %foo.0.4, -2, !dbg !16
; CHECK-NOT: @llvm.dbg.value(metadata i32 %sub
; CHECK:  %result.0 = select i1 %cmp, i32 %add, i32 %sub
; CHECK:  call void @llvm.dbg.value(metadata i32 %result.0, metadata !12, metadata !DIExpression()), !dbg !13

; ModuleID = 'pr38763.cpp'
source_filename = "pr38763.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: norecurse nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #0 !dbg !7 {
entry:
  %foo = alloca i32, align 4
  %foo.0..sroa_cast = bitcast i32* %foo to i8*
  store volatile i32 4, i32* %foo, align 4
  %foo.0. = load volatile i32, i32* %foo, align 4
  %foo.0.4 = load volatile i32, i32* %foo, align 4
  call void @llvm.dbg.value(metadata i32 0, metadata !16, metadata !DIExpression()), !dbg !27
  %cmp = icmp eq i32 %foo.0., 4, !dbg !28
  br i1 %cmp, label %if.then, label %if.else, !dbg !30

if.then:                                          ; preds = %entry
  %add = add nsw i32 %foo.0.4, 2, !dbg !31
  call void @llvm.dbg.value(metadata i32 %add, metadata !16, metadata !DIExpression()), !dbg !27
  br label %if.end

if.else:                                          ; preds = %entry
  %sub = add nsw i32 %foo.0.4, -2, !dbg !34
  call void @llvm.dbg.value(metadata i32 %sub, metadata !16, metadata !DIExpression()), !dbg !27
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %result.0 = phi i32 [ %add, %if.then ], [ %sub, %if.else ]
  call void @llvm.dbg.value(metadata i32 %result.0, metadata !16, metadata !DIExpression()), !dbg !27
  ret i32 %result.0
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0 (trunk 342209)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "pr38763.cpp", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0 (trunk 342209)"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!16}
!16 = !DILocalVariable(name: "result", scope: !7, file: !1, line: 6, type: !10)
!27 = !DILocation(line: 6, column: 7, scope: !7)
!28 = !DILocation(line: 7, column: 12, scope: !29)
!29 = distinct !DILexicalBlock(scope: !7, file: !1, line: 7, column: 7)
!30 = !DILocation(line: 7, column: 7, scope: !7)
!31 = !DILocation(line: 8, column: 20, scope: !32)
!32 = distinct !DILexicalBlock(scope: !29, file: !1, line: 7, column: 18)
!34 = !DILocation(line: 10, column: 20, scope: !35)
!35 = distinct !DILexicalBlock(scope: !29, file: !1, line: 9, column: 10)
