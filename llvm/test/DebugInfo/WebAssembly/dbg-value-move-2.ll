; FIXME: use -stop-after when MIR serialization output includes needed debug info.
; RUN: llc < %s -print-after=wasm-reg-stackify 2>&1 | FileCheck %s

; CHECK: {{.*}}After WebAssembly Register Stackify
; CHECK: bb.2.for.body:
; CHECK: [[TEEREG:%[0-9]+]]:i32, %{{[0-9]+}}:i32 = TEE_I32 {{.*}} fib2.c:6:7
; CHECK-NEXT: DBG_VALUE [[TEEREG]]:i32, $noreg, !"a", {{.*}} fib2.c:2:13

; ModuleID = 'fib2.bc'
; The test generated via: clang --target=wasm32-unknown-unknown-wasm fib2.c -g -O2
; All non-"!dbg !18" llvm.dbg.value calls and attributes were removed.
source_filename = "fib2.c"
; int fib(int n) {
;   int i, t, a = 0, b = 1;
;   for (i = 0; i < n; i++) {
;     t = a;
;     a = b;
;     b += t;
;   }
;   return b;
; }
target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

; Function Attrs: nounwind readnone
define hidden i32 @fib(i32 %n) local_unnamed_addr #0 !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !18
  %cmp8 = icmp sgt i32 %n, 0, !dbg !21
  br i1 %cmp8, label %for.body, label %for.end, !dbg !24

for.body:                                         ; preds = %entry, %for.body
  %b.011 = phi i32 [ %add, %for.body ], [ 1, %entry ]
  %a.010 = phi i32 [ %b.011, %for.body ], [ 0, %entry ]
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  call void @llvm.dbg.value(metadata i32 %a.010, metadata !15, metadata !DIExpression()), !dbg !18
  %add = add nsw i32 %b.011, %a.010, !dbg !26
  %inc = add nuw nsw i32 %i.09, 1, !dbg !28
  call void @llvm.dbg.value(metadata i32 %b.011, metadata !15, metadata !DIExpression()), !dbg !18
  %exitcond = icmp eq i32 %inc, %n, !dbg !21
  br i1 %exitcond, label %for.end, label %for.body, !dbg !24, !llvm.loop !29

for.end:                                          ; preds = %for.body, %entry
  %b.0.lcssa = phi i32 [ 1, %entry ], [ %add, %for.body ]
  ret i32 %b.0.lcssa, !dbg !31
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 7.0.0 (trunk 337180)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "fib2.c", directory: "/d/y/llvmwasm")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 7.0.0 (trunk 337180)"}
!7 = distinct !DISubprogram(name: "fib", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!15}
!15 = !DILocalVariable(name: "a", scope: !7, file: !1, line: 2, type: !10)
!17 = !DILocation(line: 1, column: 13, scope: !7)
!18 = !DILocation(line: 2, column: 13, scope: !7)
!19 = !DILocation(line: 2, column: 20, scope: !7)
!20 = !DILocation(line: 2, column: 7, scope: !7)
!21 = !DILocation(line: 3, column: 17, scope: !22)
!22 = distinct !DILexicalBlock(scope: !23, file: !1, line: 3, column: 3)
!23 = distinct !DILexicalBlock(scope: !7, file: !1, line: 3, column: 3)
!24 = !DILocation(line: 3, column: 3, scope: !23)
!25 = !DILocation(line: 2, column: 10, scope: !7)
!26 = !DILocation(line: 6, column: 7, scope: !27)
!27 = distinct !DILexicalBlock(scope: !22, file: !1, line: 3, column: 27)
!28 = !DILocation(line: 3, column: 23, scope: !22)
!29 = distinct !{!29, !24, !30}
!30 = !DILocation(line: 7, column: 3, scope: !23)
!31 = !DILocation(line: 8, column: 3, scope: !7)
