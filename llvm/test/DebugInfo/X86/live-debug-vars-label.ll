; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O2 -stop-after=livedebugvalues %s -o - | FileCheck %s

; LiveDebugVariables records the SlotIndex of a DBG_LABEL in UserLabel::loc and
; re-inserts the label there after register allocation. Nothing tells it when
; an instruction is erased in between, so here loc is left naming one that has
; been erased. The label still comes out in the right place because
; insertDebugLabel goes through findInsertLocation, which scans backwards from
; the recorded index for a surviving instruction to insert after, so this test
; pins down that currently correct output.
;
; Reduced from MultiSource/Benchmarks/Prolangs-C/cdecl built at -O2 -g.

; CHECK-LABEL: name: yyparse
; CHECK: DBG_LABEL !{{[0-9]+}}, debug-location !{{[0-9]+}}

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @yyparse(ptr %add.ptr132, i16 %0, i16 %1, i1 %cmp.not, i1 %cmp8) !dbg !4 {
entry:
    #dbg_label(!7, !8)
  br label %if.end

if.then:                                          ; preds = %yystack.backedge
  ret i32 0

if.end:                                           ; preds = %yystack.backedge, %entry
  %yystate.01028 = phi i16 [ 0, %entry ], [ %0, %yystack.backedge ]
  store i16 %yystate.01028, ptr null, align 2
  br i1 %cmp8, label %if.then10, label %if.end15

if.then10:                                        ; preds = %if.end
  %call = tail call i32 null()
  br label %if.end15

if.end15:                                         ; preds = %if.then10, %if.end
  tail call void @llvm.memcpy.p0.p0.i64(ptr null, ptr %add.ptr132, i64 1, i1 false)
  switch i16 %1, label %yystack.backedge [
    i16 2, label %sw.bb162
    i16 4, label %sw.bb164
  ]

yystack.backedge:                                 ; preds = %if.end15
  br i1 %cmp.not, label %if.end, label %if.then

sw.bb162:                                         ; preds = %if.end15
  ret i32 0

sw.bb164:                                         ; preds = %if.end15
  ret i32 0
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C89, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "cdecl.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "yyparse", scope: !5, file: !5, line: 1022, type: !6, scopeLine: 1022, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2, keyInstructions: true)
!5 = !DIFile(filename: "cdgram.y", directory: "/")
!6 = !DISubroutineType(types: !2)
!7 = !DILabel(scope: !4, name: "yystack", file: !5, line: 1038, column: 2)
!8 = !DILocation(line: 1038, column: 2, scope: !4)
