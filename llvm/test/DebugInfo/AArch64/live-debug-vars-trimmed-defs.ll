; RUN: llc -mtriple=aarch64-unknown-linux-gnu -O2 -stop-after=virtregrewriter %s -o - | FileCheck %s

; UserValue::trimmedDefs records interval start indexes that computeIntervals
; trimmed to a lexical scope, and emitDebugValues matches them against the
; starts in locInts to decide whether to step a DBG_VALUE back one index.
;
; Nothing tells LiveDebugVariables when an instruction is erased during
; register allocation, so here both a trimmed start and the locInts start it
; belongs to are left naming an instruction that has been erased. The
; locations still come out in the right place, so this test pins down that
; currently correct output.
;
; Reduced from MultiSource/Benchmarks/DOE-ProxyApps-C++/PENNANT built at -O2 -g.

; CHECK-LABEL: name: map_index
; CHECK: DBG_VALUE $x0, $noreg, ![[VAR:[0-9]+]], !DIExpression()
; CHECK: DBG_VALUE $x{{[0-9]+}}, $noreg, ![[VAR]], !DIExpression()

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

define ptr @map_index(ptr %this, ptr %k, ptr %p0, ptr %x.addr.1, ptr %p1, i64 %n, i1 %cmp.i) !dbg !4 {
entry:
    #dbg_value(ptr %this, !10, !DIExpression(), !17)
  %add.ptr = getelementptr i8, ptr %this, i64 8, !dbg !27
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %x.addr.011 = phi ptr [ null, %entry ], [ %x.addr.1, %while.body ]
  %call = call i32 @memcmp(ptr %p1)
  %sub = sub i64 0, %n
  %spec.select = tail call i64 @llvm.smax.i64(i64 %sub, i64 1)
  %retval.07 = tail call i64 @llvm.smin.i64(i64 %spec.select, i64 2147483647)
  br i1 %cmp.i, label %lower_bound.exit, label %while.body

lower_bound.exit:                                 ; preds = %while.body
  %retval.0.i12 = trunc i64 %retval.07 to i32
  %cmp.i3 = icmp slt i32 %retval.0.i12, 0
  %y.addr.1 = select i1 %cmp.i3, ptr null, ptr %x.addr.011
  %cmp = icmp eq ptr %y.addr.1, %add.ptr
  br i1 %cmp, label %if.then, label %common.ret

common.ret:                                       ; preds = %if.then, %lower_bound.exit
  ret ptr null

if.then:                                          ; preds = %lower_bound.exit
  store ptr %k, ptr %p0, align 8
  %call12 = load volatile ptr, ptr %this, align 8
  br label %common.ret
}

declare i32 @memcmp(ptr)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, emissionKind: FullDebug)
!1 = !DIFile(filename: "InputFile.cc", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "operator[]", scope: !6, file: !5, line: 504, type: !8, scopeLine: 505, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DIFile(filename: "stl_map.h", directory: "/")
!6 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "map", scope: !7, file: !5, line: 102, size: 384, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !2)
!7 = !DINamespace(name: "std", scope: null)
!8 = distinct !DISubroutineType(types: !2)
!10 = !DILocalVariable(name: "this", arg: 1, scope: !11, type: !16, flags: DIFlagArtificial | DIFlagObjectPointer)
!11 = distinct !DISubprogram(name: "_M_end", scope: !13, file: !12, line: 747, type: !14, scopeLine: 748, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!12 = !DIFile(filename: "stl_tree.h", directory: "/")
!13 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "_Rb_tree", scope: !7, file: !12, line: 423, size: 384, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !2)
!14 = distinct !DISubroutineType(types: !2)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!17 = distinct !DILocation(line: 0, scope: !11, inlinedAt: !18)
!18 = distinct !DILocation(line: 1267, column: 43, scope: !19, inlinedAt: !22)
!19 = distinct !DISubprogram(name: "lower_bound", scope: !13, file: !12, line: 1266, type: !20, scopeLine: 1267, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!20 = distinct !DISubroutineType(types: !2)
!22 = distinct !DILocation(line: 1308, column: 21, scope: !23, inlinedAt: !26)
!23 = distinct !DISubprogram(name: "lower_bound", scope: !6, file: !5, line: 1307, type: !24, scopeLine: 1308, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!24 = distinct !DISubroutineType(types: !2)
!26 = distinct !DILocation(line: 509, column: 17, scope: !4)
!27 = distinct !DILocation(line: 748, column: 17, scope: !11, inlinedAt: !18)
