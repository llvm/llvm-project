; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/inline-hot-callsite-threshold.prof -S -pass-remarks=sample-profile -sample-profile-hot-inline-threshold=100 2>&1 | FileCheck %s

; CHECK: remark: a.cc:6:12: 'bar' inlined into 'foo' to match profiling context with (cost={{.*}}, threshold=100)
; CHECK:     define dso_local noundef i32 @foo(i32 noundef %0)
; CHECK-NOT:   %2 = tail call noundef i32 @bar(i32 noundef %0)
; CHECK-NEXT:  %2 = icmp sgt i32 %0, 1
; CHECK-NEXT:  br i1 %2, label %3, label %bar.exit

; Manually lower cost threshold for hot function inlining, so that the function
; is not inlined even profile indicates it as hot.
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/inline-hot-callsite-threshold.prof -S -pass-remarks=sample-profile -sample-profile-hot-inline-threshold=1 2>&1 | FileCheck %s --check-prefix=COST

; COST-NOT:  remark
; COST: define dso_local noundef i32 @foo(i32 noundef %0)
; COST-NEXT: %2 = tail call noundef i32 @bar(i32 noundef %0)

define dso_local noundef i32 @bar(i32 noundef %0) #0 !dbg !10 {
  %2 = icmp sgt i32 %0, 1
  br i1 %2, label %3, label %15
3:                                                ; preds = %1
  %4 = add nsw i32 %0, -2
  %5 = mul i32 %4, %4
  %6 = add i32 %5, %0
  %7 = zext nneg i32 %4 to i33
  %8 = add nsw i32 %0, -3
  %9 = zext i32 %8 to i33
  %10 = mul i33 %7, %9
  %11 = lshr i33 %10, 1
  %12 = trunc nuw i33 %11 to i32
  %13 = xor i32 %12, -1
  %14 = add i32 %6, %13
  br label %15
15:                                               ; preds = %3, %1
  %16 = phi i32 [ 0, %1 ], [ %14, %3 ]
  ret i32 %16
}

define dso_local noundef i32 @foo(i32 noundef %0) #1 !dbg !20 {
  %2 = tail call noundef i32 @bar(i32 noundef %0), !dbg !24
  ret i32 %2
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable  "use-sample-profile" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable  "use-sample-profile" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug)
!1 = !DIFile(filename: "a.cc", directory: ".")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!10 = distinct !DISubprogram(name: "bar", linkageName: "bar", scope: !1, file: !1, line: 1, type: !12, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0)
!11 = !DIFile(filename: "a.cc", directory: ".")
!12 = !DISubroutineType(types: !13)
!13 = !{!14, !14}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!20 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !11, file: !11, line: 5, type: !12, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0)
!23 = !DILocation(line: 0, scope: !20)
!24 = !DILocation(line: 6, column: 12, scope: !20)
