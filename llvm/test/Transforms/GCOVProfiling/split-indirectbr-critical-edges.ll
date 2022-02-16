; RUN: mkdir -p %t && cd %t
; RUN: opt < %s -passes=insert-gcov-profiling -S | FileCheck %s

; CHECK:       @__llvm_gcov_ctr = internal global [1 x i64] zeroinitializer

;; If an indirectbr critical edge cannot be split, ignore it.
;; The edge will not be profiled.
; CHECK-LABEL: @cannot_split(
; CHECK:       indirect.preheader:
; CHECK-NEXT:    load {{.*}} @__llvm_gcov_ctr
; CHECK-NOT:     load {{.*}} @__llvm_gcov_ctr

define dso_local i32 @cannot_split(i8* nocapture readonly %p) #0 !dbg !7 {
entry:
  %targets = alloca <2 x i8*>, align 16
  store <2 x i8*> <i8* blockaddress(@cannot_split, %indirect), i8* blockaddress(@cannot_split, %end)>, <2 x i8*>* %targets, align 16, !dbg !9
  br label %for.cond, !dbg !14

for.cond:                                         ; preds = %for.cond, %entry
  %p.addr.0 = phi i8* [ %p, %entry ], [ %incdec.ptr, %for.cond ]
  %0 = load i8, i8* %p.addr.0, align 1, !dbg !15
  %cmp = icmp eq i8 %0, 7, !dbg !17
  %incdec.ptr = getelementptr inbounds i8, i8* %p.addr.0, i64 1, !dbg !18
  br i1 %cmp, label %indirect.preheader, label %for.cond, !dbg !15, !llvm.loop !19

indirect.preheader:                               ; preds = %for.cond
  %1 = load i8, i8* %incdec.ptr, align 1, !dbg !21
  %idxprom = sext i8 %1 to i64, !dbg !21
  %arrayidx4 = getelementptr inbounds <2 x i8*>, <2 x i8*>* %targets, i64 0, i64 %idxprom, !dbg !21
  %2 = load i8*, i8** %arrayidx4, align 8, !dbg !21
  br label %indirect

indirect:                                         ; preds = %indirect.preheader, %indirect
  indirectbr i8* %2, [label %indirect, label %end]

indirect2:
  ; For this test we do not want critical edges split. Adding a 2nd `indirectbr`
  ; does the trick.
  indirectbr i8* %2, [label %indirect, label %end]

end:                                              ; preds = %indirect
  ret i32 0, !dbg !22
}

attributes #0 = { norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "a.c", directory: "/tmp/c")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!7 = distinct !DISubprogram(name: "cannot_split", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 3, column: 14, scope: !7)
!14 = !DILocation(line: 5, column: 3, scope: !7)
!15 = !DILocation(line: 6, column: 9, scope: !7)
!17 = !DILocation(line: 6, column: 12, scope: !7)
!18 = !DILocation(line: 5, column: 12, scope: !7)
!19 = distinct !{!19, !14, !20}
!20 = !DILocation(line: 9, column: 5, scope: !7)
!21 = !DILocation(line: 0, scope: !7)
!22 = !DILocation(line: 11, column: 3, scope: !7)
