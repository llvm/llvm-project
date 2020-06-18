; Original C++ test case
;
; #include <stdio.h>
;
; __attribute__((noinline)) int goo() { return 3 };
; __attribute__((noinline)) int hoo() { return 4 };
;
; int sum(int x, int y) {
;   return x + y;
; }
;
; int main() {
;   int s, i = 0;
;   while (i++ < 20000 * 20000)
;     if (i != 100) s = sum(i, s); else s = 30;
;   printf("sum is %d\n", s);
;   return goo() + hoo() != 7;
; }
;
; Both goo and hoo don't show up in the input profile.
; Suppose function goo shows up in the binary generating the input profile
; and function hoo doesn't show up. Then the profile symbol list in the input
; profile will contain goo but not hoo. Verify the entry count of goo is
; 0 and the entry count of hoo is -1.
; CHECK: define {{.*}} i32 @_Z3goov() {{.*}} !prof ![[IDX1:[0-9]*]]
; CHECK: define {{.*}} i32 @_Z3hoov() {{.*}} !prof ![[IDX2:[0-9]*]]
; CHECK: ![[IDX1]] = !{!"function_entry_count", i64 0}
; CHECK: ![[IDX2]] = !{!"function_entry_count", i64 -1}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [11 x i8] c"sum is %d\0A\00", align 1

; Function Attrs: noinline norecurse nounwind readnone uwtable
define dso_local i32 @_Z3goov() local_unnamed_addr #0 !dbg !7 {
entry:
  ret i32 3, !dbg !9
}

; Function Attrs: noinline norecurse nounwind readnone uwtable
define dso_local i32 @_Z3hoov() local_unnamed_addr #0 !dbg !10 {
entry:
  ret i32 4, !dbg !11
}

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local i32 @_Z3sumii(i32 %x, i32 %y) local_unnamed_addr #1 !dbg !12 {
entry:
  %add = add nsw i32 %y, %x, !dbg !13
  ret i32 %add, !dbg !14
}

; Function Attrs: nofree norecurse nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #2 !dbg !15 {
entry:
  br label %while.body, !dbg !16

while.body:                                       ; preds = %while.body, %entry
  %inc12 = phi i32 [ 1, %entry ], [ %inc.4, %while.body ]
  %s.011 = phi i32 [ undef, %entry ], [ %spec.select.4, %while.body ]
  %cmp1 = icmp eq i32 %inc12, 100, !dbg !18
  %add.i = add nsw i32 %inc12, %s.011, !dbg !20
  %spec.select = select i1 %cmp1, i32 30, i32 %add.i, !dbg !23
  %inc = add nuw nsw i32 %inc12, 1, !dbg !24
  %cmp1.1 = icmp eq i32 %inc, 100, !dbg !18
  %add.i.1 = add nsw i32 %inc, %spec.select, !dbg !20
  %spec.select.1 = select i1 %cmp1.1, i32 30, i32 %add.i.1, !dbg !23
  %inc.1 = add nuw nsw i32 %inc12, 2, !dbg !24
  %cmp1.2 = icmp eq i32 %inc.1, 100, !dbg !18
  %add.i.2 = add nsw i32 %inc.1, %spec.select.1, !dbg !20
  %spec.select.2 = select i1 %cmp1.2, i32 30, i32 %add.i.2, !dbg !23
  %inc.2 = add nuw nsw i32 %inc12, 3, !dbg !24
  %cmp1.3 = icmp eq i32 %inc.2, 100, !dbg !18
  %add.i.3 = add nsw i32 %inc.2, %spec.select.2, !dbg !20
  %spec.select.3 = select i1 %cmp1.3, i32 30, i32 %add.i.3, !dbg !23
  %inc.3 = add nuw nsw i32 %inc12, 4, !dbg !24
  %cmp1.4 = icmp eq i32 %inc.3, 100, !dbg !18
  %add.i.4 = add nsw i32 %inc.3, %spec.select.3, !dbg !20
  %spec.select.4 = select i1 %cmp1.4, i32 30, i32 %add.i.4, !dbg !23
  %inc.4 = add nuw nsw i32 %inc12, 5, !dbg !24
  %exitcond.4 = icmp eq i32 %inc.4, 400000001, !dbg !26
  br i1 %exitcond.4, label %while.end, label %while.body, !dbg !27, !llvm.loop !28

while.end:                                        ; preds = %while.body
  %call2 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), i32 %spec.select.4), !dbg !31
  ret i32 0, !dbg !32
}

; Function Attrs: nofree nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #3

attributes #0 = { noinline norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" "use-sample-profile" }
attributes #1 = { norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" "use-sample-profile" }
attributes #2 = { nofree norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" "use-sample-profile" }
attributes #3 = { nofree nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 10.0.0 (trunk 369144)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "1.cc", directory: "/usr/local/google/home/wmi/workarea/llvm-r369144/src")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 10.0.0 (trunk 369144)"}
!7 = distinct !DISubprogram(name: "goo", linkageName: "_Z3goov", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 3, column: 39, scope: !7)
!10 = distinct !DISubprogram(name: "hoo", linkageName: "_Z3hoov", scope: !1, file: !1, line: 4, type: !8, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!11 = !DILocation(line: 4, column: 39, scope: !10)
!12 = distinct !DISubprogram(name: "sum", linkageName: "_Z3sumii", scope: !1, file: !1, line: 6, type: !8, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!13 = !DILocation(line: 7, column: 12, scope: !12)
!14 = !DILocation(line: 7, column: 3, scope: !12)
!15 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 10, type: !8, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!16 = !DILocation(line: 12, column: 3, scope: !17)
!17 = !DILexicalBlockFile(scope: !15, file: !1, discriminator: 2)
!18 = !DILocation(line: 13, column: 11, scope: !19)
!19 = !DILexicalBlockFile(scope: !15, file: !1, discriminator: 21)
!20 = !DILocation(line: 7, column: 12, scope: !21, inlinedAt: !22)
!21 = !DILexicalBlockFile(scope: !12, file: !1, discriminator: 21)
!22 = distinct !DILocation(line: 13, column: 23, scope: !17)
!23 = !DILocation(line: 13, column: 9, scope: !19)
!24 = !DILocation(line: 12, column: 11, scope: !25)
!25 = !DILexicalBlockFile(scope: !15, file: !1, discriminator: 1282)
!26 = !DILocation(line: 12, column: 14, scope: !25)
!27 = !DILocation(line: 12, column: 3, scope: !25)
!28 = distinct !{!28, !29, !30}
!29 = !DILocation(line: 12, column: 3, scope: !15)
!30 = !DILocation(line: 13, column: 43, scope: !15)
!31 = !DILocation(line: 14, column: 3, scope: !15)
!32 = !DILocation(line: 15, column: 3, scope: !15)
