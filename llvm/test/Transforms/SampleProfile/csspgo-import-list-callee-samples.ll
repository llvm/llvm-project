; RUN: opt < %s -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -sample-profile-file=%S/Inputs/csspgo-import-list-callee-samples.prof  -S | FileCheck %s

; Original source code:

; file1.c:
;
; int a = 1;
; int bar(int x);
;
; int(*foo())()  {
;  a++;
;  return bar;
; };

; __attribute__((noinline)) void func()
; {
;     int (*fptr)(int);
;     fptr = foo();
;     a +=  (*fptr)(10);
; }

; int main() {
;   for(int i = 0; i < 1000 * 1000; i++)
;     func();
; }

; file2.c:
;
; int bar(int x) { return x + 1;}



; GUID for bar is -2012135647395072713, make sure bar is not imported.
; CHECK: ![[#]] = !{!"function_entry_count", i64 1557}
; CHECK-NOT: ![[#]] = !{!"function_entry_count", i64 1557, i64 -2012135647395072713}

@a = dso_local global i32 1, align 4

; Function Attrs: nounwind uwtable
define dso_local ptr @foo() #0 !dbg !11 {
entry:
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 1, i32 0, i64 -1), !dbg !14
  %0 = load volatile i32, ptr @a, align 4, !dbg !14, !tbaa !15
  %inc = add nsw i32 %0, 1, !dbg !14
  store volatile i32 %inc, ptr @a, align 4, !dbg !14, !tbaa !15
  ret ptr @bar, !dbg !19
}

declare dso_local i32 @bar(i32 noundef) #1

; Function Attrs: noinline nounwind uwtable
define dso_local void @func() #2 !dbg !20 {
entry:
  call void @llvm.pseudoprobe(i64 7289175272376759421, i64 1, i32 0, i64 -1), !dbg !21
  %call = call ptr @foo(), !dbg !22
  %call1 = call i32 %call(i32 noundef 10), !dbg !24
  %0 = load volatile i32, ptr @a, align 4, !dbg !26, !tbaa !15
  %add = add nsw i32 %0, %call1, !dbg !26
  store volatile i32 %add, ptr @a, align 4, !dbg !26, !tbaa !15
  ret void, !dbg !27
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #3

; Function Attrs: nounwind uwtable
define dso_local i32 @main() #0 !dbg !28 {
entry:
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 1, i32 0, i64 -1), !dbg !29
  br label %for.cond, !dbg !30

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ], !dbg !31
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 2, i32 0, i64 -1), !dbg !32
  %cmp = icmp slt i32 %i.0, 1000000, !dbg !34
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !36

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 3, i32 0, i64 -1), !dbg !37
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 6, i32 0, i64 -1), !dbg !38
  ret i32 0, !dbg !38

for.body:                                         ; preds = %for.cond
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 4, i32 0, i64 -1), !dbg !39
  call void @func(), !dbg !40
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 5, i32 0, i64 -1), !dbg !42
  %inc = add nsw i32 %i.0, 1, !dbg !43
  br label %for.cond, !dbg !45, !llvm.loop !46
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #4

attributes #0 = { nounwind uwtable "disable-tail-calls"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #1 = { "disable-tail-calls"="true" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { noinline nounwind uwtable "disable-tail-calls"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.ident = !{!7}
!llvm.pseudo_probe_desc = !{!8, !9, !10}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 17.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "file.c", directory: "/tmp/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{i32 1, !"EnableSplitLTOUnit", i32 0}
!7 = !{!"clang version 17.0.0"}
!8 = !{i64 6699318081062747564, i64 4294967295, !"foo"}
!9 = !{i64 7289175272376759421, i64 562954248388607, !"func"}
!10 = !{i64 -2624081020897602054, i64 281582081721716, !"main"}
!11 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 6, type: !12, scopeLine: 6, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!12 = !DISubroutineType(types: !13)
!13 = !{}
!14 = !DILocation(line: 7, column: 4, scope: !11)
!15 = !{!16, !16, i64 0}
!16 = !{!"int", !17, i64 0}
!17 = !{!"omnipotent char", !18, i64 0}
!18 = !{!"Simple C/C++ TBAA"}
!19 = !DILocation(line: 8, column: 3, scope: !11)
!20 = distinct !DISubprogram(name: "func", scope: !1, file: !1, line: 11, type: !12, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!21 = !DILocation(line: 14, column: 12, scope: !20)
!22 = !DILocation(line: 14, column: 12, scope: !23)
!23 = !DILexicalBlockFile(scope: !20, file: !1, discriminator: 186646551)
!24 = !DILocation(line: 15, column: 11, scope: !25)
!25 = !DILexicalBlockFile(scope: !20, file: !1, discriminator: 119537695)
!26 = !DILocation(line: 15, column: 7, scope: !20)
!27 = !DILocation(line: 16, column: 1, scope: !20)
!28 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 18, type: !12, scopeLine: 18, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!29 = !DILocation(line: 19, column: 11, scope: !28)
!30 = !DILocation(line: 19, column: 7, scope: !28)
!31 = !DILocation(line: 19, scope: !28)
!32 = !DILocation(line: 19, column: 18, scope: !33)
!33 = !DILexicalBlockFile(scope: !28, file: !1, discriminator: 0)
!34 = !DILocation(line: 19, column: 20, scope: !35)
!35 = !DILexicalBlockFile(scope: !28, file: !1, discriminator: 2)
!36 = !DILocation(line: 19, column: 3, scope: !35)
!37 = !DILocation(line: 0, scope: !28)
!38 = !DILocation(line: 21, column: 1, scope: !28)
!39 = !DILocation(line: 20, column: 5, scope: !28)
!40 = !DILocation(line: 20, column: 5, scope: !41)
!41 = !DILexicalBlockFile(scope: !28, file: !1, discriminator: 186646591)
!42 = !DILocation(line: 19, column: 36, scope: !33)
!43 = !DILocation(line: 19, column: 36, scope: !44)
!44 = !DILexicalBlockFile(scope: !28, file: !1, discriminator: 4)
!45 = !DILocation(line: 19, column: 3, scope: !44)
!46 = distinct !{!46, !47, !48, !49}
!47 = !DILocation(line: 19, column: 3, scope: !28)
!48 = !DILocation(line: 20, column: 10, scope: !28)
!49 = !{!"llvm.loop.mustprogress"}
