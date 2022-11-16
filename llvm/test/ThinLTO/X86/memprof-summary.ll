;; Check memprof summaries (per module, combined index, and distributed indexes)

; RUN: split-file %s %t
; RUN: opt -module-summary %t/a.ll -o %ta.bc
; RUN: opt -module-summary %t/b.ll -o %tb.bc

; RUN: llvm-dis -o - %ta.bc | FileCheck %s --check-prefix=PRELINKDISA
; PRELINKDISA: gv: (name: "main", {{.*}} callsites: ((callee: ^2, clones: (0), stackIds: (8632435727821051414)), (callee: ^2, clones: (0), stackIds: (15025054523792398438)))))) ; guid = 15822663052811949562

; RUN: llvm-dis -o - %tb.bc | FileCheck %s --check-prefix=PRELINKDISB
; PRELINKDISB: gv: (name: "_Z3foov", {{.*}} callsites: ((callee: ^2, clones: (0), stackIds: (2732490490862098848)))))) ; guid = 9191153033785521275
; PRELINKDISB: gv: (name: "_Z3bazv", {{.*}} callsites: ((callee: ^3, clones: (0), stackIds: (12481870273128938184)))))) ; guid = 15176620447596392000
; PRELINKDISB: gv: (name: "_Z3barv", {{.*}} allocs: ((versions: (none), memProf: ((type: notcold, stackIds: (12481870273128938184, 2732490490862098848, 8632435727821051414)), (type: cold, stackIds: (12481870273128938184, 2732490490862098848, 15025054523792398438)))))))) ; guid = 17377440600225628772

; RUN: llvm-bcanalyzer -dump %ta.bc | FileCheck %s --check-prefix=PRELINKBCANA
; PRELINKBCANA: <STACK_IDS abbrevid=4 op0=8632435727821051414 op1=-3421689549917153178/>

; RUN: llvm-bcanalyzer -dump %tb.bc | FileCheck %s --check-prefix=PRELINKBCANB
; PRELINKBCANB: <STACK_IDS abbrevid=4 op0=-5964873800580613432 op1=2732490490862098848 op2=8632435727821051414 op3=-3421689549917153178/>

; RUN: llvm-lto2 run %ta.bc %tb.bc -o %t -save-temps \
; RUN:     -thinlto-distributed-indexes \
; RUN:     -r=%ta.bc,main,plx \
; RUN:     -r=%ta.bc,_Z3foov, \
; RUN:     -r=%ta.bc,free, \
; RUN:     -r=%ta.bc,sleep, \
; RUN:     -r=%tb.bc,_Z3foov,pl \
; RUN:     -r=%tb.bc,_Znam, \
; RUN:     -r=%tb.bc,_Z3barv,pl \
; RUN:     -r=%tb.bc,_Z3bazv,pl

; RUN: llvm-dis -o - %t.index.bc | FileCheck %s --check-prefix=COMBINEDDIS
; COMBINEDDIS: gv: (guid: 9191153033785521275, {{.*}} callsites: ((callee: ^3, clones: (0), stackIds: (2732490490862098848))))))
; COMBINEDDIS: gv: (guid: 15176620447596392000, {{.*}} callsites: ((callee: ^5, clones: (0), stackIds: (12481870273128938184))))))
; COMBINEDDIS: gv: (guid: 15822663052811949562, {{.*}} callsites: ((callee: ^2, clones: (0), stackIds: (8632435727821051414)), (callee: ^2, clones: (0), stackIds: (15025054523792398438))))))
; COMBINEDDIS: gv: (guid: 17377440600225628772, {{.*}} allocs: ((versions: (none), memProf: ((type: notcold, stackIds: (12481870273128938184, 2732490490862098848, 8632435727821051414)), (type: cold, stackIds: (12481870273128938184, 2732490490862098848, 15025054523792398438))))))))

; RUN: llvm-bcanalyzer -dump %t.index.bc | FileCheck %s --check-prefix=COMBINEDBCAN
; COMBINEDBCAN: <STACK_IDS abbrevid=4 op0=8632435727821051414 op1=-3421689549917153178 op2=-5964873800580613432 op3=2732490490862098848/>

; RUN: llvm-dis -o - %ta.bc.thinlto.bc | FileCheck %s --check-prefix=DISTRIBUTEDDISA
; DISTRIBUTEDDISA: gv: (guid: 9191153033785521275, {{.*}} callsites: ((callee: null, clones: (0), stackIds: (2732490490862098848))))))
; DISTRIBUTEDDISA: gv: (guid: 15822663052811949562, {{.*}} callsites: ((callee: ^2, clones: (0), stackIds: (8632435727821051414)), (callee: ^2, clones: (0), stackIds: (15025054523792398438))))))

; RUN: llvm-dis -o - %tb.bc.thinlto.bc | FileCheck %s --check-prefix=DISTRIBUTEDDISB
; DISTRIBUTEDDISB: gv: (guid: 9191153033785521275, {{.*}} callsites: ((callee: ^2, clones: (0), stackIds: (2732490490862098848))))))
; DISTRIBUTEDDISB: gv: (guid: 15176620447596392000, {{.*}} callsites: ((callee: ^3, clones: (0), stackIds: (12481870273128938184))))))
; DISTRIBUTEDDISB: gv: (guid: 17377440600225628772, {{.*}} allocs: ((versions: (none), memProf: ((type: notcold, stackIds: (12481870273128938184, 2732490490862098848, 8632435727821051414)), (type: cold, stackIds: (12481870273128938184, 2732490490862098848, 15025054523792398438))))))))

; RUN: llvm-bcanalyzer -dump %ta.bc.thinlto.bc | FileCheck %s --check-prefix=DISTRIBUTEDBCANA
; DISTRIBUTEDBCANA: <STACK_IDS abbrevid=4 op0=8632435727821051414 op1=-3421689549917153178 op2=2732490490862098848/>

; RUN: llvm-bcanalyzer -dump %tb.bc.thinlto.bc | FileCheck %s --check-prefix=DISTRIBUTEDBCANB
; DISTRIBUTEDBCANB: <STACK_IDS abbrevid=4 op0=8632435727821051414 op1=-3421689549917153178 op2=-5964873800580613432 op3=2732490490862098848/>

;--- a.ll
; ModuleID = 'a.cc'
source_filename = "a.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %argc, ptr nocapture noundef readnone %argv) local_unnamed_addr #0 !dbg !39 {
entry:
  %call = call noundef ptr @_Z3foov(), !dbg !42, !callsite !43
  %call1 = call noundef ptr @_Z3foov(), !dbg !44, !callsite !45
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(10) %call, i8 0, i64 10, i1 false), !dbg !46
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(10) %call1, i8 0, i64 10, i1 false), !dbg !47
  call void @free(ptr noundef %call) #4, !dbg !48
  %call2 = call i32 @sleep(i32 noundef 10), !dbg !49
  call void @free(ptr noundef %call1) #4, !dbg !50
  ret i32 0, !dbg !51
}

declare !dbg !52 noundef ptr @_Z3foov() local_unnamed_addr #1

; Function Attrs: argmemonly mustprogress nocallback nofree nounwind willreturn writeonly
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #2

; Function Attrs: inaccessiblemem_or_argmemonly mustprogress nounwind willreturn allockind("free")
declare void @free(ptr allocptr nocapture noundef) local_unnamed_addr #3

declare !dbg !53 i32 @sleep(i32 noundef) local_unnamed_addr #1

attributes #0 = { mustprogress norecurse uwtable "disable-tail-calls"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "disable-tail-calls"="true" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { argmemonly mustprogress nocallback nofree nounwind willreturn writeonly }
attributes #3 = { inaccessiblemem_or_argmemonly mustprogress nounwind willreturn allockind("free") "alloc-family"="malloc" "disable-tail-calls"="true" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 16.0.0 (git@github.com:llvm/llvm-project.git ffecb643ee2c49e55e0689339b6d5921b5e6ff8b)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "a.cc", directory: ".", checksumkind: CSK_MD5, checksum: "ebabd56909271a1d4a7cac81c10624d5")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!39 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 5, type: !40, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !41)
!40 = !DISubroutineType(types: !41)
!41 = !{}
!42 = !DILocation(line: 6, column: 13, scope: !39)
!43 = !{i64 8632435727821051414}
!44 = !DILocation(line: 7, column: 13, scope: !39)
!45 = !{i64 -3421689549917153178}
!46 = !DILocation(line: 8, column: 3, scope: !39)
!47 = !DILocation(line: 9, column: 3, scope: !39)
!48 = !DILocation(line: 10, column: 3, scope: !39)
!49 = !DILocation(line: 11, column: 3, scope: !39)
!50 = !DILocation(line: 12, column: 3, scope: !39)
!51 = !DILocation(line: 13, column: 3, scope: !39)
!52 = !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 4, type: !40, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !41)
!53 = !DISubprogram(name: "sleep", scope: !54, file: !54, line: 453, type: !40, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !41)
!54 = !DIFile(filename: "include/unistd.h", directory: "/usr", checksumkind: CSK_MD5, checksum: "ee8f41a17f563f029d0e930ad871815a")

;--- b.ll
; ModuleID = 'b.cc'
source_filename = "b.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress noinline uwtable
define dso_local noalias noundef nonnull ptr @_Z3barv() local_unnamed_addr #0 !dbg !39 {
entry:
  %call = call noalias noundef nonnull dereferenceable(10) ptr @_Znam(i64 noundef 10) #2, !dbg !42, !memprof !43, !callsite !48
  ret ptr %call, !dbg !49
}

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znam(i64 noundef) local_unnamed_addr #1

; Function Attrs: mustprogress noinline uwtable
define dso_local noalias noundef nonnull ptr @_Z3bazv() local_unnamed_addr #0 !dbg !50 {
entry:
  %call = call noundef ptr @_Z3barv(), !dbg !51, !callsite !52
  ret ptr %call, !dbg !53
}

; Function Attrs: mustprogress uwtable
define dso_local noalias noundef nonnull ptr @_Z3foov() local_unnamed_addr #3 !dbg !54 {
entry:
  %call = call noundef ptr @_Z3bazv(), !dbg !55, !callsite !56
  ret ptr %call, !dbg !57
}

attributes #0 = { mustprogress noinline uwtable "disable-tail-calls"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nobuiltin allocsize(0) "disable-tail-calls"="true" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { builtin allocsize(0) }
attributes #3 = { mustprogress uwtable "disable-tail-calls"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 16.0.0 (git@github.com:llvm/llvm-project.git ffecb643ee2c49e55e0689339b6d5921b5e6ff8b)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "b.cc", directory: ".", checksumkind: CSK_MD5, checksum: "335f81d275af57725cfc9ffc7be49bc2")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!39 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 1, type: !40, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !41)
!40 = !DISubroutineType(types: !41)
!41 = !{}
!42 = !DILocation(line: 2, column: 10, scope: !39)
!43 = !{!44, !46}
!44 = !{!45, !"notcold"}
!45 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 8632435727821051414}
!46 = !{!47, !"cold"}
!47 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 -3421689549917153178}
!48 = !{i64 9086428284934609951}
!49 = !DILocation(line: 2, column: 3, scope: !39)
!50 = distinct !DISubprogram(name: "baz", linkageName: "_Z3bazv", scope: !1, file: !1, line: 5, type: !40, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !41)
!51 = !DILocation(line: 6, column: 10, scope: !50)
!52 = !{i64 -5964873800580613432}
!53 = !DILocation(line: 6, column: 3, scope: !50)
!54 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 9, type: !40, scopeLine: 9, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !41)
!55 = !DILocation(line: 10, column: 10, scope: !54)
!56 = !{i64 2732490490862098848}
!57 = !DILocation(line: 10, column: 3, scope: !54)
