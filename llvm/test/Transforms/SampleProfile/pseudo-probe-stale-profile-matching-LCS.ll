; REQUIRES: x86_64-linux
; REQUIRES: asserts
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/pseudo-probe-stale-profile-matching-LCS.prof --salvage-stale-profile -S --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl 2>&1 | FileCheck %s
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/pseudo-probe-stale-profile-matching-LCS.prof --salvage-stale-profile -S --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl --salvage-stale-profile-max-callsites=6 2>&1 | FileCheck %s -check-prefix=CHECK-MAX-CALLSITES

; CHECK: Run stale profile matching for test_indirect_call
; CHECK: Location is matched from 1 to 1
; CHECK: Location is matched from 2 to 2
; CHECK: Location is matched from 3 to 3
; CHECK: Location is matched from 4 to 4
; CHECK: Callsite with callee:C is matched from 5 to 2
; CHECK: Location is rematched backwards from 3 to 0
; CHECK: Location is rematched backwards from 4 to 1
; CHECK: Callsite with callee:unknown.indirect.callee is matched from 6 to 3
; CHECK:Callsite with callee:B is matched from 7 to 4
; CHECK: Location is matched from 8 to 5
; CHECK: Callsite with callee:unknown.indirect.callee is matched from 9 to 6
; CHECK: Callsite with callee:C is matched from 10 to 7

; CHECK: Run stale profile matching for test_direct_call
; CHECK: Location is matched from 1 to 1
; CHECK: Location is matched from 2 to 2
; CHECK: Location is matched from 3 to 3
; CHECK: Callsite with callee:C is matched from 4 to 2
; CHECK: Location is rematched backwards from 3 to 1
; CHECK: Callsite with callee:A is matched from 5 to 4
; CHECK: Callsite with callee:B is matched from 6 to 5
; CHECK: Location is matched from 7 to 6
; CHECK: Callsite with callee:A is matched from 8 to 6

; CHECK-MAX-CALLSITES: Skip stale profile matching for test_direct_call
; CHECK-MAX-CALLSITES-NOT: Skip stale profile matching for test_indirect_call

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@c = external global i32, align 4

; Function Attrs: nounwind uwtable
define dso_local i32 @test_direct_call(i32 noundef %x) #0 !dbg !12 {
entry:
    #dbg_value(i32 %x, !17, !DIExpression(), !18)
  call void @llvm.pseudoprobe(i64 -4364451034228175269, i64 1, i32 0, i64 -1), !dbg !19
  %call = call i32 @A(i32 noundef %x), !dbg !20
  %add = add nsw i32 %x, %call, !dbg !22
    #dbg_value(i32 %add, !17, !DIExpression(), !18)
  %call1 = call i32 @B(i32 noundef %add), !dbg !23
  %add2 = add nsw i32 %add, %call1, !dbg !25
    #dbg_value(i32 %add2, !17, !DIExpression(), !18)
  %call3 = call i32 @C(i32 noundef %add2), !dbg !26
  %add4 = add nsw i32 %add2, %call3, !dbg !28
    #dbg_value(i32 %add4, !17, !DIExpression(), !18)
  %call5 = call i32 @A(i32 noundef %add4), !dbg !29
  %add6 = add nsw i32 %add4, %call5, !dbg !31
    #dbg_value(i32 %add6, !17, !DIExpression(), !18)
  %call7 = call i32 @B(i32 noundef %add6), !dbg !32
  %add8 = add nsw i32 %add6, %call7, !dbg !34
    #dbg_value(i32 %add8, !17, !DIExpression(), !18)
  %call9 = call i32 @B(i32 noundef %add8), !dbg !35
  %add10 = add nsw i32 %add8, %call9, !dbg !37
    #dbg_value(i32 %add10, !17, !DIExpression(), !18)
  %call11 = call i32 @A(i32 noundef %add10), !dbg !38
  %add12 = add nsw i32 %add10, %call11, !dbg !40
    #dbg_value(i32 %add12, !17, !DIExpression(), !18)
  ret i32 %add12, !dbg !41
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare !dbg !42 i32 @A(i32 noundef) #2

declare !dbg !43 i32 @B(i32 noundef) #2

declare !dbg !44 i32 @C(i32 noundef) #2

; Function Attrs: nounwind uwtable
define dso_local i32 @test_indirect_call(i32 noundef %x) #0 !dbg !45 {
entry:
    #dbg_value(i32 %x, !47, !DIExpression(), !50)
  call void @llvm.pseudoprobe(i64 -8563147518712133441, i64 1, i32 0, i64 -1), !dbg !51
  %0 = load i32, ptr @c, align 4, !dbg !51, !tbaa !53
  %tobool = icmp ne i32 %0, 0, !dbg !51
  br i1 %tobool, label %if.then, label %if.else, !dbg !57

if.then:                                          ; preds = %entry
  call void @llvm.pseudoprobe(i64 -8563147518712133441, i64 2, i32 0, i64 -1), !dbg !58
    #dbg_value(ptr @A, !48, !DIExpression(), !50)
  br label %if.end, !dbg !59

if.else:                                          ; preds = %entry
  call void @llvm.pseudoprobe(i64 -8563147518712133441, i64 3, i32 0, i64 -1), !dbg !60
    #dbg_value(ptr @B, !48, !DIExpression(), !50)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %fp.0 = phi ptr [ @A, %if.then ], [ @B, %if.else ], !dbg !61
    #dbg_value(ptr %fp.0, !48, !DIExpression(), !50)
  call void @llvm.pseudoprobe(i64 -8563147518712133441, i64 4, i32 0, i64 -1), !dbg !62
  %call = call i32 @C(i32 noundef %x), !dbg !63
  %add = add nsw i32 %x, %call, !dbg !65
    #dbg_value(i32 %add, !47, !DIExpression(), !50)
  %call1 = call i32 %fp.0(i32 noundef %add), !dbg !66
  %add2 = add nsw i32 %add, %call1, !dbg !68
    #dbg_value(i32 %add2, !47, !DIExpression(), !50)
  %call3 = call i32 @B(i32 noundef %add2), !dbg !69
  %add4 = add nsw i32 %add2, %call3, !dbg !71
    #dbg_value(i32 %add4, !47, !DIExpression(), !50)
  %call5 = call i32 @C(i32 noundef %add4), !dbg !72
  %add6 = add nsw i32 %add4, %call5, !dbg !74
    #dbg_value(i32 %add6, !47, !DIExpression(), !50)
  %call7 = call i32 %fp.0(i32 noundef %add6), !dbg !75
  %add8 = add nsw i32 %add6, %call7, !dbg !77
    #dbg_value(i32 %add8, !47, !DIExpression(), !50)
  %call9 = call i32 @C(i32 noundef %add8), !dbg !78
  %add10 = add nsw i32 %add8, %call9, !dbg !80
    #dbg_value(i32 %add10, !47, !DIExpression(), !50)
  ret i32 %add10, !dbg !81
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #4

attributes #0 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn }
attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}
!llvm.pseudo_probe_desc = !{!10, !11}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 19.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/home/", checksumkind: CSK_MD5, checksum: "be98aa946f37f0ad8d307c9121efe101")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 19.0.0"}
!10 = !{i64 -4364451034228175269, i64 1970329131941887, !"test_direct_call"}
!11 = !{i64 -8563147518712133441, i64 1688922477484692, !"test_indirect_call"}
!12 = distinct !DISubprogram(name: "test_direct_call", scope: !1, file: !1, line: 10, type: !13, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !16)
!13 = !DISubroutineType(types: !14)
!14 = !{!15, !15}
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !{!17}
!17 = !DILocalVariable(name: "x", arg: 1, scope: !12, file: !1, line: 10, type: !15)
!18 = !DILocation(line: 0, scope: !12)
!19 = !DILocation(line: 11, column: 10, scope: !12)
!20 = !DILocation(line: 11, column: 8, scope: !21)
!21 = !DILexicalBlockFile(scope: !12, file: !1, discriminator: 186646551)
!22 = !DILocation(line: 11, column: 5, scope: !12)
!23 = !DILocation(line: 12, column: 8, scope: !24)
!24 = !DILexicalBlockFile(scope: !12, file: !1, discriminator: 186646559)
!25 = !DILocation(line: 12, column: 5, scope: !12)
!26 = !DILocation(line: 13, column: 8, scope: !27)
!27 = !DILexicalBlockFile(scope: !12, file: !1, discriminator: 186646567)
!28 = !DILocation(line: 13, column: 5, scope: !12)
!29 = !DILocation(line: 14, column: 8, scope: !30)
!30 = !DILexicalBlockFile(scope: !12, file: !1, discriminator: 186646575)
!31 = !DILocation(line: 14, column: 5, scope: !12)
!32 = !DILocation(line: 15, column: 8, scope: !33)
!33 = !DILexicalBlockFile(scope: !12, file: !1, discriminator: 186646583)
!34 = !DILocation(line: 15, column: 5, scope: !12)
!35 = !DILocation(line: 16, column: 8, scope: !36)
!36 = !DILexicalBlockFile(scope: !12, file: !1, discriminator: 186646591)
!37 = !DILocation(line: 16, column: 5, scope: !12)
!38 = !DILocation(line: 17, column: 8, scope: !39)
!39 = !DILexicalBlockFile(scope: !12, file: !1, discriminator: 186646599)
!40 = !DILocation(line: 17, column: 5, scope: !12)
!41 = !DILocation(line: 18, column: 3, scope: !12)
!42 = !DISubprogram(name: "A", scope: !1, file: !1, line: 2, type: !13, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!43 = !DISubprogram(name: "B", scope: !1, file: !1, line: 3, type: !13, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!44 = !DISubprogram(name: "C", scope: !1, file: !1, line: 4, type: !13, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!45 = distinct !DISubprogram(name: "test_indirect_call", scope: !1, file: !1, line: 21, type: !13, scopeLine: 21, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !46)
!46 = !{!47, !48}
!47 = !DILocalVariable(name: "x", arg: 1, scope: !45, file: !1, line: 21, type: !15)
!48 = !DILocalVariable(name: "fp", scope: !45, file: !1, line: 22, type: !49)
!49 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!50 = !DILocation(line: 0, scope: !45)
!51 = !DILocation(line: 23, column: 6, scope: !52)
!52 = distinct !DILexicalBlock(scope: !45, file: !1, line: 23, column: 6)
!53 = !{!54, !54, i64 0}
!54 = !{!"int", !55, i64 0}
!55 = !{!"omnipotent char", !56, i64 0}
!56 = !{!"Simple C/C++ TBAA"}
!57 = !DILocation(line: 23, column: 6, scope: !45)
!58 = !DILocation(line: 24, column: 8, scope: !52)
!59 = !DILocation(line: 24, column: 5, scope: !52)
!60 = !DILocation(line: 26, column: 8, scope: !52)
!61 = !DILocation(line: 0, scope: !52)
!62 = !DILocation(line: 27, column: 10, scope: !45)
!63 = !DILocation(line: 27, column: 8, scope: !64)
!64 = !DILexicalBlockFile(scope: !45, file: !1, discriminator: 186646575)
!65 = !DILocation(line: 27, column: 5, scope: !45)
!66 = !DILocation(line: 28, column: 8, scope: !67)
!67 = !DILexicalBlockFile(scope: !45, file: !1, discriminator: 119537719)
!68 = !DILocation(line: 28, column: 5, scope: !45)
!69 = !DILocation(line: 29, column: 8, scope: !70)
!70 = !DILexicalBlockFile(scope: !45, file: !1, discriminator: 186646591)
!71 = !DILocation(line: 29, column: 5, scope: !45)
!72 = !DILocation(line: 30, column: 8, scope: !73)
!73 = !DILexicalBlockFile(scope: !45, file: !1, discriminator: 186646599)
!74 = !DILocation(line: 30, column: 5, scope: !45)
!75 = !DILocation(line: 31, column: 8, scope: !76)
!76 = !DILexicalBlockFile(scope: !45, file: !1, discriminator: 119537743)
!77 = !DILocation(line: 31, column: 5, scope: !45)
!78 = !DILocation(line: 32, column: 8, scope: !79)
!79 = !DILexicalBlockFile(scope: !45, file: !1, discriminator: 186646615)
!80 = !DILocation(line: 32, column: 5, scope: !45)
!81 = !DILocation(line: 33, column: 3, scope: !45)
