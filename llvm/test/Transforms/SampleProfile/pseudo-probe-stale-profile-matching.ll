; REQUIRES: x86_64-linux
; REQUIRES: asserts
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/pseudo-probe-stale-profile-matching.prof --salvage-stale-profile -S --debug-only=sample-profile,sample-profile-impl 2>&1 | FileCheck %s

; The profiled source code:

;  volatile int x = 1;
;  __attribute__((noinline)) int bar(int p) {
;    return p;
;  }

;  __attribute__((always_inline)) int foo(int i, int p) {
;    if (i % 10) return  bar(p);
;    else return bar(p + 1);
;  }

;  int main() {
;    for (int i = 0; i < 1000 * 1000; i++) {
;       x += foo(i, x);
;       x += bar(x);
;       x += foo(i, x);
;       x += bar(x);
;    }
;  }

; The source code for the current build:

;  volatile int x = 1;
;  __attribute__((noinline)) int bar(int p) {
;    return p;
;  }

;  __attribute__((always_inline)) int foo(int i, int p) {
;    if (i % 10) return  bar(p);
;    else return bar(p + 1);
;  }

;  int main() {
;    if (x == 0)          // code change
;      return 0;          // code change
;    for (int i = 0; i < 1000 * 1000; i++) {
;       x += foo(i, x);
;       x += bar(x);
;       if (i < 0)        // code change
;         return 0;       // code change
;       x += foo(i, x);
;       x += bar(x);
;    }
;  }


; CHECK: Run stale profile matching for main

; CHECK: Location is matched from 1 to 1
; CHECK: Location is matched from 2 to 2
; CHECK: Location is matched from 3 to 3
; CHECK: Location is matched from 4 to 4
; CHECK: Location is matched from 5 to 5
; CHECK: Location is matched from 6 to 6
; CHECK: Location is matched from 7 to 7
; CHECK: Location is matched from 8 to 8
; CHECK: Location is matched from 9 to 9
; CHECK: Location is matched from 10 to 10
; CHECK: Location is matched from 11 to 11

; CHECK: Callsite with callee:foo is matched from 13 to 6
; CHECK: Location is rematched backwards from 7 to 0
; CHECK: Location is rematched backwards from 8 to 1
; CHECK: Location is rematched backwards from 9 to 2
; CHECK: Location is rematched backwards from 10 to 3
; CHECK: Location is rematched backwards from 11 to 4
; CHECK: Callsite with callee:bar is matched from 14 to 7
; CHECK: Callsite with callee:foo is matched from 15 to 8
; CHECK: Callsite with callee:bar is matched from 16 to 9


; CHECK:    2:  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 2, i32 0, i64 -1), !dbg ![[#]] - weight: 112 - factor: 1.00)
; CHECK:    3:  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 3, i32 0, i64 -1), !dbg ![[#]] - weight: 112 - factor: 1.00)
; CHECK:    4:  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 4, i32 0, i64 -1), !dbg ![[#]] - weight: 116 - factor: 1.00)
; CHECK:    5:  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 5, i32 0, i64 -1), !dbg ![[#]] - weight: 0 - factor: 1.00)
; CHECK:    1:  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 1, i32 0, i64 -1), !dbg ![[#]] - weight: 112 - factor: 1.00)
; CHECK:    2:  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 2, i32 0, i64 -1), !dbg ![[#]] - weight: 101 - factor: 1.00)
; CHECK:    5:  %call.i8 = call i32 @bar(i32 noundef %1), !dbg ![[#]] - weight: 101 - factor: 1.00)
; CHECK:    3:  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 3, i32 0, i64 -1), !dbg ![[#]] - weight: 13 - factor: 1.00)
; CHECK:    6:  %call1.i5 = call i32 @bar(i32 noundef %add.i4), !dbg ![[#]] - weight: 13 - factor: 1.00)
; CHECK:    4:  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 4, i32 0, i64 -1), !dbg ![[#]] - weight: 112 - factor: 1.00)
; CHECK:    14:  %call2 = call i32 @bar(i32 noundef %3), !dbg ![[#]] - weight: 124 - factor: 1.00)
; CHECK:    8:  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 8, i32 0, i64 -1), !dbg ![[#]] - weight: 0 - factor: 1.00)
; CHECK:    1:  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 1, i32 0, i64 -1), !dbg ![[#]] - weight: 117 - factor: 1.00)
; CHECK:    2:  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 2, i32 0, i64 -1), !dbg ![[#]] - weight: 104 - factor: 1.00)
; CHECK:    5:  %call.i = call i32 @bar(i32 noundef %5), !dbg ![[#]] - weight: 104 - factor: 1.00)
; CHECK:    3:  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 3, i32 0, i64 -1), !dbg ![[#]] - weight: 13 - factor: 1.00)
; CHECK:    6:  %call1.i = call i32 @bar(i32 noundef %add.i), !dbg ![[#]] - weight: 14 - factor: 1.00)
; CHECK:    4:  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 4, i32 0, i64 -1), !dbg ![[#]] - weight: 121 - factor: 1.00)
; CHECK:    16:  %call9 = call i32 @bar(i32 noundef %7), !dbg ![[#]] - weight: 126 - factor: 1.00)
; CHECK:    9:  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 9, i32 0, i64 -1), !dbg ![[#]] - weight: 112 - factor: 1.00)
; CHECK:    10:  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 10, i32 0, i64 -1), !dbg ![[#]] - weight: 112 - factor: 1.00)
; CHECK:    11:  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 11, i32 0, i64 -1), !dbg ![[#]] - weight: 116 - factor: 1.00)
; CHECK:    1:  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 1, i32 0, i64 -1), !dbg ![[#]] - weight: 0 - factor: 1.00)


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = dso_local global i32 1, align 4, !dbg !0

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @bar(i32 noundef %p) #0 !dbg !16 {
entry:
  call void @llvm.dbg.value(metadata i32 %p, metadata !20, metadata !DIExpression()), !dbg !21
  call void @llvm.pseudoprobe(i64 -2012135647395072713, i64 1, i32 0, i64 -1), !dbg !22
  ret i32 %p, !dbg !23
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: alwaysinline nounwind uwtable
define dso_local i32 @foo(i32 noundef %i, i32 noundef %p) #2 !dbg !24 {
entry:
  call void @llvm.dbg.value(metadata i32 %i, metadata !28, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 %p, metadata !29, metadata !DIExpression()), !dbg !30
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 1, i32 0, i64 -1), !dbg !31
  %rem = srem i32 %i, 10, !dbg !33
  %tobool = icmp ne i32 %rem, 0, !dbg !33
  br i1 %tobool, label %if.then, label %if.else, !dbg !34

if.then:                                          ; preds = %entry
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 2, i32 0, i64 -1), !dbg !35
  %call = call i32 @bar(i32 noundef %p), !dbg !36
  br label %return, !dbg !38

if.else:                                          ; preds = %entry
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 3, i32 0, i64 -1), !dbg !39
  %add = add nsw i32 %p, 1, !dbg !40
  %call1 = call i32 @bar(i32 noundef %add), !dbg !41
  br label %return, !dbg !43

return:                                           ; preds = %if.else, %if.then
  %retval.0 = phi i32 [ %call, %if.then ], [ %call1, %if.else ], !dbg !44
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 4, i32 0, i64 -1), !dbg !45
  ret i32 %retval.0, !dbg !45
}

; Function Attrs: nounwind uwtable
define dso_local i32 @main() #3 !dbg !46 {
entry:
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 1, i32 0, i64 -1), !dbg !52
  %0 = load volatile i32, ptr @x, align 4, !dbg !52, !tbaa !54
  %cmp = icmp eq i32 %0, 0, !dbg !58
  br i1 %cmp, label %if.then, label %if.end, !dbg !59

if.then:                                          ; preds = %entry
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 2, i32 0, i64 -1), !dbg !60
  br label %for.end, !dbg !60

if.end:                                           ; preds = %entry
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 3, i32 0, i64 -1), !dbg !61
  call void @llvm.dbg.value(metadata i32 0, metadata !50, metadata !DIExpression()), !dbg !62
  br label %for.cond, !dbg !63

for.cond:                                         ; preds = %if.end6, %if.end
  %i.0 = phi i32 [ 0, %if.end ], [ %inc, %if.end6 ], !dbg !64
  call void @llvm.dbg.value(metadata i32 %i.0, metadata !50, metadata !DIExpression()), !dbg !62
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 4, i32 0, i64 -1), !dbg !65
  %cmp1 = icmp slt i32 %i.0, 1000000, !dbg !67
  br i1 %cmp1, label %for.body, label %for.cond.cleanup, !dbg !68

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 5, i32 0, i64 -1), !dbg !68
  br label %cleanup, !dbg !68

for.body:                                         ; preds = %for.cond
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 6, i32 0, i64 -1), !dbg !69
  %1 = load volatile i32, ptr @x, align 4, !dbg !71, !tbaa !54
  %call = call i32 @foo(i32 noundef %i.0, i32 noundef %1), !dbg !72
  %2 = load volatile i32, ptr @x, align 4, !dbg !74, !tbaa !54
  %add = add nsw i32 %2, %call, !dbg !74
  store volatile i32 %add, ptr @x, align 4, !dbg !74, !tbaa !54
  %3 = load volatile i32, ptr @x, align 4, !dbg !75, !tbaa !54
  %call2 = call i32 @bar(i32 noundef %3), !dbg !76
  %4 = load volatile i32, ptr @x, align 4, !dbg !78, !tbaa !54
  %add3 = add nsw i32 %4, %call2, !dbg !78
  store volatile i32 %add3, ptr @x, align 4, !dbg !78, !tbaa !54
  br i1 false, label %if.then5, label %if.end6, !dbg !79

if.then5:                                         ; preds = %for.body
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 7, i32 0, i64 -1), !dbg !80
  br label %cleanup, !dbg !80

if.end6:                                          ; preds = %for.body
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 8, i32 0, i64 -1), !dbg !82
  %5 = load volatile i32, ptr @x, align 4, !dbg !83, !tbaa !54
  %call7 = call i32 @foo(i32 noundef %i.0, i32 noundef %5), !dbg !84
  %6 = load volatile i32, ptr @x, align 4, !dbg !86, !tbaa !54
  %add8 = add nsw i32 %6, %call7, !dbg !86
  store volatile i32 %add8, ptr @x, align 4, !dbg !86, !tbaa !54
  %7 = load volatile i32, ptr @x, align 4, !dbg !87, !tbaa !54
  %call9 = call i32 @bar(i32 noundef %7), !dbg !88
  %8 = load volatile i32, ptr @x, align 4, !dbg !90, !tbaa !54
  %add10 = add nsw i32 %8, %call9, !dbg !90
  store volatile i32 %add10, ptr @x, align 4, !dbg !90, !tbaa !54
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 9, i32 0, i64 -1), !dbg !91
  %inc = add nsw i32 %i.0, 1, !dbg !91
  call void @llvm.dbg.value(metadata i32 %inc, metadata !50, metadata !DIExpression()), !dbg !62
  br label %for.cond, !dbg !92, !llvm.loop !93

cleanup:                                          ; preds = %if.then5, %for.cond.cleanup
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 10, i32 0, i64 -1), !dbg !96
  br label %for.end

for.end:                                          ; preds = %cleanup, %if.then
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 11, i32 0, i64 -1), !dbg !97
  ret i32 0, !dbg !97
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #5

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #6

attributes #0 = { noinline nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { alwaysinline nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #3 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #6 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10, !11}
!llvm.ident = !{!12}
!llvm.pseudo_probe_desc = !{!13, !14, !15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 17.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "path")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !6)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 5}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 7, !"uwtable", i32 2}
!11 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!12 = !{!"clang version 17.0.0"}
!13 = !{i64 -2012135647395072713, i64 4294967295, !"bar"}
!14 = !{i64 6699318081062747564, i64 563022570642068, !"foo"}
!15 = !{i64 -2624081020897602054, i64 1126158552146340, !"main"}
!16 = distinct !DISubprogram(name: "bar", scope: !3, file: !3, line: 2, type: !17, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !19)
!17 = !DISubroutineType(types: !18)
!18 = !{!6, !6}
!19 = !{!20}
!20 = !DILocalVariable(name: "p", arg: 1, scope: !16, file: !3, line: 2, type: !6)
!21 = !DILocation(line: 0, scope: !16)
!22 = !DILocation(line: 3, column: 10, scope: !16)
!23 = !DILocation(line: 3, column: 3, scope: !16)
!24 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 6, type: !25, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !27)
!25 = !DISubroutineType(types: !26)
!26 = !{!6, !6, !6}
!27 = !{!28, !29}
!28 = !DILocalVariable(name: "i", arg: 1, scope: !24, file: !3, line: 6, type: !6)
!29 = !DILocalVariable(name: "p", arg: 2, scope: !24, file: !3, line: 6, type: !6)
!30 = !DILocation(line: 0, scope: !24)
!31 = !DILocation(line: 7, column: 6, scope: !32)
!32 = distinct !DILexicalBlock(scope: !24, file: !3, line: 7, column: 6)
!33 = !DILocation(line: 7, column: 8, scope: !32)
!34 = !DILocation(line: 7, column: 6, scope: !24)
!35 = !DILocation(line: 7, column: 26, scope: !32)
!36 = !DILocation(line: 7, column: 22, scope: !37)
!37 = !DILexicalBlockFile(scope: !32, file: !3, discriminator: 186646575)
!38 = !DILocation(line: 7, column: 14, scope: !32)
!39 = !DILocation(line: 8, column: 19, scope: !32)
!40 = !DILocation(line: 8, column: 21, scope: !32)
!41 = !DILocation(line: 8, column: 15, scope: !42)
!42 = !DILexicalBlockFile(scope: !32, file: !3, discriminator: 186646583)
!43 = !DILocation(line: 8, column: 8, scope: !32)
!44 = !DILocation(line: 0, scope: !32)
!45 = !DILocation(line: 9, column: 1, scope: !24)
!46 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 11, type: !47, scopeLine: 11, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !49)
!47 = !DISubroutineType(types: !48)
!48 = !{!6}
!49 = !{!50}
!50 = !DILocalVariable(name: "i", scope: !51, file: !3, line: 14, type: !6)
!51 = distinct !DILexicalBlock(scope: !46, file: !3, line: 14, column: 3)
!52 = !DILocation(line: 12, column: 6, scope: !53)
!53 = distinct !DILexicalBlock(scope: !46, file: !3, line: 12, column: 6)
!54 = !{!55, !55, i64 0}
!55 = !{!"int", !56, i64 0}
!56 = !{!"omnipotent char", !57, i64 0}
!57 = !{!"Simple C/C++ TBAA"}
!58 = !DILocation(line: 12, column: 8, scope: !53)
!59 = !DILocation(line: 12, column: 6, scope: !46)
!60 = !DILocation(line: 13, column: 5, scope: !53)
!61 = !DILocation(line: 14, column: 11, scope: !51)
!62 = !DILocation(line: 0, scope: !51)
!63 = !DILocation(line: 14, column: 7, scope: !51)
!64 = !DILocation(line: 14, scope: !51)
!65 = !DILocation(line: 14, column: 18, scope: !66)
!66 = distinct !DILexicalBlock(scope: !51, file: !3, line: 14, column: 3)
!67 = !DILocation(line: 14, column: 20, scope: !66)
!68 = !DILocation(line: 14, column: 3, scope: !51)
!69 = !DILocation(line: 15, column: 15, scope: !70)
!70 = distinct !DILexicalBlock(scope: !66, file: !3, line: 14, column: 40)
!71 = !DILocation(line: 15, column: 18, scope: !70)
!72 = !DILocation(line: 15, column: 11, scope: !73)
!73 = !DILexicalBlockFile(scope: !70, file: !3, discriminator: 186646639)
!74 = !DILocation(line: 15, column: 8, scope: !70)
!75 = !DILocation(line: 16, column: 15, scope: !70)
!76 = !DILocation(line: 16, column: 11, scope: !77)
!77 = !DILexicalBlockFile(scope: !70, file: !3, discriminator: 186646647)
!78 = !DILocation(line: 16, column: 8, scope: !70)
!79 = !DILocation(line: 17, column: 9, scope: !70)
!80 = !DILocation(line: 18, column: 8, scope: !81)
!81 = distinct !DILexicalBlock(scope: !70, file: !3, line: 17, column: 9)
!82 = !DILocation(line: 19, column: 15, scope: !70)
!83 = !DILocation(line: 19, column: 18, scope: !70)
!84 = !DILocation(line: 19, column: 11, scope: !85)
!85 = !DILexicalBlockFile(scope: !70, file: !3, discriminator: 186646655)
!86 = !DILocation(line: 19, column: 8, scope: !70)
!87 = !DILocation(line: 20, column: 15, scope: !70)
!88 = !DILocation(line: 20, column: 11, scope: !89)
!89 = !DILexicalBlockFile(scope: !70, file: !3, discriminator: 186646663)
!90 = !DILocation(line: 20, column: 8, scope: !70)
!91 = !DILocation(line: 14, column: 36, scope: !66)
!92 = !DILocation(line: 14, column: 3, scope: !66)
!93 = distinct !{!93, !68, !94, !95}
!94 = !DILocation(line: 21, column: 3, scope: !51)
!95 = !{!"llvm.loop.mustprogress"}
!96 = !DILocation(line: 0, scope: !46)
!97 = !DILocation(line: 22, column: 1, scope: !46)
