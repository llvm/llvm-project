; REQUIRES: x86_64-linux
; REQUIRES: asserts
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/non-probe-stale-profile-matching.prof --salvage-stale-profile -S --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl -profile-isfs 2>&1 | FileCheck %s

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

; CHECK: Run stale profile matching for bar

; CHECK: Run stale profile matching for foo
; CHECK: Callsite with callee:bar is matched from 1.15 to 1.15
; CHECK: Callsite with callee:bar is matched from 2 to 2

; CHECK: Run stale profile matching for main
; CHECK: Callsite with callee:foo is matched from 4 to 2
; CHECK: Callsite with callee:bar is matched from 5 to 3
; CHECK: Callsite with callee:foo is matched from 8 to 4
; CHECK: Callsite with callee:bar is matched from 9 to 5

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = dso_local global i32 1, align 4

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @bar(i32 noundef %p) #0 !dbg !9 {
entry:
  ret i32 %p, !dbg !13
}

; Function Attrs: alwaysinline nounwind uwtable
define dso_local i32 @foo(i32 noundef %i, i32 noundef %p) #1 !dbg !14 {
entry:
  %rem = srem i32 %i, 10, !dbg !15
  %tobool = icmp ne i32 %rem, 0, !dbg !15
  br i1 %tobool, label %if.then, label %if.else, !dbg !16

if.then:                                          ; preds = %entry
  %call = call i32 @bar(i32 noundef %p), !dbg !17
  br label %return, !dbg !19

if.else:                                          ; preds = %entry
  %add = add nsw i32 %p, 1, !dbg !20
  %call1 = call i32 @bar(i32 noundef %add), !dbg !21
  br label %return, !dbg !22

return:                                           ; preds = %if.else, %if.then
  %retval.0 = phi i32 [ %call, %if.then ], [ %call1, %if.else ], !dbg !23
  ret i32 %retval.0, !dbg !24
}

; Function Attrs: nounwind uwtable
define dso_local i32 @main() #2 !dbg !25 {
entry:
  %0 = load volatile i32, ptr @x, align 4, !dbg !26, !tbaa !27
  %cmp = icmp eq i32 %0, 0, !dbg !31
  br i1 %cmp, label %if.then, label %if.end, !dbg !26

if.then:                                          ; preds = %entry
  br label %for.end, !dbg !32

if.end:                                           ; preds = %entry
  br label %for.cond, !dbg !33

for.cond:                                         ; preds = %if.end6, %if.end
  %i.0 = phi i32 [ 0, %if.end ], [ %inc, %if.end6 ], !dbg !34
  %cmp1 = icmp slt i32 %i.0, 1000000, !dbg !35
  br i1 %cmp1, label %for.body, label %for.cond.cleanup, !dbg !37

for.cond.cleanup:                                 ; preds = %for.cond
  br label %cleanup, !dbg !38

for.body:                                         ; preds = %for.cond
  %1 = load volatile i32, ptr @x, align 4, !dbg !40, !tbaa !27
  %call = call i32 @foo(i32 noundef %i.0, i32 noundef %1), !dbg !41
  %2 = load volatile i32, ptr @x, align 4, !dbg !42, !tbaa !27
  %add = add nsw i32 %2, %call, !dbg !42
  store volatile i32 %add, ptr @x, align 4, !dbg !42, !tbaa !27
  %3 = load volatile i32, ptr @x, align 4, !dbg !43, !tbaa !27
  %call2 = call i32 @bar(i32 noundef %3), !dbg !44
  %4 = load volatile i32, ptr @x, align 4, !dbg !45, !tbaa !27
  %add3 = add nsw i32 %4, %call2, !dbg !45
  store volatile i32 %add3, ptr @x, align 4, !dbg !45, !tbaa !27
  br i1 false, label %if.then5, label %if.end6, !dbg !46

if.then5:                                         ; preds = %for.body
  br label %cleanup, !dbg !47

if.end6:                                          ; preds = %for.body
  %5 = load volatile i32, ptr @x, align 4, !dbg !48, !tbaa !27
  %call7 = call i32 @foo(i32 noundef %i.0, i32 noundef %5), !dbg !49
  %6 = load volatile i32, ptr @x, align 4, !dbg !50, !tbaa !27
  %add8 = add nsw i32 %6, %call7, !dbg !50
  store volatile i32 %add8, ptr @x, align 4, !dbg !50, !tbaa !27
  %7 = load volatile i32, ptr @x, align 4, !dbg !51, !tbaa !27
  %call9 = call i32 @bar(i32 noundef %7), !dbg !52
  %8 = load volatile i32, ptr @x, align 4, !dbg !53, !tbaa !27
  %add10 = add nsw i32 %8, %call9, !dbg !53
  store volatile i32 %add10, ptr @x, align 4, !dbg !53, !tbaa !27
  %inc = add nsw i32 %i.0, 1, !dbg !54
  br label %for.cond, !dbg !56, !llvm.loop !57

cleanup:                                          ; preds = %if.then5, %for.cond.cleanup
  br label %for.end

for.end:                                          ; preds = %cleanup, %if.then
  ret i32 0, !dbg !61
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #3

attributes #0 = { noinline nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #1 = { alwaysinline nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #2 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 19.0.0git", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "path")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{!"clang version 19.0.0git"}
!9 = distinct !DISubprogram(name: "bar", scope: !10, file: !10, line: 2, type: !11, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!10 = !DIFile(filename: "test.c", directory: "path")
!11 = !DISubroutineType(types: !12)
!12 = !{}
!13 = !DILocation(line: 3, column: 3, scope: !9)
!14 = distinct !DISubprogram(name: "foo", scope: !10, file: !10, line: 6, type: !11, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!15 = !DILocation(line: 7, column: 9, scope: !14)
!16 = !DILocation(line: 7, column: 7, scope: !14)
!17 = !DILocation(line: 7, column: 23, scope: !18)
!18 = !DILexicalBlockFile(scope: !14, file: !10, discriminator: 15)
!19 = !DILocation(line: 7, column: 15, scope: !18)
!20 = !DILocation(line: 8, column: 21, scope: !14)
!21 = !DILocation(line: 8, column: 15, scope: !14)
!22 = !DILocation(line: 8, column: 8, scope: !14)
!23 = !DILocation(line: 0, scope: !14)
!24 = !DILocation(line: 9, column: 1, scope: !14)
!25 = distinct !DISubprogram(name: "main", scope: !10, file: !10, line: 11, type: !11, scopeLine: 11, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!26 = !DILocation(line: 12, column: 7, scope: !25)
!27 = !{!28, !28, i64 0}
!28 = !{!"int", !29, i64 0}
!29 = !{!"omnipotent char", !30, i64 0}
!30 = !{!"Simple C/C++ TBAA"}
!31 = !DILocation(line: 12, column: 9, scope: !25)
!32 = !DILocation(line: 13, column: 5, scope: !25)
!33 = !DILocation(line: 14, column: 8, scope: !25)
!34 = !DILocation(line: 14, scope: !25)
!35 = !DILocation(line: 14, column: 21, scope: !36)
!36 = !DILexicalBlockFile(scope: !25, file: !10, discriminator: 15)
!37 = !DILocation(line: 14, column: 3, scope: !36)
!38 = !DILocation(line: 14, column: 3, scope: !39)
!39 = !DILexicalBlockFile(scope: !25, file: !10, discriminator: 4)
!40 = !DILocation(line: 15, column: 18, scope: !25)
!41 = !DILocation(line: 15, column: 11, scope: !25)
!42 = !DILocation(line: 15, column: 8, scope: !25)
!43 = !DILocation(line: 16, column: 15, scope: !25)
!44 = !DILocation(line: 16, column: 11, scope: !25)
!45 = !DILocation(line: 16, column: 8, scope: !25)
!46 = !DILocation(line: 17, column: 10, scope: !25)
!47 = !DILocation(line: 18, column: 8, scope: !25)
!48 = !DILocation(line: 19, column: 18, scope: !25)
!49 = !DILocation(line: 19, column: 11, scope: !25)
!50 = !DILocation(line: 19, column: 8, scope: !25)
!51 = !DILocation(line: 20, column: 15, scope: !25)
!52 = !DILocation(line: 20, column: 11, scope: !25)
!53 = !DILocation(line: 20, column: 8, scope: !25)
!54 = !DILocation(line: 14, column: 37, scope: !55)
!55 = !DILexicalBlockFile(scope: !25, file: !10, discriminator: 6)
!56 = !DILocation(line: 14, column: 3, scope: !55)
!57 = distinct !{!57, !58, !59, !60}
!58 = !DILocation(line: 14, column: 3, scope: !25)
!59 = !DILocation(line: 21, column: 3, scope: !25)
!60 = !{!"llvm.loop.mustprogress"}
!61 = !DILocation(line: 22, column: 1, scope: !25)
