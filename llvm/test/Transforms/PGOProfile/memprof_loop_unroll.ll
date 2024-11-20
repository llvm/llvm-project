;; Tests memprof when contains loop unroll.

;; Avoid failures on big-endian systems that can't read the profile properly
; REQUIRES: x86_64-linux

;; TODO: Use text profile inputs once that is available for memprof.
;; # To update the Inputs below, run Inputs/update_memprof_inputs.sh.
;; # To generate below LLVM IR for use in matching.
;; $ clang++ -gmlt -fdebug-info-for-profiling -S %S/Inputs/memprof_loop_unroll_b.cc -emit-llvm

; RUN: llvm-profdata merge %S/Inputs/memprof_loop_unroll.memprofraw --profiled-binary %S/Inputs/memprof_loop_unroll.exe -o %t.memprofdata
; RUN: opt < %s -passes='memprof-use<profile-filename=%t.memprofdata>' -S | FileCheck %s

; CHECK: call {{.*}} @_Znam{{.*}} #[[ATTR:[0-9]+]]
; CHECK: attributes #[[ATTR]] = { builtin allocsize(0) "memprof"="notcold" }
; CHECK-NOT: stackIds: ()

; ModuleID = 'memprof_loop_unroll_b.cc'
source_filename = "memprof_loop_unroll_b.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = external global [2 x ptr], align 16

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local void @_Z3foov() #0 !dbg !10 {
entry:
  %i = alloca i32, align 4
  store i32 0, ptr %i, align 4, !dbg !13
  br label %for.cond, !dbg !14

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4, !dbg !15
  %cmp = icmp slt i32 %0, 2, !dbg !17
  br i1 %cmp, label %for.body, label %for.end, !dbg !18

for.body:                                         ; preds = %for.cond
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 4) #2, !dbg !19
  %1 = load i32, ptr %i, align 4, !dbg !20
  %idxprom = sext i32 %1 to i64, !dbg !21
  %arrayidx = getelementptr inbounds [2 x ptr], ptr @a, i64 0, i64 %idxprom, !dbg !21
  store ptr %call, ptr %arrayidx, align 8, !dbg !22
  br label %for.inc, !dbg !23

for.inc:                                          ; preds = %for.body
  %2 = load i32, ptr %i, align 4, !dbg !24
  %inc = add nsw i32 %2, 1, !dbg !24
  store i32 %inc, ptr %i, align 4, !dbg !24
  br label %for.cond, !dbg !26, !llvm.loop !27

for.end:                                          ; preds = %for.cond
  ret void, !dbg !30
}

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znam(i64 noundef) #1

attributes #0 = { mustprogress noinline optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nobuiltin allocsize(0) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { builtin allocsize(0) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 18.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "memprof_loop_unroll_b.cc", directory: "/", checksumkind: CSK_MD5, checksum: "00276e590d606451dc54f3ff5f4bba25")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 18.0.0"}
!10 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 4, type: !11, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{}
!13 = !DILocation(line: 5, column: 14, scope: !10)
!14 = !DILocation(line: 5, column: 10, scope: !10)
!15 = !DILocation(line: 5, column: 21, scope: !16)
!16 = !DILexicalBlockFile(scope: !10, file: !1, discriminator: 2)
!17 = !DILocation(line: 5, column: 23, scope: !16)
!18 = !DILocation(line: 5, column: 5, scope: !16)
!19 = !DILocation(line: 6, column: 16, scope: !10)
!20 = !DILocation(line: 6, column: 11, scope: !10)
!21 = !DILocation(line: 6, column: 9, scope: !10)
!22 = !DILocation(line: 6, column: 14, scope: !10)
!23 = !DILocation(line: 7, column: 5, scope: !10)
!24 = !DILocation(line: 5, column: 28, scope: !25)
!25 = !DILexicalBlockFile(scope: !10, file: !1, discriminator: 4)
!26 = !DILocation(line: 5, column: 5, scope: !25)
!27 = distinct !{!27, !28, !23, !29}
!28 = !DILocation(line: 5, column: 5, scope: !10)
!29 = !{!"llvm.loop.mustprogress"}
!30 = !DILocation(line: 8, column: 1, scope: !10)