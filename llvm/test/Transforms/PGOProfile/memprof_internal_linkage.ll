;; Tests memprof when contains internal linkage function.

;; Avoid failures on big-endian systems that can't read the profile properly
; REQUIRES: x86_64-linux

;; TODO: Use text profile inputs once that is available for memprof.
;; # To update the Inputs below, run Inputs/update_memprof_inputs.sh.
;; # To generate below LLVM IR for use in matching.
;; $ clang++ -gmlt -fdebug-info-for-profiling -S %S/Inputs/memprof_internal_linkage.cc -emit-llvm -funique-internal-linkage-names

; RUN: llvm-profdata merge %S/Inputs/memprof_internal_linkage.memprofraw --profiled-binary %S/Inputs/memprof_internal_linkage.exe -o %t.memprofdata
; RUN: opt < %s -passes='memprof-use<profile-filename=%t.memprofdata>' -S | FileCheck %s

; CHECK: call {{.*}} @_Znam{{.*}} #[[ATTR:[0-9]+]]
; CHECK: attributes #[[ATTR]] = { builtin allocsize(0) "memprof"="notcold" }

; ModuleID = 'memprof_internal_linkage.cc'
source_filename = "memprof_internal_linkage.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local noundef i32 @main(i32 noundef %argc, ptr noundef %argv) #0 !dbg !10 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca ptr, align 8
  store i32 0, ptr %retval, align 4
  store i32 %argc, ptr %argc.addr, align 4
  store ptr %argv, ptr %argv.addr, align 8
  call void @_ZL3foov.__uniq.231888424933890731874095357293037629092() #4, !dbg !14
  ret i32 0, !dbg !15
}

; Function Attrs: mustprogress noinline optnone uwtable
define internal void @_ZL3foov.__uniq.231888424933890731874095357293037629092() #1 !dbg !16 {
entry:
  %a = alloca ptr, align 8
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 20) #5, !dbg !17
  store ptr %call, ptr %a, align 8, !dbg !18
  %0 = load ptr, ptr %a, align 8, !dbg !19
  call void @llvm.memset.p0.i64(ptr align 4 %0, i8 0, i64 5, i1 false), !dbg !20
  ret void, !dbg !21
}

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znam(i64 noundef) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #3

attributes #0 = { mustprogress noinline norecurse optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress noinline optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "sample-profile-suffix-elision-policy"="selected" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nobuiltin allocsize(0) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #4 = { "sample-profile-suffix-elision-policy"="selected" }
attributes #5 = { builtin allocsize(0) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 18.0.0 (https://github.com/llvm/llvm-project.git a604a1112a611ea867dc4e8d164021c7b055e18a)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "memprof_internal_linkage.cc", directory: ".", checksumkind: CSK_MD5, checksum: "de848419432086fd9ed6dda04f3bf0ac")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 18.0.0 (https://github.com/llvm/llvm-project.git a604a1112a611ea867dc4e8d164021c7b055e18a)"}
!10 = distinct !DISubprogram(name: "main", scope: !11, file: !11, line: 7, type: !12, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DIFile(filename: "memprof_internal_linkage.cc", directory: ".", checksumkind: CSK_MD5, checksum: "de848419432086fd9ed6dda04f3bf0ac")
!12 = !DISubroutineType(types: !13)
!13 = !{}
!14 = !DILocation(line: 8, column: 3, scope: !10)
!15 = !DILocation(line: 9, column: 3, scope: !10)
!16 = distinct !DISubprogram(name: "foo", linkageName: "_ZL3foov.__uniq.231888424933890731874095357293037629092", scope: !11, file: !11, line: 3, type: !12, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0)
!17 = !DILocation(line: 4, column: 12, scope: !16)
!18 = !DILocation(line: 4, column: 8, scope: !16)
!19 = !DILocation(line: 5, column: 10, scope: !16)
!20 = !DILocation(line: 5, column: 3, scope: !16)
!21 = !DILocation(line: 6, column: 1, scope: !16)