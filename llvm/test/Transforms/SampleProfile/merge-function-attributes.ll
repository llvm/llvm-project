; REQUIRES: x86_64-linux
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/merge-function-attributes.afdo -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

; Verify that yyy is inlined into xxx with the function attibutes properly merged.
; CHECK:      define <8 x double> @xxx(){{.*}} #[[ATTRNO:[0-9]+]]
; CHECK-NEXT: call <8 x double> @llvm.x86.avx512.mask.rndscale.pd.512
define <8 x double> @xxx() #0 !dbg !5 {
  %x = call <8 x double> @yyy(), !dbg !7
  ret <8 x double> %x
}

define available_externally <8 x double> @yyy() #1 !dbg !8 {
  %y = call <8 x double> @llvm.x86.avx512.mask.rndscale.pd.512(<8 x double> zeroinitializer, i32 9, <8 x double> zeroinitializer, i8 -1, i32 4)
  ret <8 x double> %y
}

; Function Attrs: nounwind readnone
declare <8 x double> @llvm.x86.avx512.mask.rndscale.pd.512(<8 x double>, i32 immarg, <8 x double>, i8, i32 immarg) #2

; CHECK: attributes #[[ATTRNO]] = { "min-legal-vector-width"="512"
attributes #0 = { "min-legal-vector-width"="128" "prefer-vector-width"="128" "target-features"="+avx512vl" "use-sample-profile" }
attributes #1 = { "min-legal-vector-width"="512" "use-sample-profile" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "test.cc", directory: "/proc/self/cwd")
!2 = !{}
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "xxx", linkageName: "xxx", scope: !1, file: !1, line: 11, type: !6, scopeLine: 14, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!6 = !DISubroutineType(types: !2)
!7 = !DILocation(line: 78, column: 10, scope: !5)
!8 = distinct !DISubprogram(name: "yyy", linkageName: "yyy", scope: !1, file: !1, line: 270, type: !6, scopeLine: 273, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !3, retainedNodes: !2)
