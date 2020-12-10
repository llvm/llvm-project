; RUN: opt -enable-new-pm=0 < %s -disable-output -instcombine -instcombine -licm -time-passes 2>&1 | FileCheck %s --check-prefix=TIME --check-prefix=TIME-LEGACY
; RUN: opt -enable-new-pm=0 < %s -disable-output -instcombine -instcombine -licm -licm -time-passes 2>&1 | FileCheck %s --check-prefix=TIME --check-prefix=TIME-LEGACY --check-prefix=TIME-DOUBLE-LICM-LEGACY
; RUN: opt < %s -disable-output -passes='default<O2>' -time-passes 2>&1 | FileCheck %s --check-prefix=TIME
;
; For new pass manager, check that -time-passes-per-run emit one report for each pass run.
; RUN: opt < %s -disable-output -passes='instcombine,instcombine,loop(licm)' -time-passes-per-run 2>&1 | FileCheck %s --check-prefix=TIME --check-prefix=TIME-NEW
; RUN: opt < %s -disable-output -passes='instcombine,loop(licm),instcombine,loop(licm)' -time-passes-per-run 2>&1 | FileCheck %s --check-prefix=TIME --check-prefix=TIME-NEW -check-prefix=TIME-DOUBLE-LICM-NEW
;
; For new pass manager, check that -time-passes emit one report for each pass.
; RUN: opt < %s -disable-output -passes='instcombine,instcombine,loop(licm)' -time-passes 2>&1 | FileCheck %s --check-prefixes=TIME,TIME-NEW-PER-PASS
; RUN: opt < %s -disable-output -passes='instcombine,loop(licm),instcombine,loop(licm)' -time-passes 2>&1 | FileCheck %s --check-prefixes=TIME,TIME-NEW-PER-PASS
;
; The following 4 test runs verify -info-output-file interaction (default goes to stderr, '-' goes to stdout).
; RUN: opt -enable-new-pm=0 < %s -disable-output -O2 -time-passes -info-output-file='-' 2>/dev/null | FileCheck %s --check-prefix=TIME
; RUN: opt < %s -disable-output -passes='default<O2>' -time-passes -info-output-file='-' 2>/dev/null | FileCheck %s --check-prefix=TIME
;
; RUN: rm -f %t; opt < %s -disable-output -O2 -time-passes -info-output-file=%t
; RUN:   cat %t | FileCheck %s --check-prefix=TIME
;
; RUN: rm -f %t; opt < %s -disable-output -passes='default<O2>' -time-passes -info-output-file=%t
; RUN:   cat %t | FileCheck %s --check-prefix=TIME
;
; TIME: Pass execution timing report
; TIME: Total Execution Time:
; TIME: Name
; TIME-LEGACY-DAG:   Combine redundant instructions{{$}}
; TIME-LEGACY-DAG:   Combine redundant instructions #2
; TIME-LEGACY-DAG:   Loop Invariant Code Motion{{$}}
; TIME-DOUBLE-LICM-LEGACY-DAG: Loop Invariant Code Motion #2
; TIME-LEGACY-DAG:   Scalar Evolution Analysis
; TIME-LEGACY-DAG:   Loop-Closed SSA Form Pass
; TIME-LEGACY-DAG:   LCSSA Verifier
; TIME-LEGACY-DAG:   Canonicalize natural loops
; TIME-LEGACY-DAG:   Natural Loop Information
; TIME-LEGACY-DAG:   Dominator Tree Construction
; TIME-LEGACY-DAG:   Module Verifier
; TIME-LEGACY-DAG:   Target Library Information
; TIME-NEW-DAG:      InstCombinePass #1
; TIME-NEW-DAG:      InstCombinePass #2
; TIME-NEW-DAG:      InstCombinePass #3
; TIME-NEW-DAG:      InstCombinePass #4
; TIME-NEW-DAG:      LICMPass #1
; TIME-NEW-DAG:      LICMPass #2
; TIME-NEW-DAG:      LICMPass #3
; TIME-DOUBLE-LICM-NEW-DAG:      LICMPass #4
; TIME-DOUBLE-LICM-NEW-DAG:      LICMPass #5
; TIME-DOUBLE-LICM-NEW-DAG:      LICMPass #6
; TIME-NEW-DAG:      LCSSAPass
; TIME-NEW-DAG:      LoopSimplifyPass
; TIME-NEW-DAG:      ScalarEvolutionAnalysis
; TIME-NEW-DAG:      LoopAnalysis
; TIME-NEW-DAG:      VerifierPass
; TIME-NEW-DAG:      DominatorTreeAnalysis
; TIME-NEW-DAG:      TargetLibraryAnalysis
; TIME-NEW-PER-PASS-DAG:   InstCombinePass
; TIME-NEW-PER-PASS-DAG:   LICMPass
; TIME-NEW-PER-PASS-DAG:   LCSSAPass
; TIME-NEW-PER-PASS-DAG:   LoopSimplifyPass
; TIME-NEW-PER-PASS-DAG:   ScalarEvolutionAnalysis
; TIME-NEW-PER-PASS-DAG:   LoopAnalysis
; TIME-NEW-PER-PASS-DAG:   VerifierPass
; TIME-NEW-PER-PASS-DAG:   DominatorTreeAnalysis
; TIME-NEW-PER-PASS-DAG:   TargetLibraryAnalysis
; TIME-NEW-PER-PASS-NOT:   InstCombinePass #
; TIME-NEW-PER-PASS-NOT:   LICMPass #
; TIME-NEW-PER-PASS-NOT:   LCSSAPass #
; TIME-NEW-PER-PASS-NOT:   LoopSimplifyPass #
; TIME-NEW-PER-PASS-NOT:   ScalarEvolutionAnalysis #
; TIME-NEW-PER-PASS-NOT:   LoopAnalysis #
; TIME-NEW-PER-PASS-NOT:   VerifierPass #
; TIME-NEW-PER-PASS-NOT:   DominatorTreeAnalysis #
; TIME-NEW-PER-PASS-NOT:   TargetLibraryAnalysis #
; TIME: Total{{$}}

define i32 @foo() {
  %res = add i32 5, 4
  br label %loop1
loop1:
  br i1 false, label %loop1, label %end
end:
  ret i32 %res
}

define void @bar_with_loops() {
  br label %loop1
loop1:
  br i1 false, label %loop1, label %loop2
loop2:
  br i1 true, label %loop2, label %end
end:
  ret void

}
