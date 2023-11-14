; RUN: opt < %s -disable-output -passes='default<O2>' -time-passes 2>&1 | FileCheck %s --check-prefix=TIME
;
; For new pass manager, check that -time-passes-per-run emit one report for each pass run.
; RUN: opt < %s -disable-output -passes='coro-cleanup,function(instcombine,instcombine,loop-mssa(licm))' -time-passes-per-run 2>&1 | FileCheck %s --check-prefix=TIME --check-prefix=TIME-PER-RUN --check-prefix=TIME-PER-RUN-CORO
; RUN: opt < %s -disable-output -passes='instcombine,loop-mssa(licm),instcombine,loop-mssa(licm)' -time-passes-per-run 2>&1 | FileCheck %s --check-prefix=TIME --check-prefix=TIME-PER-RUN -check-prefix=TIME-DOUBLE-LICM
;
; For new pass manager, check that -time-passes emit one report for each pass.
; RUN: opt < %s -disable-output -passes='coro-cleanup,function(instcombine,instcombine,loop-mssa(licm))' -time-passes 2>&1 | FileCheck %s --check-prefixes=TIME,TIME-PER-PASS,TIME-PER-PASS-CORO
; RUN: opt < %s -disable-output -passes='instcombine,loop-mssa(licm),instcombine,loop-mssa(licm)' -time-passes 2>&1 | FileCheck %s --check-prefixes=TIME,TIME-PER-PASS
;
; The following 2 test runs verify -info-output-file interaction (default goes to stderr, '-' goes to stdout).
; RUN: opt < %s -disable-output -passes='default<O2>' -time-passes -info-output-file='-' 2>/dev/null | FileCheck %s --check-prefix=TIME
;
; RUN: rm -f %t; opt < %s -disable-output -passes='default<O2>' -time-passes -info-output-file=%t
; RUN:   cat %t | FileCheck %s --check-prefix=TIME
;
; TIME: Pass execution timing report
; TIME: Total Execution Time:
; TIME: Name
; TIME-PER-RUN-DAG:      InstCombinePass #1
; TIME-PER-RUN-DAG:      InstCombinePass #2
; TIME-PER-RUN-DAG:      InstCombinePass #3
; TIME-PER-RUN-DAG:      InstCombinePass #4
; TIME-PER-RUN-DAG:      LICMPass #1
; TIME-PER-RUN-DAG:      LICMPass #2
; TIME-PER-RUN-DAG:      LICMPass #3
; TIME-DOUBLE-LICM-DAG:      LICMPass #4
; TIME-DOUBLE-LICM-DAG:      LICMPass #5
; TIME-DOUBLE-LICM-DAG:      LICMPass #6
; TIME-PER-RUN-DAG:      LCSSAPass
; TIME-PER-RUN-DAG:      LoopSimplifyPass
; TIME-PER-RUN-DAG:      VerifierPass
; TIME-PER-RUN-CORO-DAG:     SimplifyCFGPass #1
; TIME-PER-RUN-CORO-DAG:     CoroCleanupPass #1
; TIME-PER-PASS-DAG:   InstCombinePass
; TIME-PER-PASS-DAG:   LICMPass
; TIME-PER-PASS-DAG:   LCSSAPass
; TIME-PER-PASS-DAG:   LoopSimplifyPass
; TIME-PER-PASS-DAG:   VerifierPass
; TIME-PER-PASS-CORO-DAG:    SimplifyCFGPass
; TIME-PER-PASS-CORO-DAG:    CoroCleanupPass
; TIME-PER-PASS-NOT:   InstCombinePass #
; TIME-PER-PASS-NOT:   LICMPass #
; TIME-PER-PASS-NOT:   LCSSAPass #
; TIME-PER-PASS-NOT:   LoopSimplifyPass #
; TIME-PER-PASS-NOT:   VerifierPass #
; TIME: Total{{$}}

; TIME: Analysis execution timing report
; TIME: Total Execution Time:
; TIME: Name
; TIME-PER-RUN-DAG:      ScalarEvolutionAnalysis
; TIME-PER-RUN-DAG:      LoopAnalysis
; TIME-PER-RUN-DAG:      DominatorTreeAnalysis
; TIME-PER-RUN-DAG:      TargetLibraryAnalysis
; TIME-PER-PASS-DAG:   ScalarEvolutionAnalysis
; TIME-PER-PASS-DAG:   LoopAnalysis
; TIME-PER-PASS-DAG:   DominatorTreeAnalysis
; TIME-PER-PASS-DAG:   TargetLibraryAnalysis
; TIME-PER-PASS-NOT:   ScalarEvolutionAnalysis #
; TIME-PER-PASS-NOT:   LoopAnalysis #
; TIME-PER-PASS-NOT:   DominatorTreeAnalysis #
; TIME-PER-PASS-NOT:   TargetLibraryAnalysis #
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

define void @baz_coro() {
  %unused = call ptr @llvm.coro.begin(token none, ptr null)
  ret void
}

declare ptr @llvm.coro.begin(token, ptr)
