; This file verifies the behavior of the OptBisect class, which is used to
; diagnose optimization related failures.  The tests check various
; invocations that result in different sets of optimization passes that
; are run in different ways.
;
; This set of tests exercises the legacy pass manager interface to the OptBisect
; class.  Because the exact set of optimizations that will be run may
; change over time, these tests are written in a more general manner than the
; corresponding tests for the new pass manager.
;
; Don't use NEXT checks or hard-code pass numbering so that this won't fail if
; new passes are inserted.


; Verify that the file can be compiled to an object file at -O3 with all
; skippable passes skipped.

; REQUIRES: default_triple
; RUN: opt -O3 -opt-bisect-limit=0 < %s | llc -O3 -opt-bisect-limit=0


; Verify that no skippable passes are run with -opt-bisect-limit=0.

; RUN: opt -disable-output -disable-verify -O3 -opt-bisect-limit=0 %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-SKIP-ALL
; CHECK-SKIP-ALL: BISECT: NOT running pass ({{[0-9]+}})
; CHECK-SKIP-ALL-NOT: BISECT: running pass ({{[0-9]+}})


; Verify that no passes run at -O0 are skipped
; RUN: opt -opt-bisect-limit=0 < %s 2>&1 | FileCheck %s --check-prefix=OPTBISECT-O0
; OPTBISECT-O0-NOT: BISECT: NOT running

; FIXME: There are still some AMDGPU passes being skipped that run at -O0.
; XFAIL: r600, amdgcn

; Verify that we can use the opt-bisect-helper.py script (derived from
; utils/bisect) to locate the optimization that inlines the call to
; f2() in f3().

; RUN: %python %S/opt-bisect-helper.py --start=0 --end=256 --optcmd=opt \
; RUN:         --filecheckcmd=FileCheck --test=%s \
; RUN:         --prefix=CHECK-BISECT-INLINE-HELPER \
; RUN:         | FileCheck %s --check-prefix=CHECK-BISECT-INLINE-RESULT
; The helper script uses this to find the optimization that inlines the call.
; CHECK-BISECT-INLINE-HELPER: call i32 @f2()
; These checks verifies that the optimization was found.
; CHECK-BISECT-INLINE-RESULT-NOT: Last good count: 0
; CHECK-BISECT-INLINE-RESULT: Last good count: {{[0-9]+}}


; Test a module pass.

; RUN: opt -disable-output -disable-verify -deadargelim -opt-bisect-limit=-1 %s \
; RUN:     2>&1 | FileCheck %s --check-prefix=CHECK-DEADARG
; CHECK-DEADARG: BISECT: running pass ({{[0-9]+}}) Dead Argument Elimination on module

; RUN: opt -disable-output -disable-verify -deadargelim -opt-bisect-limit=0 %s \
; RUN:     2>&1 | FileCheck %s --check-prefix=CHECK-NOT-DEADARG
; CHECK-NOT-DEADARG: BISECT: NOT running pass ({{[0-9]+}}) Dead Argument Elimination on module


; Test an SCC pass.

; RUN: opt -disable-output -disable-verify -inline -opt-bisect-limit=-1 %s \
; RUN:     2>&1 | FileCheck %s --check-prefix=CHECK-INLINE
; CHECK-INLINE: BISECT: running pass ({{[0-9]+}}) Function Integration/Inlining on SCC (<<null function>>)
; CHECK-INLINE: BISECT: running pass ({{[0-9]+}}) Function Integration/Inlining on SCC (g)
; CHECK-INLINE: BISECT: running pass ({{[0-9]+}}) Function Integration/Inlining on SCC (f1)
; CHECK-INLINE: BISECT: running pass ({{[0-9]+}}) Function Integration/Inlining on SCC (f2)
; CHECK-INLINE: BISECT: running pass ({{[0-9]+}}) Function Integration/Inlining on SCC (f3)
; CHECK-INLINE: BISECT: running pass ({{[0-9]+}}) Function Integration/Inlining on SCC (<<null function>>)

; RUN: opt -disable-output -disable-verify -inline -opt-bisect-limit=0 %s \
; RUN:     2>&1 | FileCheck %s --check-prefix=CHECK-NOT-INLINE
; CHECK-NOT-INLINE: BISECT: NOT running pass ({{[0-9]+}}) Function Integration/Inlining on SCC (<<null function>>)
; CHECK-NOT-INLINE: BISECT: NOT running pass ({{[0-9]+}}) Function Integration/Inlining on SCC (g)
; CHECK-NOT-INLINE: BISECT: NOT running pass ({{[0-9]+}}) Function Integration/Inlining on SCC (f1)
; CHECK-NOT-INLINE: BISECT: NOT running pass ({{[0-9]+}}) Function Integration/Inlining on SCC (f2)
; CHECK-NOT-INLINE: BISECT: NOT running pass ({{[0-9]+}}) Function Integration/Inlining on SCC (f3)
; CHECK-NOT-INLINE: BISECT: NOT running pass ({{[0-9]+}}) Function Integration/Inlining on SCC (<<null function>>)


; Test a function pass.

; RUN: opt -disable-output -disable-verify -early-cse -earlycse-debug-hash -opt-bisect-limit=-1 \
; RUN:     %s 2>&1 | FileCheck %s --check-prefix=CHECK-EARLY-CSE
; CHECK-EARLY-CSE: BISECT: running pass ({{[0-9]+}}) Early CSE on function (f1)
; CHECK-EARLY-CSE: BISECT: running pass ({{[0-9]+}}) Early CSE on function (f2)
; CHECK-EARLY-CSE: BISECT: running pass ({{[0-9]+}}) Early CSE on function (f3)

; RUN: opt -disable-output -disable-verify -early-cse -earlycse-debug-hash -opt-bisect-limit=0 \
; RUN:     %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOT-EARLY-CSE
; CHECK-NOT-EARLY-CSE: BISECT: NOT running pass ({{[0-9]+}}) Early CSE on function (f1)
; CHECK-NOT-EARLY-CSE: BISECT: NOT running pass ({{[0-9]+}}) Early CSE on function (f2)
; CHECK-NOT-EARLY-CSE: BISECT: NOT running pass ({{[0-9]+}}) Early CSE on function (f3)


; Test a loop pass.

; RUN: opt -disable-output -disable-verify -loop-reduce -opt-bisect-limit=-1 \
; RUN:      %s 2>&1 | FileCheck %s --check-prefix=CHECK-LOOP-REDUCE
; CHECK-LOOP-REDUCE: BISECT: running pass ({{[0-9]+}}) Loop Strength Reduction on loop
; CHECK-LOOP-REDUCE: BISECT: running pass ({{[0-9]+}}) Loop Strength Reduction on loop
; CHECK-LOOP-REDUCE: BISECT: running pass ({{[0-9]+}}) Loop Strength Reduction on loop
; CHECK-LOOP-REDUCE: BISECT: running pass ({{[0-9]+}}) Loop Strength Reduction on loop
; CHECK-LOOP-REDUCE: BISECT: running pass ({{[0-9]+}}) Loop Strength Reduction on loop

; RUN: opt -disable-output -disable-verify -loop-reduce -opt-bisect-limit=0 \
; RUN:     %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOT-LOOP-REDUCE
; CHECK-NOT-LOOP-REDUCE: BISECT: NOT running pass ({{[0-9]+}}) Loop Strength Reduction on loop
; CHECK-NOT-LOOP-REDUCE: BISECT: NOT running pass ({{[0-9]+}}) Loop Strength Reduction on loop
; CHECK-NOT-LOOP-REDUCE: BISECT: NOT running pass ({{[0-9]+}}) Loop Strength Reduction on loop
; CHECK-NOT-LOOP-REDUCE: BISECT: NOT running pass ({{[0-9]+}}) Loop Strength Reduction on loop
; CHECK-NOT-LOOP-REDUCE: BISECT: NOT running pass ({{[0-9]+}}) Loop Strength Reduction on loop


declare i32 @g()

define void @f1() {
entry:
  br label %loop.0
loop.0:
  br i1 undef, label %loop.0.0, label %loop.1
loop.0.0:
  br i1 undef, label %loop.0.0, label %loop.0.1
loop.0.1:
  br i1 undef, label %loop.0.1, label %loop.0
loop.1:
  br i1 undef, label %loop.1, label %loop.1.bb1
loop.1.bb1:
  br i1 undef, label %loop.1, label %loop.1.bb2
loop.1.bb2:
  br i1 undef, label %end, label %loop.1.0
loop.1.0:
  br i1 undef, label %loop.1.0, label %loop.1
end:
  ret void
}

define i32 @f2() {
entry:
  ret i32 0
}

define i32 @f3() {
entry:
  %temp = call i32 @g()
  %icmp = icmp ugt i32 %temp, 2
  br i1 %icmp, label %bb.true, label %bb.false
bb.true:
  %temp2 = call i32 @f2()
  ret i32 %temp2
bb.false:
  ret i32 0
}

; This function is here to verify that opt-bisect can skip all passes for
; functions that contain lifetime intrinsics.
define void @f4() {
entry:
  %i = alloca i32, align 4
  %tmp = bitcast i32* %i to i8*
  call void @llvm.lifetime.start(i64 4, i8* %tmp)
  br label %for.cond

for.cond:
  br i1 undef, label %for.body, label %for.end

for.body:
  br label %for.cond

for.end:
  ret void
}

declare void @llvm.lifetime.start(i64, i8* nocapture)

