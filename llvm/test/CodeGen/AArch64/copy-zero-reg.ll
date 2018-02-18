; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu | FileCheck %s

; Verify there is no tiny block having only one mov wzr instruction between for.body.lr.ph and sw.epilog.loopexit
define void @unroll_by_2(i32 %trip_count, i32* %p) {
; CHECK-LABEL: unroll_by_2
; CHECK: // %for.body.lr.ph
; CHECK:     mov   w{{[0-9]+}}, wzr
; CHECK:     b.eq
; CHECK-NOT: mov   w{{[0-9]+}}, wzr
; CHECK: // %for.body.lr.ph.new
; CHECK: // %for.body
; CHECK: // %sw.epilog.loopexit
; CHECK: // %for.body.epil
; CHECK: // %exit
; CHECK-NEXT:   ret
for.body.lr.ph:
  %xtraiter = and i32 %trip_count, 1
  %cmp = icmp eq i32 %trip_count, 1
  br i1 %cmp, label %sw.epilog.loopexit, label %for.body.lr.ph.new

for.body.lr.ph.new:
  %unroll_iter = sub nsw i32 %trip_count, %xtraiter
  br label %for.body

for.body:
  %indvars = phi i32 [ 0, %for.body.lr.ph.new ], [ %indvars.next, %for.body ]
  %niter = phi i32 [ %unroll_iter, %for.body.lr.ph.new ], [ %niter.nsub, %for.body ]
  %array = getelementptr inbounds i32, i32 * %p, i32 %indvars
  store  i32 %niter, i32* %array
  %indvars.next = add i32 %indvars, 2
  %niter.nsub = add i32 %niter, -2
  %niter.ncmp = icmp eq i32 %niter.nsub, 0
  br i1 %niter.ncmp, label %sw.epilog.loopexit, label %for.body

sw.epilog.loopexit:
  %indvars.unr = phi i32 [ 0, %for.body.lr.ph ], [ %indvars.next, %for.body ]
  %lcmp.mod = icmp eq i32 %xtraiter, 0
  br i1 %lcmp.mod, label %exit, label %for.body.epil

for.body.epil:
  %array.epil = getelementptr inbounds i32, i32* %p, i32 %indvars.unr
  store  i32 %indvars.unr, i32* %array.epil
  br label %exit

exit:
  ret void
}
