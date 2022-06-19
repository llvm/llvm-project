; RUN: llc < %s -verify-machineinstrs -stop-after=hardware-loops -mcpu=pwr9 \
; RUN:   -mtriple=powerpc64le-unknown-unknown | FileCheck %s
; RUN: llc < %s -verify-machineinstrs -stop-after=hardware-loops -mcpu=pwr8 \
; RUN:   -mtriple=powerpc64le-unknown-unknown | FileCheck %s

@a = internal global fp128 0xL00000000000000000000000000000000, align 16
@x = internal global [4 x fp128] zeroinitializer, align 16
@y = internal global [4 x fp128] zeroinitializer, align 16

define void @fmul_ctrloop_fp128() {
entry:
  %0 = load fp128, fp128* @a, align 16
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.06 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [4 x fp128], [4 x fp128]* @x, i64 0, i64 %i.06
  %1 = load fp128, fp128* %arrayidx, align 16
  %mul = fmul fp128 %0, %1
  %arrayidx1 = getelementptr inbounds [4 x fp128], [4 x fp128]* @y, i64 0, i64 %i.06
  store fp128 %mul, fp128* %arrayidx1, align 16
  %inc = add nuw nsw i64 %i.06, 1
  %exitcond = icmp eq i64 %inc, 4
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void

; CHECK-LABEL: fmul_ctrloop_fp128
; CHECK-NOT:     call void @llvm.set.loop.iterations.i64(i64 4)
; CHECK-NOT:     call i1 @llvm.loop.decrement.i64(i64 1)
}

define void @fpext_ctrloop_fp128(double* %a) {
entry:
  br label %for.body

for.body:
  %i.06 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %a, i64 %i.06
  %0 = load double, double* %arrayidx, align 8
  %ext = fpext double %0 to fp128
  %arrayidx1 = getelementptr inbounds [4 x fp128], [4 x fp128]* @y, i64 0, i64 %i.06
  store fp128 %ext, fp128* %arrayidx1, align 16
  %inc = add nuw nsw i64 %i.06, 1
  %exitcond = icmp eq i64 %inc, 4
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void

; CHECK-LABEL: fpext_ctrloop_fp128
; CHECK-NOT: call void @llvm.set.loop.iterations.i64(i64 4)
; CHECK-NOT: call i1 @llvm.loop.decrement.i64(i64 1)
}

define void @fptrunc_ctrloop_fp128(double* %a) {
entry:
  br label %for.body

for.body:
  %i.06 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [4 x fp128], [4 x fp128]* @x, i64 0, i64 %i.06
  %0 = load fp128, fp128* %arrayidx, align 16
  %trunc = fptrunc fp128 %0 to double
  %arrayidx1 = getelementptr inbounds double, double* %a, i64 %i.06
  store double %trunc, double* %arrayidx1, align 16
  %inc = add nuw nsw i64 %i.06, 1
  %exitcond = icmp eq i64 %inc, 4
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void

; CHECK-LABEL: fptrunc_ctrloop_fp128
; CHECK-NOT: call void @llvm.set.loop.iterations.i64(i64 4)
; CHECK-NOT: call i1 @llvm.loop.decrement.i64(i64 1)
}

declare void @obfuscate(i8*, ...) local_unnamed_addr #2
