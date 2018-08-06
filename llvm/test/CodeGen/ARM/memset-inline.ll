; RUN: llc < %s -mtriple=thumbv7-apple-ios -mcpu=cortex-a8 -pre-RA-sched=source -disable-post-ra | FileCheck %s -check-prefix=CHECK-7A
; RUN: llc < %s -mtriple=thumbv6m -pre-RA-sched=source -disable-post-ra -mattr=+strict-align | FileCheck %s -check-prefix=CHECK-6M

define void @t1(i8* nocapture %c) nounwind optsize {
entry:
; CHECK-7A-LABEL: t1:
; CHECK-7A: movs r1, #0
; CHECK-7A: strd r1, r1, [r0]
; CHECK-7A: str r1, [r0, #8]
; CHECK-6M-LABEL: t1:
; CHECK-6M: movs r1, #0
; CHECK-6M: str r1, [r0]
; CHECK-6M: str r1, [r0, #4]
; CHECK-6M: str r1, [r0, #8]
  call void @llvm.memset.p0i8.i64(i8* align 8 %c, i8 0, i64 12, i1 false)
  ret void
}

define void @t2() nounwind ssp {
entry:
; CHECK-7A-LABEL: t2:
; CHECK-7A: vmov.i32 {{q[0-9]+}}, #0x0
; CHECK-7A: movs r1, #10
; CHECK-7A: vst1.64 {d{{[0-9]+}}, d{{[0-9]+}}}, [r2], r1
; CHECK-7A: vst1.16 {d{{[0-9]+}}, d{{[0-9]+}}}, [r2]
; CHECK-6M-LABEL: t2:
; CHECK-6M: movs [[REG:r[0-9]+]], #0
; CHECK-6M: str  [[REG]], [sp, #20]
; CHECK-6M: str  [[REG]], [sp, #16]
; CHECK-6M: str  [[REG]], [sp, #12]
; CHECK-6M: str  [[REG]], [sp, #8]
; CHECK-6M: str  [[REG]], [sp, #4]
; CHECK-6M: str  [[REG]], [sp]
  %buf = alloca [26 x i8], align 1
  %0 = getelementptr inbounds [26 x i8], [26 x i8]* %buf, i32 0, i32 0
  call void @llvm.memset.p0i8.i32(i8* %0, i8 0, i32 26, i1 false)
  call void @something(i8* %0) nounwind
  ret void
}

define void @t3(i8* %p) {
entry:
; CHECK-7A-LABEL: t3:
; CHECK-7A: muls [[REG:r[0-9]+]],
; CHECK-7A: str  [[REG]],
; CHECK-6M-LABEL: t3:
; CHECK-6M-NOT: muls
; CHECK-6M: strb [[REG:r[0-9]+]],
; CHECK-6M: strb [[REG]],
; CHECK-6M: strb [[REG]],
; CHECK-6M: strb [[REG]],
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %0 = trunc i32 %i to i8
  call void @llvm.memset.p0i8.i32(i8* %p, i8 %0, i32 4, i1 false)
  call void @something(i8* %p)
  %inc = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %inc, 255
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @t4(i8* %p) {
entry:
; CHECK-7A-LABEL: t4:
; CHECK-7A: muls [[REG:r[0-9]+]],
; CHECK-7A: str  [[REG]],
; CHECK-6M-LABEL: t4:
; CHECK-6M: muls [[REG:r[0-9]+]],
; CHECK-6M: strh [[REG]],
; CHECK-6M: strh [[REG]],
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %0 = trunc i32 %i to i8
  call void @llvm.memset.p0i8.i32(i8* align 2 %p, i8 %0, i32 4, i1 false)
  call void @something(i8* %p)
  %inc = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %inc, 255
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare void @something(i8*) nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1) nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) nounwind
