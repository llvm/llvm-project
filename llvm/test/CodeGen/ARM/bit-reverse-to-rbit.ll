;RUN: opt -instcombine -S < %s | llc -mtriple=armv5e--linux-gnueabi | FileCheck %s
;RUN: opt -instcombine -S < %s | llc -mtriple=thumbv4t--linux-gnueabi | FileCheck %s
;RUN: opt -instcombine -S < %s | llc -mtriple=armv6--linux-gnueabi | FileCheck %s

;RUN: opt -instcombine -S < %s | llc -mtriple=armv7--linux-gnueabi | FileCheck %s --check-prefix=RBIT
;RUN: opt -instcombine -S < %s | llc -mtriple=thumbv8--linux-gnueabi | FileCheck %s --check-prefix=RBIT

;CHECK-NOT: rbit
;RBIT: rbit

define void @byte_reversal(i8* %p, i32 %n) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp ult i32 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %0 = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i8, i8* %p, i64 %0
  %1 = load i8, i8* %arrayidx, align 1
  %or19 = call i8 @llvm.bitreverse.i8(i8 %1)
  store i8 %or19, i8* %arrayidx, align 1
  %inc = add i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; Function Attrs: nounwind readnone
declare i8 @llvm.bitreverse.i8(i8)
