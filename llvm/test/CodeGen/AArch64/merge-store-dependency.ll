; RUN: llc < %s -mcpu cortex-a53 -mtriple=aarch64-eabi | FileCheck %s --check-prefix=A53

; PR26827 - Merge stores causes wrong dependency.
%struct1 = type { %struct1*, %struct1*, i32, i32, i16, i16, void (i32, i32, i8*)*, i8* }
@gv0 = internal unnamed_addr global i32 0, align 4
@gv1 = internal unnamed_addr global %struct1** null, align 8

define void @test(%struct1* %fde, i32 %fd, void (i32, i32, i8*)* %func, i8* %arg)  {
;CHECK-LABEL: test
entry:
; A53: str q{{[0-9]+}}, {{.*}}
; A53: str q{{[0-9]+}}, {{.*}}
; A53: str w1, {{.*}}

  %0 = bitcast %struct1* %fde to i8*
  tail call void @llvm.memset.p0i8.i64(i8* %0, i8 0, i64 40, i32 8, i1 false)
  %state = getelementptr inbounds %struct1, %struct1* %fde, i64 0, i32 4
  store i16 256, i16* %state, align 8
  %fd1 = getelementptr inbounds %struct1, %struct1* %fde, i64 0, i32 2
  store i32 %fd, i32* %fd1, align 8
  %force_eof = getelementptr inbounds %struct1, %struct1* %fde, i64 0, i32 3
  store i32 0, i32* %force_eof, align 4
  %func2 = getelementptr inbounds %struct1, %struct1* %fde, i64 0, i32 6
  store void (i32, i32, i8*)* %func, void (i32, i32, i8*)** %func2, align 8
  %arg3 = getelementptr inbounds %struct1, %struct1* %fde, i64 0, i32 7
  store i8* %arg, i8** %arg3, align 8
  %call = tail call i32 (i32, i32, ...) @fcntl(i32 %fd, i32 4, i8* %0) #6
  %1 = load i32, i32* %fd1, align 8
  %cmp.i = icmp slt i32 %1, 0
  br i1 %cmp.i, label %if.then.i, label %while.body.i.preheader
if.then.i:
  unreachable

while.body.i.preheader:
  %2 = load i32, i32* @gv0, align 4
  %3 = icmp eq i32* %fd1, @gv0
  br i1 %3, label %while.body.i.split, label %while.body.i.split.ver.us.preheader

while.body.i.split.ver.us.preheader:
  br label %while.body.i.split.ver.us

while.body.i.split.ver.us:
  %.reg2mem21.0 = phi i32 [ %mul.i.ver.us, %while.body.i.split.ver.us ], [ %2, %while.body.i.split.ver.us.preheader ]
  %mul.i.ver.us = shl nsw i32 %.reg2mem21.0, 1
  %4 = icmp sgt i32 %mul.i.ver.us, %1
  br i1 %4, label %while.end.i, label %while.body.i.split.ver.us

while.body.i.split:
  br label %while.body.i.split

while.end.i:
  %call.i = tail call i8* @foo()
  store i8* %call.i, i8** bitcast (%struct1*** @gv1 to i8**), align 8
  br label %exit

exit:
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1)
declare i32 @fcntl(i32, i32, ...)
declare noalias i8* @foo()
