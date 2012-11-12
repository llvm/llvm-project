; RUN: llc -march=mips64el -mcpu=mips64r2 -mattr=n64 -O3 < %s | FileCheck %s

%struct.S = type { [8 x i32] }

@g = common global %struct.S zeroinitializer, align 4

define void @f(%struct.S* noalias sret %agg.result) nounwind {
entry:
; CHECK: daddu $2, $zero, $4

  %0 = bitcast %struct.S* %agg.result to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* bitcast (%struct.S* @g to i8*), i64 32, i32 4, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind
