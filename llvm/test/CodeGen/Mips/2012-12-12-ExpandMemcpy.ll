; RUN: llc -march=mips64el -mcpu=mips64r2 < %s

@.str = private unnamed_addr constant [7 x i8] c"hello\0A\00", align 1

define void @t(i8* %ptr) {
entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %ptr, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str, i64 0, i64 0), i64 7, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind
