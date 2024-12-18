; RUN: llc -mtriple=mips64el -mcpu=mips64r2 < %s

@.str = private unnamed_addr constant [7 x i8] c"hello\0A\00", align 1

define void @t(ptr %ptr) {
entry:
  tail call void @llvm.memcpy.p0.p0.i64(ptr %ptr, ptr @.str, i64 7, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind
