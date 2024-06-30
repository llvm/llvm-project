; RUN: opt < %s -O0 -S | FileCheck --check-prefixes=CHECK %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@0 = internal constant { i32, i32 } { i32 trunc (i64 sub (i64 ptrtoint (ptr @craSH to i64), i64 ptrtoint (ptr getelementptr inbounds ({ i32, i32 }, ptr @0, i32 0, i32 1) to i64)) to i32), i32 64 }

define dso_local void @af_suspend_fn(ptr %0, i64 %1, ptr %2) #0 {
  ret void
}

; Make sure that this test case does not crash and produce a split function.
; CHECK: craSH.resume.0

define dso_local void @craSH(ptr %0) #0 {
  %2 = call token @llvm.coro.id.async(i32 64, i32 8, i32 0, ptr @0)
  %3 = call ptr @llvm.coro.begin(token %2, ptr null)
  %4 = getelementptr inbounds { ptr, { ptr, ptr }, i64, { ptr, i1 }, i64, i64 }, ptr poison, i32 0, i32 0
  %5 = call ptr @llvm.coro.async.resume()
  store ptr %5, ptr %4, align 8
  %6 = call { ptr, ptr, ptr } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0p0p0s(i32 0, ptr %5, ptr @ctxt_proj_fn, ptr @af_suspend_fn, ptr poison, i64 -1, ptr poison)
  ret void
}

define dso_local ptr @ctxt_proj_fn(ptr %0) #0 {
  ret ptr %0
}

declare { ptr, ptr, ptr } @llvm.coro.suspend.async.sl_p0p0p0s(i32, ptr, ptr, ...) #1

declare token @llvm.coro.id.async(i32, i32, i32, ptr) #2

declare ptr @llvm.coro.begin(token, ptr writeonly) #2

declare ptr @llvm.coro.async.resume() #1

attributes #0 = { "target-features"="+adx,+aes,+avx,+avx2,+bmi,+bmi2,+clflushopt,+clwb,+clzero,+crc32,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+invpcid,+lzcnt,+mmx,+movbe,+mwaitx,+pclmul,+pku,+popcnt,+prfchw,+rdpid,+rdpru,+rdrnd,+rdseed,+sahf,+sha,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+sse4a,+ssse3,+vaes,+vpclmulqdq,+wbnoinvd,+x87,+xsave,+xsavec,+xsaveopt,+xsaves" }
attributes #1 = { nomerge nounwind }
attributes #2 = { nounwind }
