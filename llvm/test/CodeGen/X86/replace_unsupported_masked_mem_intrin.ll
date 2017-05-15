; RUN: llc -O0 -mtriple=x86_64-unknown-linux-gnu -mattr=+sse,+sse2 < %s -o /dev/null
; pr33001 - Check that llc doesn't crash when running with O0 option.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define <4 x i32> @test_masked_load(<4 x i32>* %base, <4 x i1> %mask) {
  %res = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %base, i32 4, <4 x i1> %mask, <4 x i32> zeroinitializer)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>*, i32, <4 x i1>, <4 x i32>)


define void @test_masked_store(<4 x i32>* %base, <4 x i32> %value, <4 x i1> %mask) {
  call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %value, <4 x i32>* %base, i32 4, <4 x i1> %mask)
  ret void
}

declare void @llvm.masked.store.v4i32.p0v4i32(<4 x i32>, <4 x i32>*, i32, <4 x i1>)


define <4 x i32> @llvm_masked_gather(<4 x i32*> %ptrs, <4 x i1> %mask) {
  %res = call <4 x i32> @llvm.masked.gather.v4i32(<4 x i32*> %ptrs, i32 4, <4 x i1> %mask, <4 x i32> undef)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.masked.gather.v4i32(<4 x i32*>, i32, <4 x i1>, <4 x i32>)


define void @llvm_masked_scatter(<4 x i32*> %ptrs, <4 x i32> %value, <4 x i1> %mask) {
  call void @llvm.masked.scatter.v4i32(<4 x i32> %value, <4 x i32*> %ptrs, i32 4, <4 x i1> %mask)
  ret void
}

declare void @llvm.masked.scatter.v4i32(<4 x i32>, <4 x i32*>, i32, <4 x i1>)

