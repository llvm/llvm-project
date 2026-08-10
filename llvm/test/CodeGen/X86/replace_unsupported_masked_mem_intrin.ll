; RUN: llc -O0 -mtriple=x86_64-unknown-linux-gnu -mattr=+sse,+sse2 < %s -o /dev/null
; pr33001 - Check that llc doesn't crash when running with O0 option.

; RUN: llc -O2 -opt-bisect-limit=0 -mtriple=x86_64-unknown-linux-gnu -mattr=+sse,+sse2 < %s -o /dev/null
; Check that llc doesn't crash due to ScalarizeMaskedMemIntring not being run
; because of opt-bisect-limit that in turn causes crash in instruction selection
; for unsupported gather/scatter.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define <4 x i32> @test_masked_load(ptr %base, <4 x i1> %mask) {
  %res = call <4 x i32> @llvm.masked.load.v4i32.p0(ptr %base, i32 4, <4 x i1> %mask, <4 x i32> zeroinitializer)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.masked.load.v4i32.p0(ptr, i32, <4 x i1>, <4 x i32>)


define void @test_masked_store(ptr %base, <4 x i32> %value, <4 x i1> %mask) {
  call void @llvm.masked.store.v4i32.p0(<4 x i32> %value, ptr %base, i32 4, <4 x i1> %mask)
  ret void
}

declare void @llvm.masked.store.v4i32.p0(<4 x i32>, ptr, i32, <4 x i1>)


define <4 x i32> @llvm_masked_gather(<4 x ptr> %ptrs, <4 x i1> %mask) {
  %res = call <4 x i32> @llvm.masked.gather.v4i32(<4 x ptr> %ptrs, i32 4, <4 x i1> %mask, <4 x i32> undef)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.masked.gather.v4i32(<4 x ptr>, i32, <4 x i1>, <4 x i32>)


define void @llvm_masked_scatter(<4 x ptr> %ptrs, <4 x i32> %value, <4 x i1> %mask) {
  call void @llvm.masked.scatter.v4i32(<4 x i32> %value, <4 x ptr> %ptrs, i32 4, <4 x i1> %mask)
  ret void
}

declare void @llvm.masked.scatter.v4i32(<4 x i32>, <4 x ptr>, i32, <4 x i1>)

