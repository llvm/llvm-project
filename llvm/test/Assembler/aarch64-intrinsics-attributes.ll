; REQUIRES: aarch64-registered-target

; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; Make sure some AArch64 intrinsics have the expected attributes.

; CHECK: declare i64 @llvm.aarch64.ldxr.p0i64(i64*) [[NOFREE_NOUNWIND_WILLRETURN:#[0-9]+]]
declare i64 @llvm.aarch64.ldxr.p0i64(i64*)

; CHECK: declare i32 @llvm.aarch64.stxp(i64, i64, i32*) [[NOFREE_NOUNWIND_WILLRETURN]]
declare i32 @llvm.aarch64.stxp(i64, i64, i32*)

; CHECK: declare i32 @llvm.aarch64.dsb(i32) [[NOFREE_NOUNWIND_WILLRETURN]]
declare i32 @llvm.aarch64.dsb(i32)

; CHECK: declare i64 @llvm.aarch64.neon.sqdmulls.scalar(i32, i32) [[NO_CALLBACK_NOFREE_NOSYNC_NOUNWIND_READNONE_WILLRETURN:#[0-9]+]]
declare i64 @llvm.aarch64.neon.sqdmulls.scalar(i32, i32)

; CHECK: declare <4 x i32> @llvm.aarch64.neon.shadd.v4i32(<4 x i32>, <4 x i32>) [[NO_CALLBACK_NOFREE_NOSYNC_NOUNWIND_READNONE_WILLRETURN]]
declare <4 x i32> @llvm.aarch64.neon.shadd.v4i32(<4 x i32>, <4 x i32>)

; CHECK: declare <vscale x 4 x i32> @llvm.aarch64.sve.dup.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, i32) [[NO_CALLBACK_NOFREE_NOSYNC_NOUNWIND_READNONE_WILLRETURN]]
declare <vscale x 4 x i32> @llvm.aarch64.sve.dup.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, i32)

; CHECK: attributes [[NOFREE_NOUNWIND_WILLRETURN]] = { nofree nounwind willreturn }
; CHECK: attributes [[NO_CALLBACK_NOFREE_NOSYNC_NOUNWIND_READNONE_WILLRETURN]] = { nocallback nofree nosync nounwind readnone willreturn }
