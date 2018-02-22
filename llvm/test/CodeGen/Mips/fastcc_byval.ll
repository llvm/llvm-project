; RUN: llc -mtriple=mipsel-linux-gnu -O3 -relocation-model=pic < %s | FileCheck %s

; Test that a load comes after a store to the same memory location when passing
; a byVal parameter to a function which has a fastcc function call

%struct.str = type { i32, i32, [3 x i32*] }

declare fastcc void @_Z1F3str(%struct.str* noalias nocapture sret %agg.result, %struct.str* byval nocapture readonly align 4 %s)

define i32 @_Z1g3str(%struct.str* byval nocapture readonly align 4 %s) {
; CHECK-LABEL: _Z1g3str:
; CHECK: sw  $7, [[OFFSET:[0-9]+]]($sp)
; CHECK: lw  ${{[0-9]+}}, [[OFFSET]]($sp)
entry:
  %ref.tmp = alloca %struct.str, align 4
  %0 = bitcast %struct.str* %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 20, i8* nonnull %0)
  call fastcc void @_Z1F3str(%struct.str* nonnull sret %ref.tmp, %struct.str* byval nonnull align 4 %s)
  %cl.sroa.3.0..sroa_idx2 = getelementptr inbounds %struct.str, %struct.str* %ref.tmp, i32 0, i32 1
  %cl.sroa.3.0.copyload = load i32, i32* %cl.sroa.3.0..sroa_idx2, align 4
  call void @llvm.lifetime.end.p0i8(i64 20, i8* nonnull %0)
  ret i32 %cl.sroa.3.0.copyload
}

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)

declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)
