; RUN: llc -mtriple=mipsel-linux-gnu -O3 -relocation-model=pic < %s | FileCheck %s

; Test that a load comes after a store to the same memory location when passing
; a byVal parameter to a function which has a fastcc function call

%struct.str = type { i32, i32, [3 x ptr] }

declare fastcc void @_Z1F3str(ptr noalias nocapture sret(%struct.str) %agg.result, ptr byval(%struct.str) nocapture readonly align 4 %s)

define i32 @_Z1g3str(ptr byval(%struct.str) nocapture readonly align 4 %s) {
; CHECK-LABEL: _Z1g3str:
; CHECK: sw  $7, [[OFFSET:[0-9]+]]($sp)
; CHECK: lw  ${{[0-9]+}}, [[OFFSET]]($sp)
entry:
  %ref.tmp = alloca %struct.str, align 4
  call void @llvm.lifetime.start.p0(i64 20, ptr nonnull %ref.tmp)
  call fastcc void @_Z1F3str(ptr nonnull sret(%struct.str) %ref.tmp, ptr byval(%struct.str) nonnull align 4 %s)
  %cl.sroa.3.0..sroa_idx2 = getelementptr inbounds %struct.str, ptr %ref.tmp, i32 0, i32 1
  %cl.sroa.3.0.copyload = load i32, ptr %cl.sroa.3.0..sroa_idx2, align 4
  call void @llvm.lifetime.end.p0(i64 20, ptr nonnull %ref.tmp)
  ret i32 %cl.sroa.3.0.copyload
}

declare void @llvm.lifetime.start.p0(i64, ptr nocapture)

declare void @llvm.lifetime.end.p0(i64, ptr nocapture)
