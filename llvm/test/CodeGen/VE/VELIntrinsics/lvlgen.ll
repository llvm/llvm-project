; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

; Test for correct placement of 'lvl' instructions

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vld.vssl(i64, i8*, i32)
declare void @llvm.ve.vl.vst.vssl(<256 x double>, i64, i8*, i32)

; Check that the backend can handle constant VL as well as parametric VL
; sources.

; Function Attrs: nounwind
define void @switching_vl(i32 %evl, i32 %evl2, i8* %P, i8* %Q) {
; CHECK-LABEL: switching_vl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s4, 256
; CHECK-NEXT:    lvl %s4
; CHECK-NEXT:    vld %v0, 8, %s2
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vst %v0, 16, %s3
; CHECK-NEXT:    lea %s4, 128
; CHECK-NEXT:    lvl %s4
; CHECK-NEXT:    vld %v0, 16, %s2
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vst %v0, 16, %s3
; CHECK-NEXT:    lvl %s4
; CHECK-NEXT:    vld %v0, 8, %s2
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vst %v0, 16, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %l0 = tail call <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %P, i32 256)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %l0, i64 16, i8* %Q, i32 %evl)
  %l1 = tail call <256 x double> @llvm.ve.vl.vld.vssl(i64 16, i8* %P, i32 128)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %l1, i64 16, i8* %Q, i32 %evl2)
  %l2 = tail call <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %P, i32 128)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %l2, i64 16, i8* %Q, i32 %evl)
  ret void
}

; Check that no redundant 'lvl' is inserted when vector length does not change
; in a basic block.

; Function Attrs: nounwind
define void @stable_vl(i32 %evl, i8* %P, i8* %Q) {
; CHECK-LABEL: stable_vl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vld %v0, 8, %s1
; CHECK-NEXT:    vst %v0, 16, %s2
; CHECK-NEXT:    vld %v0, 16, %s1
; CHECK-NEXT:    vst %v0, 16, %s2
; CHECK-NEXT:    vld %v0, 8, %s1
; CHECK-NEXT:    vst %v0, 16, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %l0 = tail call <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %P, i32 %evl)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %l0, i64 16, i8* %Q, i32 %evl)
  %l1 = tail call <256 x double> @llvm.ve.vl.vld.vssl(i64 16, i8* %P, i32 %evl)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %l1, i64 16, i8* %Q, i32 %evl)
  %l2 = tail call <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %P, i32 %evl)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %l2, i64 16, i8* %Q, i32 %evl)
  ret void
}

;;; Check the case we have a call in the middle of vector instructions.

; Function Attrs: nounwind
define void @call_invl(i32 %evl, i8* %P, i8* %Q) {
; CHECK-LABEL: call_invl:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st %s18, 288(, %s11) # 8-byte Folded Spill
; CHECK-NEXT:    st %s19, 296(, %s11) # 8-byte Folded Spill
; CHECK-NEXT:    st %s20, 304(, %s11) # 8-byte Folded Spill
; CHECK-NEXT:    or %s18, 0, %s1
; CHECK-NEXT:    and %s20, %s0, (32)0
; CHECK-NEXT:    lvl %s20
; CHECK-NEXT:    vld %v0, 8, %s1
; CHECK-NEXT:    or %s19, 0, %s2
; CHECK-NEXT:    vst %v0, 16, %s2
; CHECK-NEXT:    lea %s0, fun@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, fun@hi(, %s0)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    lvl %s20
; CHECK-NEXT:    vld %v0, 16, %s18
; CHECK-NEXT:    vst %v0, 16, %s19
; CHECK-NEXT:    vld %v0, 8, %s18
; CHECK-NEXT:    vst %v0, 16, %s19
; CHECK-NEXT:    ld %s20, 304(, %s11) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s19, 296(, %s11) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s18, 288(, %s11) # 8-byte Folded Reload
; CHECK-NEXT:    or %s11, 0, %s9
  %l0 = tail call <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %P, i32 %evl)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %l0, i64 16, i8* %Q, i32 %evl)
  call void @fun()
  %l1 = tail call <256 x double> @llvm.ve.vl.vld.vssl(i64 16, i8* %P, i32 %evl)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %l1, i64 16, i8* %Q, i32 %evl)
  %l2 = tail call <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %P, i32 %evl)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %l2, i64 16, i8* %Q, i32 %evl)
  ret void
}

declare void @fun()
