; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector store intrinsic instructions
;;;
;;; Note:
;;;   We test VST*rrvl, VST*rrvml, VST*irvl, and VST*irvml instructions.

; Function Attrs: nounwind
define void @vst_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vst_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vld.vssl(i64, ptr, i32)

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vst.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vst_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vst_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vst %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vst.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vst.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vst_vssl_imm(ptr %0) {
; CHECK-LABEL: vst_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vst_vssml_imm(ptr %0) {
; CHECK-LABEL: vst_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vst %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vst.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstnc_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vstnc_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vst.nc %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstnc.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstnc.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vstnc_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vstnc_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vst.nc %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstnc.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstnc.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vstnc_vssl_imm(ptr %0) {
; CHECK-LABEL: vstnc_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vst.nc %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstnc.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstnc_vssml_imm(ptr %0) {
; CHECK-LABEL: vstnc_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vst.nc %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstnc.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstot_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vstot_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vst.ot %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstot.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstot.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vstot_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vstot_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vst.ot %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstot.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstot.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vstot_vssl_imm(ptr %0) {
; CHECK-LABEL: vstot_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vst.ot %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstot.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstot_vssml_imm(ptr %0) {
; CHECK-LABEL: vstot_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vst.ot %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstot.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstncot_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vstncot_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vst.nc.ot %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstncot.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstncot.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vstncot_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vstncot_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vst.nc.ot %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstncot.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstncot.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vstncot_vssl_imm(ptr %0) {
; CHECK-LABEL: vstncot_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vst.nc.ot %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstncot.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstncot_vssml_imm(ptr %0) {
; CHECK-LABEL: vstncot_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vst.nc.ot %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstncot.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstu_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vstu_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstu %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstu.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstu.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vstu_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vstu_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstu %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstu.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstu.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vstu_vssl_imm(ptr %0) {
; CHECK-LABEL: vstu_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstu %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstu.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstu_vssml_imm(ptr %0) {
; CHECK-LABEL: vstu_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstu %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstu.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstunc_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vstunc_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstu.nc %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstunc.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstunc.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vstunc_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vstunc_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstu.nc %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstunc.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstunc.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vstunc_vssl_imm(ptr %0) {
; CHECK-LABEL: vstunc_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstu.nc %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstunc.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstunc_vssml_imm(ptr %0) {
; CHECK-LABEL: vstunc_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstu.nc %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstunc.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstuot_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vstuot_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstu.ot %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstuot.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstuot.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vstuot_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vstuot_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstu.ot %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstuot.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstuot.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vstuot_vssl_imm(ptr %0) {
; CHECK-LABEL: vstuot_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstu.ot %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstuot.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstuot_vssml_imm(ptr %0) {
; CHECK-LABEL: vstuot_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstu.ot %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstuot.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstuncot_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vstuncot_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstu.nc.ot %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstuncot.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstuncot.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vstuncot_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vstuncot_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstu.nc.ot %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstuncot.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstuncot.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vstuncot_vssl_imm(ptr %0) {
; CHECK-LABEL: vstuncot_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstu.nc.ot %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstuncot.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstuncot_vssml_imm(ptr %0) {
; CHECK-LABEL: vstuncot_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstu.nc.ot %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstuncot.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstl_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vstl_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstl %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstl.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstl.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vstl_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vstl_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstl %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstl.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstl.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vstl_vssl_imm(ptr %0) {
; CHECK-LABEL: vstl_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstl %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstl.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstl_vssml_imm(ptr %0) {
; CHECK-LABEL: vstl_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstl %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstl.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstlnc_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vstlnc_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstl.nc %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstlnc.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstlnc.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vstlnc_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vstlnc_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstl.nc %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstlnc.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstlnc.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vstlnc_vssl_imm(ptr %0) {
; CHECK-LABEL: vstlnc_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstl.nc %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstlnc.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstlnc_vssml_imm(ptr %0) {
; CHECK-LABEL: vstlnc_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstl.nc %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstlnc.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstlot_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vstlot_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstl.ot %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstlot.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstlot.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vstlot_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vstlot_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstl.ot %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstlot.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstlot.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vstlot_vssl_imm(ptr %0) {
; CHECK-LABEL: vstlot_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstl.ot %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstlot.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstlot_vssml_imm(ptr %0) {
; CHECK-LABEL: vstlot_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstl.ot %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstlot.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstlncot_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vstlncot_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstl.nc.ot %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstlncot.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstlncot.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vstlncot_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vstlncot_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstl.nc.ot %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstlncot.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstlncot.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vstlncot_vssl_imm(ptr %0) {
; CHECK-LABEL: vstlncot_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstl.nc.ot %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstlncot.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstlncot_vssml_imm(ptr %0) {
; CHECK-LABEL: vstlncot_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstl.nc.ot %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstlncot.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vst2d_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vst2d_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vst2d %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vst2d.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vst2d.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vst2d_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vst2d_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vst2d %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vst2d.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vst2d.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vst2d_vssl_imm(ptr %0) {
; CHECK-LABEL: vst2d_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vst2d %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vst2d.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vst2d_vssml_imm(ptr %0) {
; CHECK-LABEL: vst2d_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vst2d %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vst2d.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vst2dnc_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vst2dnc_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vst2d.nc %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vst2dnc.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vst2dnc.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vst2dnc_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vst2dnc_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vst2d.nc %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vst2dnc.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vst2dnc.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vst2dnc_vssl_imm(ptr %0) {
; CHECK-LABEL: vst2dnc_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vst2d.nc %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vst2dnc.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vst2dnc_vssml_imm(ptr %0) {
; CHECK-LABEL: vst2dnc_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vst2d.nc %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vst2dnc.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vst2dot_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vst2dot_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vst2d.ot %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vst2dot.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vst2dot.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vst2dot_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vst2dot_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vst2d.ot %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vst2dot.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vst2dot.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vst2dot_vssl_imm(ptr %0) {
; CHECK-LABEL: vst2dot_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vst2d.ot %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vst2dot.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vst2dot_vssml_imm(ptr %0) {
; CHECK-LABEL: vst2dot_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vst2d.ot %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vst2dot.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vst2dncot_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vst2dncot_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vst2d.nc.ot %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vst2dncot.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vst2dncot.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vst2dncot_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vst2dncot_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vst2d.nc.ot %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vst2dncot.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vst2dncot.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vst2dncot_vssl_imm(ptr %0) {
; CHECK-LABEL: vst2dncot_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vst2d.nc.ot %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vst2dncot.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vst2dncot_vssml_imm(ptr %0) {
; CHECK-LABEL: vst2dncot_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vst2d.nc.ot %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vst2dncot.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstu2d_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vstu2d_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstu2d %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstu2d.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstu2d.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vstu2d_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vstu2d_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstu2d %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstu2d.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstu2d.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vstu2d_vssl_imm(ptr %0) {
; CHECK-LABEL: vstu2d_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstu2d %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstu2d.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstu2d_vssml_imm(ptr %0) {
; CHECK-LABEL: vstu2d_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstu2d %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstu2d.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstu2dnc_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vstu2dnc_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstu2d.nc %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstu2dnc.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstu2dnc.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vstu2dnc_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vstu2dnc_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstu2d.nc %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstu2dnc.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstu2dnc.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vstu2dnc_vssl_imm(ptr %0) {
; CHECK-LABEL: vstu2dnc_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstu2d.nc %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstu2dnc.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstu2dnc_vssml_imm(ptr %0) {
; CHECK-LABEL: vstu2dnc_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstu2d.nc %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstu2dnc.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstu2dot_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vstu2dot_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstu2d.ot %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstu2dot.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstu2dot.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vstu2dot_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vstu2dot_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstu2d.ot %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstu2dot.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstu2dot.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vstu2dot_vssl_imm(ptr %0) {
; CHECK-LABEL: vstu2dot_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstu2d.ot %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstu2dot.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstu2dot_vssml_imm(ptr %0) {
; CHECK-LABEL: vstu2dot_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstu2d.ot %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstu2dot.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstu2dncot_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vstu2dncot_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstu2d.nc.ot %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstu2dncot.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstu2dncot.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vstu2dncot_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vstu2dncot_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstu2d.nc.ot %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstu2dncot.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstu2dncot.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vstu2dncot_vssl_imm(ptr %0) {
; CHECK-LABEL: vstu2dncot_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstu2d.nc.ot %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstu2dncot.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstu2dncot_vssml_imm(ptr %0) {
; CHECK-LABEL: vstu2dncot_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstu2d.nc.ot %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstu2dncot.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstl2d_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vstl2d_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstl2d %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstl2d.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstl2d.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vstl2d_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vstl2d_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstl2d %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstl2d.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstl2d.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vstl2d_vssl_imm(ptr %0) {
; CHECK-LABEL: vstl2d_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstl2d %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstl2d.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstl2d_vssml_imm(ptr %0) {
; CHECK-LABEL: vstl2d_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstl2d %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstl2d.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstl2dnc_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vstl2dnc_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstl2d.nc %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstl2dnc.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstl2dnc.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vstl2dnc_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vstl2dnc_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstl2d.nc %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstl2dnc.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstl2dnc.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vstl2dnc_vssl_imm(ptr %0) {
; CHECK-LABEL: vstl2dnc_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstl2d.nc %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstl2dnc.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstl2dnc_vssml_imm(ptr %0) {
; CHECK-LABEL: vstl2dnc_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstl2d.nc %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstl2dnc.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstl2dot_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vstl2dot_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstl2d.ot %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstl2dot.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstl2dot.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vstl2dot_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vstl2dot_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstl2d.ot %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstl2dot.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstl2dot.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vstl2dot_vssl_imm(ptr %0) {
; CHECK-LABEL: vstl2dot_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstl2d.ot %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstl2dot.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstl2dot_vssml_imm(ptr %0) {
; CHECK-LABEL: vstl2dot_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstl2d.ot %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstl2dot.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstl2dncot_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: vstl2dncot_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstl2d.nc.ot %v0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstl2dncot.vssl(<256 x double> %3, i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstl2dncot.vssl(<256 x double>, i64, ptr, i32)

; Function Attrs: nounwind
define void @vstl2dncot_vssml(ptr %0, i64 %1) {
; CHECK-LABEL: vstl2dncot_vssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    vstl2d.nc.ot %v0, %s1, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstl2dncot.vssml(<256 x double> %3, i64 %1, ptr %0, <256 x i1> undef, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vstl2dncot.vssml(<256 x double>, i64, ptr, <256 x i1>, i32)

; Function Attrs: nounwind
define void @vstl2dncot_vssl_imm(ptr %0) {
; CHECK-LABEL: vstl2dncot_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstl2d.nc.ot %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstl2dncot.vssl(<256 x double> %2, i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vstl2dncot_vssml_imm(ptr %0) {
; CHECK-LABEL: vstl2dncot_vssml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vstl2d.nc.ot %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, ptr %0, i32 256)
  tail call void @llvm.ve.vl.vstl2dncot.vssml(<256 x double> %2, i64 8, ptr %0, <256 x i1> undef, i32 256)
  ret void
}
