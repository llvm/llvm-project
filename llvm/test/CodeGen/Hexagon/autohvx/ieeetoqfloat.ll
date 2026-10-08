; RUN: llc -march=hexagon -mattr=+hvx-ieee-fp,+hvx-length128b,+hvxv79 < %s \
; RUN: | FileCheck %s --enable-var-scope
; RUN: llc -march=hexagon -mattr=+hvx-ieee-fp,+hvx-length128b,+hvxv81 < %s \
; RUN: | FileCheck %s --enable-var-scope
; RUN: llc -march=hexagon -mattr=+hvx-ieee-fp,+hvx-length128b,+hvxv75 < %s \
; RUN: | FileCheck %s --check-prefix=V75 --enable-var-scope

;On v79 and above,
;v0.hf = vabs(v0.hf) is translated to
;r2 = #32767 ; v1.h = vsplat(r2) ; v0 = vand(v0,v1)
;Reset the sign bit by splatting 0x7FFF and does a vector and.

; CHECK-LABEL: vec_abs_hf:
; CHECK-NOT: v{{.*}}.hf = vabs(v{{.*}}.hf)
; #0x7FFF
; CHECK: r[[REG0:[0-9]+]] = #32767
; CHECK: v[[REG1:[0-9]+]].h = vsplat(r[[REG0]])
; CHECK: v{{.*}} = vand({{.*}}[[REG1]]

; V75-LABEL: vec_abs_hf:
; V75: v{{.*}}.hf = vabs(v{{.*}}.hf)

define dso_local inreg <32 x i32> @vec_abs_hf(<32 x i32> noundef %a) local_unnamed_addr {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vabs.hf.128B(<32 x i32> %a)
  ret <32 x i32> %0
}

;On v79 and above,
;v0.sf = vabs(v0.sf) is translated to
;r2 = ##2147483647 ; v1 = vsplat(r2) ; v0 = vand(v0,v1)
;Reset the sign bit by splatting 0x7FFF FFFF and does a vector and.

; CHECK-LABEL: vec_abs_sf:
; CHECK-NOT: v{{.*}}.sf = vabs(v{{.*}}.sf)
; #0x7FFF FFFF
; CHECK: r[[REG0:[0-9]+]] = ##2147483647
; CHECK: v[[REG1:[0-9]+]] = vsplat(r[[REG0]])
; CHECK: v{{.*}} = vand({{.*}}[[REG1]]

; V75-LABEL: vec_abs_sf:
; V75: v{{.*}}.sf = vabs(v{{.*}}.sf)

define dso_local inreg <32 x i32> @vec_abs_sf(<32 x i32> noundef %a) local_unnamed_addr {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vabs.sf.128B(<32 x i32> %a)
  ret <32 x i32> %0
}

;On v79 and above,
;v0.w = vfmv(v1.w) is translated to v0 = v1.

; CHECK-LABEL: vec_assign:
; CHECK-NOT: v{{.*}}.w = vfmv(v{{.*}}.w)
; CHECK: v[[REG0:[0-9]+]] = v[[REG0]]

; V75-LABEL: vec_assign:
; V75: v{{.*}}.w = vfmv(v{{.*}}.w)

define dso_local inreg <32 x i32> @vec_assign(<32 x i32> noundef %a) local_unnamed_addr {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vassign.fp.128B(<32 x i32> %a)
  ret <32 x i32> %0
}

;On v79 and above,
;v0.hf = vadd(v0.hf,v1.hf) is translated to
;v2.qf16 = vadd(v0.hf,v1.hf); v0.hf = v2.qf16

; CHECK-LABEL: vec_add_hf:
; CHECK-NOT: v{{.*}}.hf = vadd(v{{.*}}.hf,v{{.*}}.hf)
; CHECK: v[[REG0:[0-9]+]].qf16 = vadd(v{{.*}}.hf,v{{.*}}.hf)
; CHECK: v0.hf = v[[REG0]].qf16

; V75-LABEL: vec_add_hf:
; V75: v{{.*}}.hf = vadd(v{{.*}}.hf,v{{.*}}.hf)

define dso_local inreg <32 x i32> @vec_add_hf(<32 x i32> noundef %a, <32 x i32> noundef %b) local_unnamed_addr {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vadd.hf.hf.128B(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

;On v79 and above,
;v0.sf = vadd(v0.sf,v1.sf) is translated to
;v2.qf32 = vadd(v0.sf,v1.sf); v0.sf = v2.qf32

; CHECK-LABEL: vec_add_sf:
; CHECK-NOT: v{{.*}}.sf = vadd(v{{.*}}.sf,v{{.*}}.sf)
; CHECK: v[[REG0:[0-9]+]].qf32 = vadd(v{{.*}}.sf,v{{.*}}.sf)
; CHECK: v0.sf = v[[REG0]].qf32

; V75-LABEL: vec_add_sf:
; V75: v{{.*}}.sf = vadd(v{{.*}}.sf,v{{.*}}.sf)

define dso_local inreg <32 x i32> @vec_add_sf(<32 x i32> noundef %a, <32 x i32> noundef %b) local_unnamed_addr {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.sf.128B(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

;On v79 and above,
;v1:0.sf = vadd(v0.hf,v1.hf) is translated to
;r2 = #15360; v2.h = vsplat(r2);
;v5:4.qf32 = vmpy(v1.hf,v2.hf); v31:30.qf32 = vmpy(v0.hf,v2.hf)
;v4.qf32 = vadd(v30.qf32,v4.qf32); v3.qf32 = vadd(v31.qf32,v5.qf32)
;v0.sf = v4.qf32; v1.sf = v3.qf32
;Widen the hf operands to sf by doing a widening multiply with 1.0f
;and perform the add.

; CHECK-LABEL: sfaddhfhf:
; CHECK-NOT: v1:0.sf = vadd(v0.hf,v1.hf)
; CHECK: r[[REG0:[0-9]+]] = #15360
; CHECK: v[[REG1:[0-9]+]].h = vsplat(r[[REG0]])
; CHECK-DAG: v[[REG2:[0-9]+]]:[[REG3:[0-9]+]].qf32 = vmpy(v{{.*}}.hf,v[[REG1]].hf)
; CHECK-DAG: v[[REG4:[0-9]+]]:[[REG5:[0-9]+]].qf32 = vmpy(v{{.*}}.hf,v[[REG1]].hf)
; CHECK-DAG: v[[REG6:[0-9]+]].qf32 = vadd(v[[REG5]].qf32,v[[REG3]].qf32)
; CHECK-DAG: v[[REG7:[0-9]+]].qf32 = vadd(v[[REG4]].qf32,v[[REG2]].qf32)
; CHECK-DAG: v{{.*}}.sf = v[[REG6]].qf32
; CHECK-DAG: v{{.*}}.sf = v[[REG7]].qf32

; V75-LABEL: sfaddhfhf:
; CHECK-NOT: v1:0.sf = vadd(v0.hf,v1.hf)
; V75: v1:0.sf = vadd(v0.hf,v1.hf)

define dso_local inreg <64 x i32> @sfaddhfhf(<32 x i32> noundef %Vu, <32 x i32> noundef %Vv) local_unnamed_addr {
entry:
  %0 = tail call <64 x i32> @llvm.hexagon.V6.vadd.sf.hf.128B(<32 x i32> %Vu, <32 x i32> %Vv)
  ret <64 x i32> %0
}

;On v79 and above,
;v0.hf = vsub(v0.hf,v1.hf) is translated to
;v2.qf16 = vsub(v0.hf,v1.hf); v0.hf = v2.qf16

; CHECK-LABEL: vec_sub_hf:
; CHECK-NOT: v{{.*}}.hf = vsub(v{{.*}}.hf,v{{.*}}.hf)
; CHECK: v[[REG0:[0-9]+]].qf16 = vsub(v{{.*}}.hf,v{{.*}}.hf)
; CHECK: v0.hf = v[[REG0]].qf16

; V75-LABEL: vec_sub_hf:
; V75: v{{.*}}.hf = vsub(v{{.*}}.hf,v{{.*}}.hf)

define dso_local inreg <32 x i32> @vec_sub_hf(<32 x i32> noundef %a, <32 x i32> noundef %b) local_unnamed_addr {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vsub.hf.hf.128B(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

;On v79 and above,
;v0.sf = vsub(v0.sf,v1.sf) is translated to
;v2.qf32 = vsub(v0.sf,v1.sf); v0.sf = v2.qf32

; CHECK-LABEL: vec_sub_sf:
; CHECK-NOT: v{{.*}}.sf = vsub(v{{.*}}.sf,v{{.*}}.sf)
; CHECK: v[[REG0:[0-9]+]].qf32 = vsub(v{{.*}}.sf,v{{.*}}.sf)
; CHECK: v0.sf = v[[REG0]].qf32

; V75-LABEL: vec_sub_sf:
; V75: v{{.*}}.sf = vsub(v{{.*}}.sf,v{{.*}}.sf)

define dso_local inreg <32 x i32> @vec_sub_sf(<32 x i32> noundef %a, <32 x i32> noundef %b) local_unnamed_addr {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vsub.sf.sf.128B(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

;On v79 and above,
;v1:0.sf = vsub(v0.hf,v1.hf) is translated to
;r2 = #15360; v2.h = vsplat(r2);
;v5:4.qf32 = vmpy(v1.hf,v2.hf); v31:30.qf32 = vmpy(v0.hf,v2.hf)
;v4.qf32 = vsub(v30.qf32,v4.qf32); v3.qf32 = vsub(v31.qf32,v5.qf32)
;v0.sf = v4.qf32; v1.sf = v3.qf32
;Widen the hf operands to sf by doing a widening multiply with 1.0f
;and perform the sub.

; CHECK-LABEL: sfsubhfhf:
; CHECK-NOT: v1:0.sf = vsub(v0.hf,v1.hf)
; CHECK: r[[REG0:[0-9]+]] = #15360
; CHECK: v[[REG1:[0-9]+]].h = vsplat(r[[REG0]])
; CHECK-DAG: v[[REG2:[0-9]+]]:[[REG3:[0-9]+]].qf32 = vmpy(v{{.*}}.hf,v[[REG1]].hf)
; CHECK-DAG: v[[REG4:[0-9]+]]:[[REG5:[0-9]+]].qf32 = vmpy(v{{.*}}.hf,v[[REG1]].hf)
; CHECK-DAG: v[[REG6:[0-9]+]].qf32 = vsub(v[[REG5]].qf32,v[[REG3]].qf32)
; CHECK-DAG: v[[REG7:[0-9]+]].qf32 = vsub(v[[REG4]].qf32,v[[REG2]].qf32)
; CHECK-DAG: v{{.*}}.sf = v[[REG6]].qf32
; CHECK-DAG: v{{.*}}.sf = v[[REG7]].qf32

; V75-LABEL: sfsubhfhf:
; CHECK-NOT: v1:0.sf = vsub(v0.hf,v1.hf)
; V75: v1:0.sf = vsub(v0.hf,v1.hf)

define dso_local inreg <64 x i32> @sfsubhfhf(<32 x i32> noundef %Vu, <32 x i32> noundef %Vv) local_unnamed_addr {
entry:
  %0 = tail call <64 x i32> @llvm.hexagon.V6.vsub.sf.hf.128B(<32 x i32> %Vu, <32 x i32> %Vv)
  ret <64 x i32> %0
}

;On v79 and above,
;v0.hf = vmpy(v0.hf,v1.hf) is translated to
;v1:0.qf32 = vmpy(v0.hf,v1.hf); v0.hf = v1:0.qf32.

; CHECK-LABEL: vec_mpy_hf:
; CHECK-NOT: v{{.*}}.hf = vmpy(v{{.*}}.hf,v{{.*}}.hf)
; CHECK: v1:0.qf32 = vmpy(v{{.*}}.hf,v{{.*}}.hf)
; CHECK: v0.hf = v1:0.qf32

; V75-LABEL: vec_mpy_hf:
; V75: v{{.*}}.hf = vmpy(v{{.*}}.hf,v{{.*}}.hf)

define dso_local inreg <32 x i32> @vec_mpy_hf(<32 x i32> noundef %a, <32 x i32> noundef %b) local_unnamed_addr {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.hf.hf.128B(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

;On v79 and above,
;v0.sf = vmpy(v0.sf,v1.sf) is translated to
;v2.qf32 = vmpy(v0.sf,v1.sf); v0.sf = v2.qf32.

; CHECK-LABEL: vec_mpy_sf:
; CHECK-NOT: v{{.*}}.sf = vmpy(v{{.*}}.sf,v{{.*}}.sf)
; CHECK: v[[REG0:[0-9]+]].qf32 = vmpy(v{{.*}}.sf,v{{.*}}.sf)
; CHECK: v0.sf = v[[REG0]].qf32

; V75-LABEL: vec_mpy_sf:
; V75: v{{.*}}.sf = vmpy(v{{.*}}.sf,v{{.*}}.sf)

define dso_local inreg <32 x i32> @vec_mpy_sf(<32 x i32> noundef %a, <32 x i32> noundef %b) local_unnamed_addr {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.sf.sf.128B(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

;On v79 and above,
;v1:0.sf = vmpy(v0.hf,v1.hf) is translated to
;v3:2.qf32 = vmpy(v0.hf,v1.hf); v0.sf = v2.qf32 ; v1.sf = v3.qf32.

; CHECK-LABEL: sfmpyhf:
; CHECK-NOT: v1:0.sf = vmpy(v0.hf,v1.hf)
; CHECK: v[[REG3:[0-9]+]]:[[REG2:[0-9]+]].qf32 = vmpy(v0.hf,v1.hf)
; CHECK-DAG: v{{.*}}.sf = v[[REG2]].qf32
; CHECK-DAG: v{{.*}}.sf = v[[REG3]].qf32

; V75-LABEL: sfmpyhf:
; V75: v1:0.sf = vmpy(v0.hf,v1.hf)

define dso_local inreg <64 x i32> @sfmpyhf(<32 x i32> noundef %Vu, <32 x i32> noundef %Vv) local_unnamed_addr {
entry:
  %0 = tail call <64 x i32> @llvm.hexagon.V6.vmpy.sf.hf.128B(<32 x i32> %Vu, <32 x i32> %Vv)
  ret <64 x i32> %0
}

;On v79 and above,
;v0.hf += vmpy(v1.hf,v2.hf) is translated to
;v7:6.qf32 = vmpy(v1.hf, v2.hf)  // widening multiply
;V5:4.qf32 = vmpy(v0.hf,1.0) // Convert accum to qf32
;V4.qf32 = vadd(V6.qf32, V4.qf32)       // accumulation
;V5.qf32 = vadd(V7.qf32, V5.qf32)       // accumulation
;V4.hf = V5:4.qf32

; CHECK-LABEL: mpyhf_acc:
; CHECK-NOT: v{{.*}}.hf += vmpy(v{{.*}}.hf,v{{.*}}.hf)
; CHECK: v[[REG3:[0-9]+]]:[[REG2:[0-9]+]].qf32 = vmpy(v{{.*}}.hf,v{{.*}}.hf)
; CHECK: r{{.*}} = #15360
; CHECK: v[[REG4:[0-9]+]].h = vsplat(r{{.*}})
; CHECK: v[[REG7:[0-9]+]]:[[REG6:[0-9]+]].qf32 = vmpy(v{{.*}}.hf,v[[REG4]].hf)
; CHECK: v[[REG4:[0-9]+]].qf32 = vadd(v[[REG6]].qf32,v[[REG2]].qf32)
; CHECK: v[[REG5:[0-9]+]].qf32 = vadd(v[[REG7]].qf32,v[[REG3]].qf32)
; CHECK: v{{.*}}.hf = v[[REG5]]:[[REG4]].qf32

; V75-LABEL: mpyhf_acc:
; V75: v{{.*}}.hf += vmpy(v{{.*}}.hf,v{{.*}}.hf)

define dso_local inreg <32 x i32> @mpyhf_acc(<32 x i32> noundef %Vx, <32 x i32> noundef %Vu, <32 x i32> noundef %Vv) local_unnamed_addr {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.hf.hf.acc.128B(<32 x i32> %Vx, <32 x i32> %Vu, <32 x i32> %Vv)
  ret <32 x i32> %0
}

;On v79 and above,
;v1:0.sf += vmpy(v2.hf,v3.hf) is translated to
;v3:2.qf32 = vmpy(v2.hf,v3.hf);
;v0.qf32 = vadd(v2.qf32,v0.sf); v1.qf32 = vadd(v3.qf32,v1.sf);
;v0.sf = v0.qf32; v1.sf = v1.qf32

; CHECK-LABEL: sfmpyhf_acc:
; CHECK-NOT: v1:0.sf += vmpy(v2.hf,v3.hf)
; CHECK: v[[REG3:[0-9]+]]:[[REG2:[0-9]+]].qf32 = vmpy(v2.hf,v3.hf)
; CHECK-DAG: v[[REG0:[0-9]+]].qf32 = vadd(v[[REG2]].qf32,v0.sf)
; CHECK-DAG: v[[REG1:[0-9]+]].qf32 = vadd(v[[REG3]].qf32,v1.sf)
; CHECK-DAG: v{{.*}}.sf = v[[REG0]].qf32
; CHECK-DAG: v{{.*}}.sf = v[[REG1]].qf32

; V75-LABEL: sfmpyhf_acc:
; V75: v1:0.sf += vmpy(v2.hf,v3.hf)

define dso_local inreg <64 x i32> @sfmpyhf_acc(<64 x i32> noundef %Vxx, <32 x i32> noundef %Vu, <32 x i32> noundef %Vv) local_unnamed_addr {
entry:
  %0 = tail call <64 x i32> @llvm.hexagon.V6.vmpy.sf.hf.acc.128B(<64 x i32> %Vxx, <32 x i32> %Vu, <32 x i32> %Vv)
  ret <64 x i32> %0
}

;On v79 and above,
;v0.hf = vfmin(v0.hf,v1.hf) is translated to v0.hf = vmin(v0.hf,v1.hf)

; CHECK-LABEL: vec_min_hf:
; CHECK-NOT: v{{.*}}.hf = vfmin(v{{.*}}.hf,v{{.*}}.hf)
; CHECK: v{{.*}}.hf = vmin(v{{.*}}.hf,v{{.*}}.hf)

; V75-LABEL: vec_min_hf:
; V75: v{{.*}}.hf = vfmin(v{{.*}}.hf,v{{.*}}.hf)

define dso_local inreg <32 x i32> @vec_min_hf(<32 x i32> noundef %a, <32 x i32> noundef %b) local_unnamed_addr {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vfmin.hf.128B(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

;On v79 and above,
;v0.sf = vfmin(v0.sf,v1.sf) is translated to v0.sf = vmin(v0.sf,v1.sf)

; CHECK-LABEL: vec_min_sf:
; CHECK-NOT: v{{.*}}.sf = vfmin(v{{.*}}.sf,v{{.*}}.sf)
; CHECK: v{{.*}}.sf = vmin(v{{.*}}.sf,v{{.*}}.sf)

; V75-LABEL: vec_min_sf:
; V75: v{{.*}}.sf = vfmin(v{{.*}}.sf,v{{.*}}.sf)

define dso_local inreg <32 x i32> @vec_min_sf(<32 x i32> noundef %a, <32 x i32> noundef %b) local_unnamed_addr {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vfmin.sf.128B(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

;On v79 and above,
;v0.hf = vfmax(v0.hf,v1.hf) is translated to v0.hf = vmax(v0.hf,v1.hf)

; CHECK-LABEL: vec_max_hf:
; CHECK-NOT: v{{.*}}.hf = vfmax(v{{.*}}.hf,v{{.*}}.hf)
; CHECK: v{{.*}}.hf = vmax(v{{.*}}.hf,v{{.*}}.hf)

; V75-LABEL: vec_max_hf:
; V75: v{{.*}}.hf = vfmax(v{{.*}}.hf,v{{.*}}.hf)

define dso_local inreg <32 x i32> @vec_max_hf(<32 x i32> noundef %a, <32 x i32> noundef %b) local_unnamed_addr {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vfmax.hf.128B(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

;On v79 and above,
;v0.sf = vfmax(v0.sf,v1.sf) is translated to v0.sf = vmax(v0.sf,v1.sf)

; CHECK-LABEL: vec_max_sf:
; CHECK-NOT: v{{.*}}.sf = vfmax(v{{.*}}.sf,v{{.*}}.sf)
; CHECK: v{{.*}}.sf = vmax(v{{.*}}.sf,v{{.*}}.sf)

; V75-LABEL: vec_max_sf:
; V75: v{{.*}}.sf = vfmax(v{{.*}}.sf,v{{.*}}.sf)

define dso_local inreg <32 x i32> @vec_max_sf(<32 x i32> noundef %a, <32 x i32> noundef %b) local_unnamed_addr {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vfmax.sf.128B(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

;On v79 and above,
;v0.hf = vfneg(v0.hf) is translated to
;r2 = #32768 ; v1.h = vsplat(r2) ; v0 = vxor(v0,v1)
;Flip the sign bit by splatting 0x8000 and does a vector xor.

; CHECK-LABEL: vec_neg_hf:
; CHECK-NOT: v{{.*}}.hf = vfneg(v{{.*}}.hf)
; #0x8000
; CHECK: r[[REG0:[0-9]+]] = ##32768
; CHECK: v[[REG1:[0-9]+]].h = vsplat(r[[REG0]])
; CHECK: v{{.*}} = vxor({{.*}}[[REG1]]

; V75-LABEL: vec_neg_hf:
; V75: v{{.*}}.hf = vfneg(v{{.*}}.hf)

define dso_local inreg <32 x i32> @vec_neg_hf(<32 x i32> noundef %a) local_unnamed_addr {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vfneg.hf.128B(<32 x i32> %a)
  ret <32 x i32> %0
}

;On v79 and above,
;v0.sf = vfneg(v0.sf) is translated to
;r2 = ##-2147483648 ; v1 = vsplat(r2) ; v0 = vxor(v0,v1)
;Flip the sign bit by splatting 0x8000 0000 and does a vector xor.

; CHECK-LABEL: vec_neg_sf:
; CHECK-NOT: v{{.*}}.sf = vneg(v{{.*}}.sf,v{{.*}}.sf)
; #0x8000 0000
; CHECK: r[[REG0:[0-9]+]] = ##-2147483648
; CHECK: v[[REG1:[0-9]+]] = vsplat(r[[REG0]])
; CHECK: v{{.*}} = vxor({{.*}}[[REG1]]

; V75-LABEL: vec_neg_sf:
; V75: v{{.*}}.sf = vfneg(v{{.*}}.sf)

define dso_local inreg <32 x i32> @vec_neg_sf(<32 x i32> noundef %a) local_unnamed_addr {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vfneg.sf.128B(<32 x i32> %a)
  ret <32 x i32> %0
}

;On v79 and above,
;v0.hf = vcvt(v0.h) is translated to v0.hf = v0.h.

; CHECK-LABEL: convhfh:
; CHECK-NOT: v{{.*}}.hf = vcvt(v0.h)
; CHECK: v{{.*}}.hf = v0.h

; V75-LABEL: convhfh:
; V75: v{{.*}}.hf = vcvt(v0.h)

define dso_local inreg <32 x i32> @convhfh(<32 x i32> noundef %Vu) local_unnamed_addr {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vcvt.hf.h.128B(<32 x i32> %Vu)
  ret <32 x i32> %0
}

;On v79 and above,
;v1:0.sf = vcvt(v0.hf) is translated to
;r2 = #15360; v1.h = vsplat(r2); v3:2.qf32 = vmpy(v0.hf,v1.hf);
;v0.sf = v2.qf32; v1.sf = v3.qf32
;Do a widening multiply with 1.0f to convert the hf to sf.

; CHECK-LABEL: convsfhf:
; CHECK-NOT: v1:0.sf = vcvt(v0.hf)
; CHECK: r[[REG0:[0-9]+]] = #15360
; CHECK: v[[REG1:[0-9]+]].h = vsplat(r[[REG0]])
; CHECK: v[[REG3:[0-9]+]]:[[REG2:[0-9]+]].qf32 = vmpy(v0.hf,v1.hf)
; CHECK-DAG: v{{.*}}.sf = v[[REG2]].qf32
; CHECK-DAG: v{{.*}}.sf = v[[REG3]].qf32

; V75-LABEL: convsfhf:
; V75: v1:0.sf = vcvt(v0.hf)

define dso_local inreg <64 x i32> @convsfhf(<32 x i32> noundef %Vu) local_unnamed_addr {
entry:
  %0 = tail call <64 x i32> @llvm.hexagon.V6.vcvt.sf.hf.128B(<32 x i32> %Vu)
  ret <64 x i32> %0
}

declare <32 x i32> @llvm.hexagon.V6.vabs.hf.128B(<32 x i32>)
declare <32 x i32> @llvm.hexagon.V6.vabs.sf.128B(<32 x i32>)
declare <32 x i32> @llvm.hexagon.V6.vassign.fp.128B(<32 x i32>)
; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vadd.hf.hf.128B(<32 x i32>, <32 x i32>)
; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vadd.sf.sf.128B(<32 x i32>, <32 x i32>)
declare <64 x i32> @llvm.hexagon.V6.vadd.sf.hf.128B(<32 x i32>, <32 x i32>)
; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vsub.hf.hf.128B(<32 x i32>, <32 x i32>)
; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vsub.sf.sf.128B(<32 x i32>, <32 x i32>)
declare <64 x i32> @llvm.hexagon.V6.vsub.sf.hf.128B(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.hexagon.V6.vmpy.hf.hf.128B(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.hexagon.V6.vmpy.sf.sf.128B(<32 x i32>, <32 x i32>)
declare <64 x i32> @llvm.hexagon.V6.vmpy.sf.hf.128B(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.hexagon.V6.vmpy.hf.hf.acc.128B(<32 x i32>, <32 x i32>, <32 x i32>)
declare <64 x i32> @llvm.hexagon.V6.vmpy.sf.hf.acc.128B(<64 x i32>, <32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.hexagon.V6.vfmin.hf.128B(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.hexagon.V6.vfmin.sf.128B(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.hexagon.V6.vfmax.hf.128B(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.hexagon.V6.vfmax.sf.128B(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.hexagon.V6.vfneg.hf.128B(<32 x i32>)
declare <32 x i32> @llvm.hexagon.V6.vfneg.sf.128B(<32 x i32>)
declare <32 x i32> @llvm.hexagon.V6.vcvt.hf.h.128B(<32 x i32>)
declare <64 x i32> @llvm.hexagon.V6.vcvt.sf.hf.128B(<32 x i32>)
