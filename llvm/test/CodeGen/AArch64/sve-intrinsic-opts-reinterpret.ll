; RUN: opt -S -aarch64-sve-intrinsic-opts -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck --check-prefix OPT %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

define <vscale x 8 x i1> @reinterpret_test_h(<vscale x 8 x i1> %a) {
; OPT-LABEL: @reinterpret_test_h(
; OPT-NOT: convert
; OPT: ret <vscale x 8 x i1> %a
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %a)
  %2 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %1)
  ret <vscale x 8 x i1> %2
}

; Reinterprets are not redundant because the second reinterpret zeros the
; lanes that don't exist within its input.
define <vscale x 16 x i1> @reinterpret_test_h_rev(<vscale x 16 x i1> %a) {
; OPT-LABEL: @reinterpret_test_h_rev(
; OPT: %1 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %a)
; OPT-NEXT: %2 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %1)
; OPT-NEXT: ret <vscale x 16 x i1> %2
  %1 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %a)
  %2 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %1)
  ret <vscale x 16 x i1> %2
}

define <vscale x 4 x i1> @reinterpret_test_w(<vscale x 4 x i1> %a) {
; OPT-LABEL: @reinterpret_test_w(
; OPT-NOT: convert
; OPT: ret <vscale x 4 x i1> %a
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %a)
  %2 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %1)
  ret <vscale x 4 x i1> %2
}

; Reinterprets are not redundant because the second reinterpret zeros the
; lanes that don't exist within its input.
define <vscale x 16 x i1> @reinterpret_test_w_rev(<vscale x 16 x i1> %a) {
; OPT-LABEL: @reinterpret_test_w_rev(
; OPT: %1 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %a)
; OPT-NEXT: %2 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %1)
; OPT-NEXT: ret <vscale x 16 x i1> %2
  %1 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %a)
  %2 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %1)
  ret <vscale x 16 x i1> %2
}

define <vscale x 2 x i1> @reinterpret_test_d(<vscale x 2 x i1> %a) {
; OPT-LABEL: @reinterpret_test_d(
; OPT-NOT: convert
; OPT: ret <vscale x 2 x i1> %a
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %a)
  %2 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %1)
  ret <vscale x 2 x i1> %2
}

; Reinterprets are not redundant because the second reinterpret zeros the
; lanes that don't exist within its input.
define <vscale x 16 x i1> @reinterpret_test_d_rev(<vscale x 16 x i1> %a) {
; OPT-LABEL: @reinterpret_test_d_rev(
; OPT: %1 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %a)
; OPT-NEXT: %2 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %1)
; OPT-NEXT: ret <vscale x 16 x i1> %2
  %1 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %a)
  %2 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %1)
  ret <vscale x 16 x i1> %2
}

define <vscale x 2 x i1> @reinterpret_reductions(i32 %cond, <vscale x 2 x i1> %a, <vscale x 2 x i1> %b, <vscale x 2 x i1> %c) {
; OPT-LABEL: reinterpret_reductions
; OPT-NOT: convert
; OPT-NOT: phi <vscale x 16 x i1>
; OPT: phi <vscale x 2 x i1> [ %a, %br_phi_a ], [ %b, %br_phi_b ], [ %c, %br_phi_c ]
; OPT-NOT: convert
; OPT: ret

entry:
  switch i32 %cond, label %br_phi_c [
         i32 43, label %br_phi_a
         i32 45, label %br_phi_b
  ]

br_phi_a:
  %a1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %a)
  br label %join

br_phi_b:
  %b1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %b)
  br label %join

br_phi_c:
  %c1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %c)
  br label %join

join:
  %pg = phi <vscale x 16 x i1> [ %a1, %br_phi_a ], [ %b1, %br_phi_b ], [ %c1, %br_phi_c ]
  %pg1 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  ret <vscale x 2 x i1> %pg1
}

; No transform as the reinterprets are converting from different types (nxv2i1 & nxv4i1)
; As the incoming values to the phi must all be the same type, we cannot remove the reinterprets.
define <vscale x 2 x i1> @reinterpret_reductions_1(i32 %cond, <vscale x 2 x i1> %a, <vscale x 4 x i1> %b, <vscale x 2 x i1> %c) {
; OPT-LABEL: reinterpret_reductions_1
; OPT: convert
; OPT: phi <vscale x 16 x i1> [ %a1, %br_phi_a ], [ %b1, %br_phi_b ], [ %c1, %br_phi_c ]
; OPT-NOT: phi <vscale x 2 x i1>
; OPT: tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
; OPT: ret

entry:
  switch i32 %cond, label %br_phi_c [
         i32 43, label %br_phi_a
         i32 45, label %br_phi_b
  ]

br_phi_a:
  %a1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %a)
  br label %join

br_phi_b:
  %b1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %b)
  br label %join

br_phi_c:
  %c1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %c)
  br label %join

join:
  %pg = phi <vscale x 16 x i1> [ %a1, %br_phi_a ], [ %b1, %br_phi_b ], [ %c1, %br_phi_c ]
  %pg1 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  ret <vscale x 2 x i1> %pg1
}

; No transform. Similar to the the test above, but here only two of the arguments need to
; be converted to svbool.
define <vscale x 2 x i1> @reinterpret_reductions_2(i32 %cond, <vscale x 2 x i1> %a, <vscale x 16 x i1> %b, <vscale x 2 x i1> %c) {
; OPT-LABEL: reinterpret_reductions_2
; OPT: convert
; OPT: phi <vscale x 16 x i1> [ %a1, %br_phi_a ], [ %b, %br_phi_b ], [ %c1, %br_phi_c ]
; OPT-NOT: phi <vscale x 2 x i1>
; OPT: tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
; OPT: ret

entry:
  switch i32 %cond, label %br_phi_c [
         i32 43, label %br_phi_a
         i32 45, label %br_phi_b
  ]

br_phi_a:
  %a1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %a)
  br label %join

br_phi_b:
  br label %join

br_phi_c:
  %c1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %c)
  br label %join

join:
  %pg = phi <vscale x 16 x i1> [ %a1, %br_phi_a ], [ %b, %br_phi_b ], [ %c1, %br_phi_c ]
  %pg1 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  ret <vscale x 2 x i1> %pg1
}

; Similar to reinterpret_reductions but the reinterprets remain because the
; original phi cannot be removed (i.e. prefer reinterprets over multiple phis).
define <vscale x 16 x i1> @reinterpret_reductions3(i32 %cond, <vscale x 2 x i1> %a, <vscale x 2 x i1> %b, <vscale x 2 x i1> %c) {
; OPT-LABEL: reinterpret_reductions3
; OPT: phi <vscale x 16 x i1> [ %a1, %br_phi_a ], [ %b1, %br_phi_b ], [ %c1, %br_phi_c ]
; OPT-NOT: phi <vscale x 2 x i1>
; OPT: tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
; OPT-NEXT: ret <vscale x 16 x i1> %pg

entry:
  switch i32 %cond, label %br_phi_c [
         i32 43, label %br_phi_a
         i32 45, label %br_phi_b
  ]

br_phi_a:
  %a1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %a)
  br label %join

br_phi_b:
  %b1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %b)
  br label %join

br_phi_c:
  %c1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %c)
  br label %join

join:
  %pg = phi <vscale x 16 x i1> [ %a1, %br_phi_a ], [ %b1, %br_phi_b ], [ %c1, %br_phi_c ]
  %pg1 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  ret <vscale x 16 x i1> %pg
}

declare <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1>)
