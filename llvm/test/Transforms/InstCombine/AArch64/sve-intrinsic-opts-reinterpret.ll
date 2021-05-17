; RUN: opt -S -instcombine < %s | FileCheck %s

target triple = "aarch64"

define <vscale x 8 x i1> @reinterpret_test_h(<vscale x 8 x i1> %a) {
; CHECK-LABEL: @reinterpret_test_h(
; CHECK-NOT: convert
; CHECK: ret <vscale x 8 x i1> %a
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %a)
  %2 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %1)
  ret <vscale x 8 x i1> %2
}

; Reinterprets are not redundant because the second reinterpret zeros the
; lanes that don't exist within its input.
define <vscale x 16 x i1> @reinterpret_test_h_rev(<vscale x 16 x i1> %a) {
; CHECK-LABEL: @reinterpret_test_h_rev(
; CHECK: %1 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %a)
; CHECK-NEXT: %2 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %1)
; CHECK-NEXT: ret <vscale x 16 x i1> %2
  %1 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %a)
  %2 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %1)
  ret <vscale x 16 x i1> %2
}

define <vscale x 4 x i1> @reinterpret_test_w(<vscale x 4 x i1> %a) {
; CHECK-LABEL: @reinterpret_test_w(
; CHECK-NOT: convert
; CHECK: ret <vscale x 4 x i1> %a
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %a)
  %2 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %1)
  ret <vscale x 4 x i1> %2
}

; Reinterprets are not redundant because the second reinterpret zeros the
; lanes that don't exist within its input.
define <vscale x 16 x i1> @reinterpret_test_w_rev(<vscale x 16 x i1> %a) {
; CHECK-LABEL: @reinterpret_test_w_rev(
; CHECK: %1 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %a)
; CHECK-NEXT: %2 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %1)
; CHECK-NEXT: ret <vscale x 16 x i1> %2
  %1 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %a)
  %2 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %1)
  ret <vscale x 16 x i1> %2
}

define <vscale x 2 x i1> @reinterpret_test_d(<vscale x 2 x i1> %a) {
; CHECK-LABEL: @reinterpret_test_d(
; CHECK-NOT: convert
; CHECK: ret <vscale x 2 x i1> %a
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %a)
  %2 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %1)
  ret <vscale x 2 x i1> %2
}

; Reinterprets are not redundant because the second reinterpret zeros the
; lanes that don't exist within its input.
define <vscale x 16 x i1> @reinterpret_test_d_rev(<vscale x 16 x i1> %a) {
; CHECK-LABEL: @reinterpret_test_d_rev(
; CHECK: %1 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %a)
; CHECK-NEXT: %2 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %1)
; CHECK-NEXT: ret <vscale x 16 x i1> %2
  %1 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %a)
  %2 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %1)
  ret <vscale x 16 x i1> %2
}

define <vscale x 2 x i1> @reinterpret_test_full_chain(<vscale x 2 x i1> %a) {
; CHECK-LABEL: @reinterpret_test_full_chain(
; CHECK: ret <vscale x 2 x i1> %a
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %a)
  %2 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %1)
  %3 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %2)
  %4 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %3)
  %5 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %4)
  %6 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %5)
  ret <vscale x 2 x i1> %6
}

; The last two reinterprets are not necessary, since they are doing the same
; work as the first two.
define <vscale x 4 x i1> @reinterpret_test_partial_chain(<vscale x 2 x i1> %a) {
; CHECK-LABEL: @reinterpret_test_partial_chain(
; CHECK: %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %a)
; CHECK-NEXT: %2 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %1)
; CHECK-NEXT: ret <vscale x 4 x i1> %2
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %a)
  %2 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %1)
  %3 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %2)
  %4 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %3)
  ret <vscale x 4 x i1> %4
}

; The chain cannot be reduced because of the second reinterpret, which causes
; zeroing.
define <vscale x 8 x i1> @reinterpret_test_irreducible_chain(<vscale x 8 x i1> %a) {
; CHECK-LABEL: @reinterpret_test_irreducible_chain(
; CHECK: %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %a)
; CHECK-NEXT: %2 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %1)
; CHECK-NEXT: %3 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %2)
; CHECK-NEXT: %4 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %3)
; CHECK-NEXT: ret <vscale x 8 x i1> %4
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %a)
  %2 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %1)
  %3 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %2)
  %4 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %3)
  ret <vscale x 8 x i1> %4
}

; Here, the candidate list is larger than the number of instructions that we
; end up removing.
define <vscale x 4 x i1> @reinterpret_test_keep_some_candidates(<vscale x 8 x i1> %a) {
; CHECK-LABEL: @reinterpret_test_keep_some_candidates(
; CHECK: %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %a)
; CHECK-NEXT: %2 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %1)
; CHECK-NEXT: ret <vscale x 4 x i1> %2
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %a)
  %2 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %1)
  %3 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %2)
  %4 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %3)
  ret <vscale x 4 x i1> %4
}

define <vscale x 2 x i1> @reinterpret_reductions(i32 %cond, <vscale x 2 x i1> %a, <vscale x 2 x i1> %b, <vscale x 2 x i1> %c) {
; CHECK-LABEL: reinterpret_reductions
; CHECK-NOT: convert
; CHECK-NOT: phi <vscale x 16 x i1>
; CHECK: phi <vscale x 2 x i1> [ %a, %br_phi_a ], [ %b, %br_phi_b ], [ %c, %br_phi_c ]
; CHECK-NOT: convert
; CHECK: ret

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
; CHECK-LABEL: reinterpret_reductions_1
; CHECK: convert
; CHECK: phi <vscale x 16 x i1> [ %a1, %br_phi_a ], [ %b1, %br_phi_b ], [ %c1, %br_phi_c ]
; CHECK-NOT: phi <vscale x 2 x i1>
; CHECK: tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
; CHECK: ret

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
; CHECK-LABEL: reinterpret_reductions_2
; CHECK: convert
; CHECK: phi <vscale x 16 x i1> [ %a1, %br_phi_a ], [ %b, %br_phi_b ], [ %c1, %br_phi_c ]
; CHECK-NOT: phi <vscale x 2 x i1>
; CHECK: tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
; CHECK: ret

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
; CHECK-LABEL: reinterpret_reductions3
; CHECK: phi <vscale x 16 x i1> [ %a1, %br_phi_a ], [ %b1, %br_phi_b ], [ %c1, %br_phi_c ]
; CHECK-NOT: phi <vscale x 2 x i1>
; CHECK: ret <vscale x 16 x i1> %pg

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

attributes #0 = { "target-features"="+sve" }
