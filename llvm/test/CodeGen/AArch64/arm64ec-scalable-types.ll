; RUN: split-file %s %t
; RUN: llc -mtriple=arm64ec-pc-windows-msvc -mattr=+sve -verify-machineinstrs < %t/positive.ll | FileCheck %s --check-prefix=POS --implicit-check-not="#internal_sve_helper"
; RUN: llc -mtriple=arm64ec-pc-windows-msvc -mattr=+sve -arm64ec-generate-thunks=false -verify-machineinstrs < %t/no_thunks.ll | FileCheck %s --check-prefix=NOTHUNKS
; RUN: not llc -mtriple=arm64ec-pc-windows-msvc -filetype=null < %t/entry_ret.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc -mtriple=arm64ec-pc-windows-msvc -mattr=+sve -filetype=null < %t/entry_ret.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc -mtriple=arm64ec-pc-windows-msvc -mattr=+sve -filetype=null < %t/entry_arg.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc -mtriple=arm64ec-pc-windows-msvc -mattr=+sve,+sme -filetype=null < %t/entry_svcount.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc -mtriple=arm64ec-pc-windows-msvc -mattr=+sve -filetype=null < %t/entry_homog_struct.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc -mtriple=arm64ec-pc-windows-msvc -mattr=+sve -filetype=null < %t/entry_hetero_struct.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc -mtriple=arm64ec-pc-windows-msvc -mattr=+sve -filetype=null < %t/entry_array.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc -mtriple=arm64ec-pc-windows-msvc -mattr=+sve -filetype=null < %t/entry_sret.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc -mtriple=arm64ec-pc-windows-msvc -mattr=+sve -filetype=null < %t/entry_sret_inreg.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc -mtriple=arm64ec-pc-windows-msvc -mattr=+sve -filetype=null < %t/entry_vararg_fixed.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc -mtriple=arm64ec-pc-windows-msvc -mattr=+sve -filetype=null < %t/entry_addr_taken.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc -mtriple=arm64ec-pc-windows-msvc -mattr=+sve -filetype=null < %t/entry_patchable.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc -mtriple=arm64ec-pc-windows-msvc -mattr=+sve -filetype=null < %t/exit_direct.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc -mtriple=arm64ec-pc-windows-msvc -mattr=+sve -filetype=null < %t/exit_direct_mismatched.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc -mtriple=arm64ec-pc-windows-msvc -mattr=+sve -filetype=null < %t/exit_indirect.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc -mtriple=arm64ec-pc-windows-msvc -mattr=+sve -filetype=null < %t/exit_vararg_direct.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc -mtriple=arm64ec-pc-windows-msvc -mattr=+sve,+sme -filetype=null < %t/exit_vararg_indirect.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc -mtriple=arm64ec-pc-windows-msvc -mattr=+sve -filetype=null < %t/entry_sret_param1.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc -mtriple=arm64ec-pc-windows-msvc -mattr=+sve -filetype=null < %t/entry_byref.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc -mtriple=arm64ec-pc-windows-msvc -mattr=+sve -filetype=null < %t/entry_inalloca.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc -mtriple=arm64ec-pc-windows-msvc -mattr=+sve -filetype=null < %t/entry_preallocated.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc -mtriple=arm64ec-pc-windows-msvc -mattr=+sve -filetype=null < %t/exit_patchable_vararg.ll 2>&1 | FileCheck %s --check-prefix=ERR

; ERR: LLVM ERROR: scalable types are not supported for Arm64EC thunks

;--- positive.ll
;; Unused external declarations, SVE intrinsics, and non-address-taken local
;; helpers do not require Arm64EC thunks and should compile cleanly.
declare <vscale x 4 x i32> @unused_sve_decl(<vscale x 4 x i32>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 immarg)
declare <vscale x 4 x i32> @llvm.aarch64.sve.add.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)

; POS-LABEL: internal_sve_helper:
; POS:       add z0.s, p0/m, z0.s, z1.s
; POS-NEXT:  ret
define internal <vscale x 4 x i32> @internal_sve_helper(<vscale x 4 x i32> %x, <vscale x 4 x i32> %y, <vscale x 4 x i1> %pg) nounwind {
  %sum = call <vscale x 4 x i32> @llvm.aarch64.sve.add.nxv4i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %x, <vscale x 4 x i32> %y)
  ret <vscale x 4 x i32> %sum
}

; POS-LABEL: "#exported_caller":
; POS:       bl internal_sve_helper
; POS-LABEL: $ientry_thunk$cdecl$v$i8:
define void @exported_caller(ptr %p) nounwind {
  %x = load <vscale x 4 x i32>, ptr %p
  %pg = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %res = call <vscale x 4 x i32> @internal_sve_helper(<vscale x 4 x i32> %x, <vscale x 4 x i32> %x, <vscale x 4 x i1> %pg)
  store <vscale x 4 x i32> %res, ptr %p
  ret void
}

;--- no_thunks.ll
; NOTHUNKS-LABEL: no_thunks:
; NOTHUNKS:       ret
define <vscale x 4 x i32> @no_thunks(<vscale x 4 x i32> %x) nounwind {
  ret <vscale x 4 x i32> %x
}

;--- entry_ret.ll
define <vscale x 4 x i32> @ret_sve(<vscale x 4 x i32> %x) {
  ret <vscale x 4 x i32> %x
}

;--- entry_arg.ll
define void @arg_sve(<vscale x 4 x i32> %x) {
  ret void
}

;--- entry_svcount.ll
define void @arg_svcount(target("aarch64.svcount") %c) {
  ret void
}

;--- entry_homog_struct.ll
define { <vscale x 4 x i32>, <vscale x 4 x i32> } @homog_struct({ <vscale x 4 x i32>, <vscale x 4 x i32> } %x) {
  ret { <vscale x 4 x i32>, <vscale x 4 x i32> } %x
}

;--- entry_hetero_struct.ll
define void @hetero_struct({ i64, <vscale x 4 x i32> } %x) {
  ret void
}

;--- entry_array.ll
define void @scalable_array([2 x <vscale x 4 x i32>] %x) {
  ret void
}

;--- entry_sret.ll
define void @sret_sve(ptr sret(<vscale x 4 x i32>) %out) {
  ret void
}

;--- entry_sret_inreg.ll
define void @sret_inreg_sve(ptr inreg sret(<vscale x 4 x i32>) %out) {
  ret void
}

;--- entry_vararg_fixed.ll
define void @vararg_fixed(<vscale x 4 x i32> %x, ...) {
  ret void
}

;--- entry_addr_taken.ll
define internal <vscale x 4 x i32> @addr_taken(<vscale x 4 x i32> %x) {
  ret <vscale x 4 x i32> %x
}
@fp = global ptr @addr_taken

;--- entry_patchable.ll
define dso_local <vscale x 4 x i32> @patchable(<vscale x 4 x i32> %x) hybrid_patchable {
  ret <vscale x 4 x i32> %x
}

;--- exit_direct.ll
declare <vscale x 4 x i32> @ext_sve(<vscale x 4 x i32>)
define void @caller(ptr %p) {
  %v = load <vscale x 4 x i32>, ptr %p
  %r = call <vscale x 4 x i32> @ext_sve(<vscale x 4 x i32> %v)
  store <vscale x 4 x i32> %r, ptr %p
  ret void
}

;--- exit_direct_mismatched.ll
declare void @ext_unproto()
define void @caller(ptr %p) {
  %v = load <vscale x 4 x i32>, ptr %p
  call void (<vscale x 4 x i32>) @ext_unproto(<vscale x 4 x i32> %v)
  ret void
}

;--- exit_indirect.ll
define void @caller(ptr %fptr, ptr %p) {
  %v = load <vscale x 4 x i32>, ptr %p
  %r = call <vscale x 4 x i32> %fptr(<vscale x 4 x i32> %v)
  store <vscale x 4 x i32> %r, ptr %p
  ret void
}

;--- exit_vararg_direct.ll
declare void @ext_vararg(ptr, ...)
define void @caller(ptr %p) {
  %v = load <vscale x 4 x i32>, ptr %p
  call void (ptr, ...) @ext_vararg(ptr %p, <vscale x 4 x i32> %v)
  ret void
}

;--- exit_vararg_indirect.ll
declare target("aarch64.svcount") @llvm.aarch64.sve.ptrue.c8()
define void @caller(ptr %fptr, ptr %p) {
  %c = call target("aarch64.svcount") @llvm.aarch64.sve.ptrue.c8()
  call void (ptr, ...) %fptr(ptr %p, target("aarch64.svcount") %c)
  ret void
}

;--- entry_sret_param1.ll
define void @sret_param1_sve(ptr %this, ptr sret(<vscale x 4 x i32>) %out) {
  ret void
}

;--- entry_byref.ll
define void @byref_sve(ptr byref(<vscale x 4 x i32>) %out) {
  ret void
}

;--- entry_inalloca.ll
define void @inalloca_sve(ptr inalloca(<vscale x 4 x i32>) %out) {
  ret void
}

;--- entry_preallocated.ll
define void @preallocated_sve(ptr preallocated(<vscale x 4 x i32>) %out) {
  ret void
}

;--- exit_patchable_vararg.ll
declare dso_local <vscale x 4 x i32> @patchable(ptr, ...) hybrid_patchable
define void @caller(ptr %p) {
  %v = load <vscale x 4 x i32>, ptr %p
  %r = call <vscale x 4 x i32> (ptr, ...) @patchable(ptr %p, <vscale x 4 x i32> %v)
  store <vscale x 4 x i32> %r, ptr %p
  ret void
}
