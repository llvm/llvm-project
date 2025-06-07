; Testfile that verifies positive case (0 or 1 only) for BCD builtins national2packed, packed2zoned and zoned2packed.
; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown \
; RUN:   -ppc-asm-full-reg-names  < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mtriple=powerpc64-unknown-unknown \
; RUN:   -ppc-asm-full-reg-names  < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mtriple=powerpc-unknown-unknown \
; RUN:   -ppc-asm-full-reg-names  < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mtriple=powerpc64-ibm-aix-xcoff \
; RUN:   -ppc-asm-full-reg-names  < %s | FileCheck %s

; CHECK-LABEL: tBcd_National2packed_imm0
; CHECK:         bcdcfn. v2, v2, 0
; CHECK-NEXT:    blr

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <16 x i8> @llvm.ppc.national2packed(<16 x i8>, i32 immarg) 

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define <16 x i8> @tBcd_National2packed_imm0(<16 x i8> noundef %a) {
entry:
  %0 = tail call <16 x i8> @llvm.ppc.national2packed(<16 x i8> %a, i32 0)
  ret <16 x i8> %0
}

; CHECK-LABEL: tBcd_National2packed_imm1
; CHECK:         bcdcfn. v2, v2, 1
; CHECK-NEXT:    blr

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define <16 x i8> @tBcd_National2packed_imm1(<16 x i8> noundef %a) {
entry:
  %0 = tail call <16 x i8> @llvm.ppc.national2packed(<16 x i8> %a, i32 1)
  ret <16 x i8> %0
}

; CHECK-LABEL: tBcd_Packed2national
; CHECK:         bcdctn. v2, v2
; CHECK-NEXT:    blr
; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <16 x i8> @llvm.ppc.packed2national(<16 x i8>)

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define <16 x i8> @tBcd_Packed2national(<16 x i8> noundef %a) {
entry:
  %0 = tail call <16 x i8> @llvm.ppc.packed2national(<16 x i8> %a)
  ret <16 x i8> %0
}

; CHECK-LABEL: tBcd_Packed2zoned_imm0
; CHECK: 	      bcdctz. v2, v2, 0
; CHECK-NEXT:	  blr
; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <16 x i8> @llvm.ppc.packed2zoned(<16 x i8>, i32 immarg)

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local <16 x i8> @tBcd_Packed2zoned_imm0(<16 x i8> noundef %a) {
entry:
  %0 = tail call <16 x i8> @llvm.ppc.packed2zoned(<16 x i8> %a, i32 0)
  ret <16 x i8> %0
}

; CHECK-LABEL: tBcd_Packed2zoned_imm1
; CHECK: 	      bcdctz. v2, v2, 1
; CHECK-NEXT:	  blr
; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local <16 x i8> @tBcd_Packed2zoned_imm1(<16 x i8> noundef %a) {
entry:
  %0 = tail call <16 x i8> @llvm.ppc.packed2zoned(<16 x i8> %a, i32 1)
  ret <16 x i8> %0
}

; CHECK-LABEL: tBcd_Zoned2packed_imm0
; CHECK:        bcdcfz. v2, v2, 0
; CHECK-NEXT:   blr
; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <16 x i8> @llvm.ppc.zoned2packed(<16 x i8>, i32 immarg)

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local <16 x i8> @tBcd_Zoned2packed_imm0(<16 x i8> noundef %a) {
entry:
  %0 = tail call <16 x i8> @llvm.ppc.zoned2packed(<16 x i8> %a, i32 0)
  ret <16 x i8> %0
}

; CHECK-LABEL: tBcd_Zoned2packed_imm1
; CHECK:        bcdcfz. v2, v2, 1
; CHECK-NEXT:   blr
; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local <16 x i8> @tBcd_Zoned2packed_imm1(<16 x i8> noundef %a) {
entry:
  %0 = tail call <16 x i8> @llvm.ppc.zoned2packed(<16 x i8> %a, i32 1)
  ret <16 x i8> %0
}