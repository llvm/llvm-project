; RUN: llvm-as %s -o - | llvm-dis - | FileCheck %s

define <vscale x 32 x i8> @ld2.nxv32i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr) {
; CHECK:  %1 = call { <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld2.sret.nxv16i8(<vscale x 16 x i1> %Pg, i8* %base_ptr)
; CHECK-NEXT:  %2 = extractvalue { <vscale x 16 x i8>, <vscale x 16 x i8> } %1, 0
; CHECK-NEXT:  %3 = call <vscale x 32 x i8> @llvm.vector.insert.nxv32i8.nxv16i8(<vscale x 32 x i8> poison, <vscale x 16 x i8> %2, i64 0)
; CHECK-NEXT:  %4 = extractvalue { <vscale x 16 x i8>, <vscale x 16 x i8> } %1, 1
; CHECK-NEXT:  %res = call <vscale x 32 x i8> @llvm.vector.insert.nxv32i8.nxv16i8(<vscale x 32 x i8> %3, <vscale x 16 x i8> %4, i64 16)
; CHECK-NEXT:  ret <vscale x 32 x i8> %res
%res = call <vscale x 32 x i8> @llvm.aarch64.sve.ld2.nxv32i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 32 x i8> %res
}

define <vscale x 48 x i8> @ld3.nxv48i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr) {
; CHECK:  %1 = call { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld3.sret.nxv16i8(<vscale x 16 x i1> %Pg, i8* %base_ptr)
; CHECK-NEXT:  %2 = extractvalue { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %1, 0
; CHECK-NEXT:  %3 = call <vscale x 48 x i8> @llvm.vector.insert.nxv48i8.nxv16i8(<vscale x 48 x i8> poison, <vscale x 16 x i8> %2, i64 0)
; CHECK-NEXT:  %4 = extractvalue { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %1, 1
; CHECK-NEXT:  %5 = call <vscale x 48 x i8> @llvm.vector.insert.nxv48i8.nxv16i8(<vscale x 48 x i8> %3, <vscale x 16 x i8> %4, i64 16)
; CHECK-NEXT:  %6 = extractvalue { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %1, 2
; CHECK-NEXT:  %res = call <vscale x 48 x i8> @llvm.vector.insert.nxv48i8.nxv16i8(<vscale x 48 x i8> %5, <vscale x 16 x i8> %6, i64 32)
; CHECK-NEXT:  ret <vscale x 48 x i8> %res
%res = call <vscale x 48 x i8> @llvm.aarch64.sve.ld3.nxv48i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 48 x i8> %res
}

define <vscale x 64 x i8> @ld4.nxv64i8_lower_bound(<vscale x 16 x i1> %Pg, i8 *%base_ptr) {
; CHECK:  %1 = call { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld4.sret.nxv16i8(<vscale x 16 x i1> %Pg, i8* %base_ptr)
; CHECK-NEXT:  %2 = extractvalue { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %1, 0
; CHECK-NEXT:  %3 = call <vscale x 64 x i8> @llvm.vector.insert.nxv64i8.nxv16i8(<vscale x 64 x i8> poison, <vscale x 16 x i8> %2, i64 0)
; CHECK-NEXT:  %4 = extractvalue { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %1, 1
; CHECK-NEXT:  %5 = call <vscale x 64 x i8> @llvm.vector.insert.nxv64i8.nxv16i8(<vscale x 64 x i8> %3, <vscale x 16 x i8> %4, i64 16)
; CHECK-NEXT:  %6 = extractvalue { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %1, 2
; CHECK-NEXT:  %7 = call <vscale x 64 x i8> @llvm.vector.insert.nxv64i8.nxv16i8(<vscale x 64 x i8> %5, <vscale x 16 x i8> %6, i64 32)
; CHECK-NEXT:  %8 = extractvalue { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %1, 3
; CHECK-NEXT:  %res = call <vscale x 64 x i8> @llvm.vector.insert.nxv64i8.nxv16i8(<vscale x 64 x i8> %7, <vscale x 16 x i8> %8, i64 48)
; CHECK-NEXT:  ret <vscale x 64 x i8> %res
%res = call <vscale x 64 x i8> @llvm.aarch64.sve.ld4.nxv64i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 64 x i8> %res
}

; Check short mangling name

; ldN intrinsic name without any element type
define <vscale x 32 x i8> @ld2.nxv32i8_no_eltty(<vscale x 16 x i1> %Pg, i8 *%base_ptr) {
; CHECK-LABEL:  @ld2.nxv32i8_no_eltty
; CHECK:  %1 = call { <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld2.sret.nxv16i8(<vscale x 16 x i1> %Pg, i8* %base_ptr)
; CHECK-NEXT:  %2 = extractvalue { <vscale x 16 x i8>, <vscale x 16 x i8> } %1, 0
; CHECK-NEXT:  %3 = call <vscale x 32 x i8> @llvm.vector.insert.nxv32i8.nxv16i8(<vscale x 32 x i8> poison, <vscale x 16 x i8> %2, i64 0)
; CHECK-NEXT:  %4 = extractvalue { <vscale x 16 x i8>, <vscale x 16 x i8> } %1, 1
; CHECK-NEXT:  %res = call <vscale x 32 x i8> @llvm.vector.insert.nxv32i8.nxv16i8(<vscale x 32 x i8> %3, <vscale x 16 x i8> %4, i64 16)
; CHECK-NEXT:  ret <vscale x 32 x i8> %res
%res = call <vscale x 32 x i8> @llvm.aarch64.sve.ld2(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 32 x i8> %res
}

; ldN instrinsic name with only output type
define <vscale x 32 x i8> @ld2.nxv32i8_no_predty_pty(<vscale x 16 x i1> %Pg, i8 *%base_ptr) {
; CHECK-LABEL:  @ld2.nxv32i8_no_predty_pty
; CHECK:  %1 = call { <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld2.sret.nxv16i8(<vscale x 16 x i1> %Pg, i8* %base_ptr)
; CHECK-NEXT:  %2 = extractvalue { <vscale x 16 x i8>, <vscale x 16 x i8> } %1, 0
; CHECK-NEXT:  %3 = call <vscale x 32 x i8> @llvm.vector.insert.nxv32i8.nxv16i8(<vscale x 32 x i8> poison, <vscale x 16 x i8> %2, i64 0)
; CHECK-NEXT:  %4 = extractvalue { <vscale x 16 x i8>, <vscale x 16 x i8> } %1, 1
; CHECK-NEXT:  %res = call <vscale x 32 x i8> @llvm.vector.insert.nxv32i8.nxv16i8(<vscale x 32 x i8> %3, <vscale x 16 x i8> %4, i64 16)
; CHECK-NEXT:  ret <vscale x 32 x i8> %res
%res = call <vscale x 32 x i8> @llvm.aarch64.sve.ld2.nxv32i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 32 x i8> %res
}

declare <vscale x 32 x i8> @llvm.aarch64.sve.ld2.nxv32i8.nxv16i1.p0i8(<vscale x 16 x i1>, i8*)
declare <vscale x 48 x i8> @llvm.aarch64.sve.ld3.nxv48i8.nxv16i1.p0i8(<vscale x 16 x i1>, i8*)
declare <vscale x 64 x i8> @llvm.aarch64.sve.ld4.nxv64i8.nxv16i1.p0i8(<vscale x 16 x i1>, i8*)
declare <vscale x 32 x i8> @llvm.aarch64.sve.ld2(<vscale x 16 x i1>, i8 *)
declare <vscale x 32 x i8> @llvm.aarch64.sve.ld2.nxv32i8(<vscale x 16 x i1>, i8 *)
