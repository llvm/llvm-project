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

; aarch64.sve.tuple.create.N
define <vscale x 32 x i8> @create2_nxv32i8_nxv16i8(<vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2) {
; CHECK-LABEL: @create2_nxv32i8_nxv16i8
; CHECK:  %1 = call <vscale x 32 x i8> @llvm.vector.insert.nxv32i8.nxv16i8(<vscale x 32 x i8> poison, <vscale x 16 x i8> %z1, i64 0)
; CHECK-NEXT:  %tuple = call <vscale x 32 x i8> @llvm.vector.insert.nxv32i8.nxv16i8(<vscale x 32 x i8> %1, <vscale x 16 x i8> %z2, i64 16)
; CHECK-NEXT:  ret <vscale x 32 x i8> %tuple

  %tuple = tail call <vscale x 32 x i8> @llvm.aarch64.sve.tuple.create2.nxv32i8.nxv16i8(<vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2)
  ret <vscale x 32 x i8> %tuple
}

define <vscale x 24 x i16> @create3_nxv24i8_nxv16i8(<vscale x 8 x i16> %unused_z0, <vscale x 8 x i16> %z1, <vscale x 8 x i16> %z2, <vscale x 8 x i16> %z3) {
; CHECK-LABEL: @create3_nxv24i8_nxv16i8
; CHECK:  %1 = call <vscale x 24 x i16> @llvm.vector.insert.nxv24i16.nxv8i16(<vscale x 24 x i16> poison, <vscale x 8 x i16> %z1, i64 0)
; CHECK-NEXT:  %2 = call <vscale x 24 x i16> @llvm.vector.insert.nxv24i16.nxv8i16(<vscale x 24 x i16> %1, <vscale x 8 x i16> %z2, i64 8)
; CHECK-NEXT:  %tuple = call <vscale x 24 x i16> @llvm.vector.insert.nxv24i16.nxv8i16(<vscale x 24 x i16> %2, <vscale x 8 x i16> %z3, i64 16)
; CHECK-NEXT:  ret <vscale x 24 x i16> %tuple

  %tuple = tail call <vscale x 24 x i16> @llvm.aarch64.sve.tuple.create3.nxv24i16.nxv8i16(<vscale x 8 x i16> %z1, <vscale x 8 x i16> %z2, <vscale x 8 x i16> %z3)
  ret <vscale x 24 x i16> %tuple
}

define <vscale x 64 x i8> @create4_nxv64i8_nxv16i8(<vscale x 16 x i8> %unused_z0, <vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2, <vscale x 16 x i8> %z3, <vscale x 16 x i8> %z4) {
; CHECK-LABEL: @create4_nxv64i8_nxv16i8
; CHECK:  %1 = call <vscale x 64 x i8> @llvm.vector.insert.nxv64i8.nxv16i8(<vscale x 64 x i8> poison, <vscale x 16 x i8> %z1, i64 0)
; CHECK-NEXT:  %2 = call <vscale x 64 x i8> @llvm.vector.insert.nxv64i8.nxv16i8(<vscale x 64 x i8> %1, <vscale x 16 x i8> %z2, i64 16)
; CHECK-NEXT:  %3 = call <vscale x 64 x i8> @llvm.vector.insert.nxv64i8.nxv16i8(<vscale x 64 x i8> %2, <vscale x 16 x i8> %z3, i64 32)
; CHECK-NEXT:  %tuple = call <vscale x 64 x i8> @llvm.vector.insert.nxv64i8.nxv16i8(<vscale x 64 x i8> %3, <vscale x 16 x i8> %z4, i64 48)
; CHECK-NEXT:  ret <vscale x 64 x i8> %tuple

  %tuple = tail call <vscale x 64 x i8> @llvm.aarch64.sve.tuple.create4.nxv64i8.nxv16i8(<vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2, <vscale x 16 x i8> %z3, <vscale x 16 x i8> %z4)
  ret <vscale x 64 x i8> %tuple
}

; Accept short mangling name
define <vscale x 32 x i8> @create2_nxv32i8(<vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2) {
; CHECK-LABEL: @create2_nxv32i8
; CHECK:  %1 = call <vscale x 32 x i8> @llvm.vector.insert.nxv32i8.nxv16i8(<vscale x 32 x i8> poison, <vscale x 16 x i8> %z1, i64 0)
; CHECK-NEXT:  %tuple = call <vscale x 32 x i8> @llvm.vector.insert.nxv32i8.nxv16i8(<vscale x 32 x i8> %1, <vscale x 16 x i8> %z2, i64 16)
; CHECK-NEXT:  ret <vscale x 32 x i8> %tuple

  %tuple = tail call <vscale x 32 x i8> @llvm.aarch64.sve.tuple.create2.nxv32i8(<vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2)
  ret <vscale x 32 x i8> %tuple
}

define <vscale x 32 x i8> @create2(<vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2) {
; CHECK-LABEL: @create2
; CHECK:  %1 = call <vscale x 32 x i8> @llvm.vector.insert.nxv32i8.nxv16i8(<vscale x 32 x i8> poison, <vscale x 16 x i8> %z1, i64 0)
; CHECK-NEXT:  %tuple = call <vscale x 32 x i8> @llvm.vector.insert.nxv32i8.nxv16i8(<vscale x 32 x i8> %1, <vscale x 16 x i8> %z2, i64 16)
; CHECK-NEXT:  ret <vscale x 32 x i8> %tuple

  %tuple = tail call <vscale x 32 x i8> @llvm.aarch64.sve.tuple.create2(<vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2)
  ret <vscale x 32 x i8> %tuple
}

; Negative test for create
; Should not upgrade when create is not 2,3 or 4
define <vscale x 16 x i8> @sve_tuple_create1(<vscale x 16 x i8> %z0) {
; CHECK-LABEL: @sve_tuple_create1
; CHECK: %tuple = tail call <vscale x 16 x i8> @llvm.aarch64.sve.tuple.create1.nxv16i8.nxv16i8(<vscale x 16 x i8> %z0)
; CHECK-NEXT:  ret <vscale x 16 x i8> %tuple

  %tuple  = tail call <vscale x 16 x i8> @llvm.aarch64.sve.tuple.create1.nxv16i8.nxv16i8(<vscale x 16 x i8> %z0);
  ret <vscale x 16 x i8> %tuple;
}

; aarch64.sve.tuple.set

define void  @set_tuple2_nxv8i32_elt1(<vscale x 8 x i32> %z0, <vscale x 4 x i32> %z1) {
; CHECK-LABEL: @set_tuple2_nxv8i32_elt1
; CHECK:  %ins = call <vscale x 8 x i32> @llvm.vector.insert.nxv8i32.nxv4i32(<vscale x 8 x i32> %z0, <vscale x 4 x i32> %z1, i64 4)
; CHECK-NEXT: ret void

  %ins = call <vscale x 8 x i32> @llvm.aarch64.sve.tuple.set.nxv8i32.nxv4i32(<vscale x 8 x i32> %z0, i32 1, <vscale x 4 x i32> %z1)
  ret void
}

; aarch64.sve.tuple.get
define <vscale x 4 x i32> @get_tuple2_nxv8i32_elt1(<vscale x 8 x i32> %tuple) {
; CHECK-LABEL: @get_tuple2_nxv8i32_elt1
; CHECK:  %ext = call <vscale x 4 x i32> @llvm.vector.extract.nxv4i32.nxv8i32(<vscale x 8 x i32> %tuple, i64 4)
; CHECK-NEXT:  ret <vscale x 4 x i32> %ext

  %ext = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv8i32(<vscale x 8 x i32> %tuple, i32 1)
  ret <vscale x 4 x i32> %ext
}

; bfdot
define <vscale x 4 x float> @bfdot_lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: @bfdot_lane
; CHECK:       %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfdot.lane.v2(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i32 0)
; CHECK-NEXT:  ret <vscale x 4 x float> %out
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfdot.lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i64 0)
  ret <vscale x 4 x float> %out
}

; bfmlalb
define <vscale x 4 x float> @bfmlalb_lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: @bfmlalb_lane
; CHECK:       %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfmlalb.lane.v2(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i32 0)
; CHECK-NEXT:  ret <vscale x 4 x float> %out
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfmlalb.lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i64 0)
  ret <vscale x 4 x float> %out
}

; bfmlalt
define <vscale x 4 x float> @bfmlalt_lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c) nounwind {
; CHECK-LABEL: @bfmlalt_lane
; CHECK:       %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfmlalt.lane.v2(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i32 0)
; CHECK-NEXT:  ret <vscale x 4 x float> %out
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.bfmlalt.lane(<vscale x 4 x float> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, i64 0)
  ret <vscale x 4 x float> %out
}

declare  <vscale x 32 x i8> @llvm.aarch64.sve.tuple.create2.nxv32i8.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare  <vscale x 32 x i8> @llvm.aarch64.sve.tuple.create2.nxv32i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare  <vscale x 32 x i8> @llvm.aarch64.sve.tuple.create2(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare  <vscale x 24 x i16> @llvm.aarch64.sve.tuple.create3.nxv24i16.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 64 x i8> @llvm.aarch64.sve.tuple.create4.nxv64i8.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.tuple.create1.nxv16i8.nxv16i8(<vscale x 16 x i8>)
declare <vscale x 8 x i32> @llvm.aarch64.sve.tuple.set.nxv8i32.nxv4i32(<vscale x 8 x i32>, i32, <vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv8i32(<vscale x 8 x i32>, i32)
declare <vscale x 4 x float> @llvm.aarch64.sve.bfdot.lane(<vscale x 4 x float>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, i64)
declare <vscale x 4 x float> @llvm.aarch64.sve.bfmlalb.lane(<vscale x 4 x float>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, i64)
declare <vscale x 4 x float> @llvm.aarch64.sve.bfmlalt.lane(<vscale x 4 x float>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, i64)
