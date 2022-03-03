; RUN: llc -mtriple aarch64 -mattr=+sve -asm-verbose=0 < %s | FileCheck %s
; RUN: llc -mtriple aarch64 -mattr=+streaming-sve -asm-verbose=0 < %s | FileCheck %s

; All these tests create a vector tuple, insert z5 into one of the elements,
; and finally extracts that element from the wide vector to return it.  These
; checks ensure that z5 is always the value that is returned.

;
; Insert into two element tuples
;

; tuple:      { tuple2.res0, tuple2.res1 }
; insert z5:  {     z5     , tuple2.res1 }
; extract z5:       ^^
define <vscale x 4 x i32> @set_tuple2_nxv8i32_elt0(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1,
                                                   <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3,
                                                   <vscale x 4 x i32> %z4, <vscale x 4 x i32> %z5) #0 {
  ; CHECK-LABEL: set_tuple2_nxv8i32_elt0:
  ; CHECK-NEXT:  mov     z0.d, z5.d
  ; CHECK-NEXT:  ret
  %tuple = call <vscale x 8 x i32> @llvm.aarch64.sve.tuple.create2.nxv8i32.nxv4i32(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1)
  %ins = call <vscale x 8 x i32> @llvm.aarch64.sve.tuple.set.nxv8i32.nxv4i32(<vscale x 8 x i32> %tuple, i32 0, <vscale x 4 x i32> %z5)
  %ext = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv8i32(<vscale x 8 x i32> %ins, i32 0)
  ret <vscale x 4 x i32> %ext
}

; tuple:       { tuple2.res0, tuple2.res1 }
; insert z5:   { tuple2.res0,     z5      }
; extract z5:                     ^^
define <vscale x 4 x i32> @set_tuple2_nxv8i32_elt1(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1,
                                                   <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3,
                                                   <vscale x 4 x i32> %z4, <vscale x 4 x i32> %z5) #0 {
  ; CHECK-LABEL: set_tuple2_nxv8i32_elt1:
  ; CHECK-NEXT:  mov     z0.d, z5.d
  ; CHECK-NEXT:  ret
  %tuple = call <vscale x 8 x i32> @llvm.aarch64.sve.tuple.create2.nxv8i32.nxv4i32(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1)
  %ins = call <vscale x 8 x i32> @llvm.aarch64.sve.tuple.set.nxv8i32.nxv4i32(<vscale x 8 x i32> %tuple, i32 1, <vscale x 4 x i32> %z5)
  %ext = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv8i32(<vscale x 8 x i32> %ins, i32 1)
  ret <vscale x 4 x i32> %ext
}

; This test checks the elements _not_ being set aren't changed.

; tuple:       { tuple2.res0, tuple2.res1 }
; insert z5:   { tuple2.res0,     z5      }
; extract z0:         ^^
define <vscale x 4 x i32> @set_tuple2_nxv8i32_elt1_ret_elt0(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1,
                                                            <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3,
                                                            <vscale x 4 x i32> %z4, <vscale x 4 x i32> %z5) #0 {
  ; CHECK-LABEL: set_tuple2_nxv8i32_elt1_ret_elt0:
  ; CHECK-NEXT:  ret
  %tuple = call <vscale x 8 x i32> @llvm.aarch64.sve.tuple.create2.nxv8i32.nxv4i32(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1)
  %ins = call <vscale x 8 x i32> @llvm.aarch64.sve.tuple.set.nxv8i32.nxv4i32(<vscale x 8 x i32> %tuple, i32 1, <vscale x 4 x i32> %z5)
  %ext = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv8i32(<vscale x 8 x i32> %ins, i32 0)
  ret <vscale x 4 x i32> %ext
}

; Test extract of tuple passed into function
define <vscale x 4 x i32> @get_tuple2_nxv8i32_elt1(<vscale x 8 x i32> %tuple) #0 {
  ; CHECK-LABEL: get_tuple2_nxv8i32_elt1:
  ; CHECK-NEXT:  mov     z0.d, z1.d
  ; CHECK-NEXT:  ret
  %ext = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv8i32(<vscale x 8 x i32> %tuple, i32 1)
  ret <vscale x 4 x i32> %ext
}

;
; Insert into three element tuples
;

; tuple:       { tuple3.res0, tuple3.res1, tuple3.res2 }
; insert z5:   {     z5     , tuple3.res0, tuple3.res2 }
; extract z5:        ^^
define <vscale x 4 x i32> @set_tuple3_nxv12i32_elt0(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1,
                                                    <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3,
                                                    <vscale x 4 x i32> %z4, <vscale x 4 x i32> %z5) #0 {
  ; CHECK-LABEL: set_tuple3_nxv12i32_elt0:
  ; CHECK-NEXT:  mov     z0.d, z5.d
  ; CHECK-NEXT:  ret
  %tuple = call <vscale x 12 x i32> @llvm.aarch64.sve.tuple.create3.nxv12i32.nxv4i32(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2)
  %ins = call <vscale x 12 x i32> @llvm.aarch64.sve.tuple.set.nxv12i32.nxv4i32(<vscale x 12 x i32> %tuple, i32 0, <vscale x 4 x i32> %z5)
  %ext = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv12i32(<vscale x 12 x i32> %ins, i32 0)
  ret <vscale x 4 x i32> %ext
}

; tuple:       { tuple3.res0, tuple3.res1, tuple3.res2 }
; insert z5:   { tuple3.res0,     z5     , tuple3.res2 }
; extract z5:                     ^^
define <vscale x 4 x i32> @set_tuple3_nxv12i32_elt1(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1,
                                                    <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3,
                                                    <vscale x 4 x i32> %z4, <vscale x 4 x i32> %z5) #0 {
  ; CHECK-LABEL: set_tuple3_nxv12i32_elt1:
  ; CHECK-NEXT:  mov     z0.d, z5.d
  ; CHECK-NEXT:  ret
  %tuple = call <vscale x 12 x i32> @llvm.aarch64.sve.tuple.create3.nxv12i32.nxv4i32(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2)
  %ins = call <vscale x 12 x i32> @llvm.aarch64.sve.tuple.set.nxv12i32.nxv4i32(<vscale x 12 x i32> %tuple, i32 1, <vscale x 4 x i32> %z5)
  %ext = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv12i32(<vscale x 12 x i32> %ins, i32 1)
  ret <vscale x 4 x i32> %ext
}

; tuple:       { tuple3.res0, tuple3.res1, tuple3.res2 }
; insert z5:   { tuple3.res0, tuple3.res1,     z5      }
; extract z5:                                  ^^
define <vscale x 4 x i32> @set_tuple3_nxv12i32_elt2(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1,
                                                    <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3,
                                                    <vscale x 4 x i32> %z4, <vscale x 4 x i32> %z5) #0 {
  ; CHECK-LABEL: set_tuple3_nxv12i32_elt2:
  ; CHECK-NEXT:  mov     z0.d, z5.d
  ; CHECK-NEXT:  ret
  %tuple = call <vscale x 12 x i32> @llvm.aarch64.sve.tuple.create3.nxv12i32.nxv4i32(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2)
  %ins = call <vscale x 12 x i32> @llvm.aarch64.sve.tuple.set.nxv12i32.nxv4i32(<vscale x 12 x i32> %tuple, i32 2, <vscale x 4 x i32> %z5)
  %ext = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv12i32(<vscale x 12 x i32> %ins, i32 2)
  ret <vscale x 4 x i32> %ext
}

; This test checks the elements _not_ being set aren't changed.

; tuple:       { tuple3.res0, tuple3.res1, tuple3.res2 }
; insert z5:   { tuple3.res0,     z5     , tuple3.res2 }
; extract z2:                                  ^^
define <vscale x 4 x i32> @set_tuple3_nxv12i32_elt1_ret_elt2(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1,
                                                             <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3,
                                                             <vscale x 4 x i32> %z4, <vscale x 4 x i32> %z5) #0 {
  ; CHECK-LABEL: set_tuple3_nxv12i32_elt1_ret_elt2:
  ; CHECK-NEXT:  mov     z0.d, z2.d
  ; CHECK-NEXT:  ret
  %tuple = call <vscale x 12 x i32> @llvm.aarch64.sve.tuple.create3.nxv12i32.nxv4i32(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2)
  %ins = call <vscale x 12 x i32> @llvm.aarch64.sve.tuple.set.nxv12i32.nxv4i32(<vscale x 12 x i32> %tuple, i32 1, <vscale x 4 x i32> %z5)
  %ext = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv12i32(<vscale x 12 x i32> %ins, i32 2)
  ret <vscale x 4 x i32> %ext
}

; Test extract of tuple passed into function
define <vscale x 4 x i32> @get_tuple3_nxv12i32_elt2(<vscale x 4 x i32> %z0, <vscale x 12 x i32> %tuple) #0 {
  ; CHECK-LABEL: get_tuple3_nxv12i32_elt2:
  ; CHECK-NEXT:  mov     z0.d, z3.d
  ; CHECK-NEXT:  ret
  %ext = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv12i32(<vscale x 12 x i32> %tuple, i32 2)
  ret <vscale x 4 x i32> %ext
}

;
; Insert into four element tuples
;

; tuple:       { tuple4.res0, tuple4.res1, tuple4.res2, tuple4.res3 }
; insert z5:   {     z5     , tuple4.res1, tuple4.res2, tuple4.res3 }
; extract z5:        ^^
define <vscale x 4 x i32> @set_tuple4_nxv16i32_elt0(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1,
                                                    <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3,
                                                    <vscale x 4 x i32> %z4, <vscale x 4 x i32> %z5) #0 {
  ; CHECK-LABEL: set_tuple4_nxv16i32_elt0:
  ; CHECK-NEXT:  mov     z0.d, z5.d
  ; CHECK-NEXT:  ret
  %tuple = tail call <vscale x 16 x i32> @llvm.aarch64.sve.tuple.create4.nxv16i32.nxv4i32(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3)
  %ins = call <vscale x 16 x i32> @llvm.aarch64.sve.tuple.set.nxv16i32.nxv4i32(<vscale x 16 x i32> %tuple, i32 0, <vscale x 4 x i32> %z5)
  %ext = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv16i32(<vscale x 16 x i32> %ins, i32 0)
  ret <vscale x 4 x i32> %ext
}

; tuple:       { tuple4.res0, tuple4.res1, tuple4.res2, tuple4.res3 }
; insert z5:   { tuple4.res0,     z5     , tuple4.res2, tuple4.res3 }
; extract z5:                     ^^
define <vscale x 4 x i32> @set_tuple4_nxv16i32_elt1(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1,
                                                    <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3,
                                                    <vscale x 4 x i32> %z4, <vscale x 4 x i32> %z5) #0 {
  ; CHECK-LABEL: set_tuple4_nxv16i32_elt1:
  ; CHECK-NEXT:  mov     z0.d, z5.d
  ; CHECK-NEXT:  ret
  %tuple = tail call <vscale x 16 x i32> @llvm.aarch64.sve.tuple.create4.nxv16i32.nxv4i32(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3)
  %ins = call <vscale x 16 x i32> @llvm.aarch64.sve.tuple.set.nxv16i32.nxv4i32(<vscale x 16 x i32> %tuple, i32 1, <vscale x 4 x i32> %z5)
  %ext = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv16i32(<vscale x 16 x i32> %ins, i32 1)
  ret <vscale x 4 x i32> %ext
}

; tuple:       { tuple4.res0, tuple4.res1, tuple4.res2, tuple4.res3 }
; insert z5:   { tuple4.res0, tuple4.res1,     z5     , tuple4.res3 }
; extract z5:                                  ^^
define <vscale x 4 x i32> @set_tuple4_nxv16i32_elt2(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1,
                                                    <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3,
                                                    <vscale x 4 x i32> %z4, <vscale x 4 x i32> %z5) #0 {
  ; CHECK-LABEL: set_tuple4_nxv16i32_elt2:
  ; CHECK-NEXT:  mov     z0.d, z5.d
  ; CHECK-NEXT:  ret
  %tuple = tail call <vscale x 16 x i32> @llvm.aarch64.sve.tuple.create4.nxv16i32.nxv4i32(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3)
  %ins = call <vscale x 16 x i32> @llvm.aarch64.sve.tuple.set.nxv16i32.nxv4i32(<vscale x 16 x i32> %tuple, i32 2, <vscale x 4 x i32> %z5)
  %ext = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv16i32(<vscale x 16 x i32> %ins, i32 2)
  ret <vscale x 4 x i32> %ext
}

; tuple:       { tuple4.res0, tuple4.res1, tuple4.res2, tuple4.res3 }
; insert z5:   { tuple4.res0, tuple4.res1, tuple4.res2,     z5      }
; extract z5:                                               ^^
define <vscale x 4 x i32> @set_tuple4_nxv16i32_elt3(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1,
                                                    <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3,
                                                    <vscale x 4 x i32> %z4, <vscale x 4 x i32> %z5) #0 {
  ; CHECK-LABEL: set_tuple4_nxv16i32_elt3:
  ; CHECK-NEXT:  mov     z0.d, z5.d
  ; CHECK-NEXT:  ret
  %tuple = tail call <vscale x 16 x i32> @llvm.aarch64.sve.tuple.create4.nxv16i32.nxv4i32(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3)
  %ins = call <vscale x 16 x i32> @llvm.aarch64.sve.tuple.set.nxv16i32.nxv4i32(<vscale x 16 x i32> %tuple, i32 3, <vscale x 4 x i32> %z5)
  %ext = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv16i32(<vscale x 16 x i32> %ins, i32 3)
  ret <vscale x 4 x i32> %ext
}

; This test checks the elements _not_ being set aren't changed.

; tuple:       { tuple4.res0, tuple4.res1, tuple4.res2, tuple4.res3 }
; insert z5:   { tuple4.res0, tuple4.res1, tuple4.res2,     z5      }
; extract z2:                                               ^^
define <vscale x 4 x i32> @set_tuple4_nxv16i32_elt3_ret_elt2(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1,
                                                             <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3,
                                                             <vscale x 4 x i32> %z4, <vscale x 4 x i32> %z5) #0 {
  ; CHECK-LABEL: set_tuple4_nxv16i32_elt3_ret_elt2:
  ; CHECK-NEXT:  mov     z0.d, z2.d
  ; CHECK-NEXT:  ret
  %tuple = tail call <vscale x 16 x i32> @llvm.aarch64.sve.tuple.create4.nxv16i32.nxv4i32(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3)
  %ins = call <vscale x 16 x i32> @llvm.aarch64.sve.tuple.set.nxv16i32.nxv4i32(<vscale x 16 x i32> %tuple, i32 3, <vscale x 4 x i32> %z5)
  %ext = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv16i32(<vscale x 16 x i32> %ins, i32 2)
  ret <vscale x 4 x i32> %ext
}

; Test extract of tuple passed into function
define <vscale x 4 x i32> @get_tuple4_nxv16i32_elt3(<vscale x 16 x i32> %tuple) #0 {
  ; CHECK-LABEL: get_tuple4_nxv16i32_elt3:
  ; CHECK-NEXT:  mov     z0.d, z3.d
  ; CHECK-NEXT:  ret
  %ext = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv16i32(<vscale x 16 x i32> %tuple, i32 3)
  ret <vscale x 4 x i32> %ext
}

attributes #0 = { nounwind }

declare <vscale x 8 x i32>  @llvm.aarch64.sve.tuple.create2.nxv8i32.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 8 x i32> @llvm.aarch64.sve.tuple.set.nxv8i32.nxv4i32(<vscale x 8 x i32>, i32, <vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv8i32(<vscale x 8 x i32>, i32)

declare <vscale x 12 x i32> @llvm.aarch64.sve.tuple.create3.nxv12i32.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 12 x i32> @llvm.aarch64.sve.tuple.set.nxv12i32.nxv4i32(<vscale x 12 x i32>, i32, <vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv12i32(<vscale x 12 x i32>, i32)

declare <vscale x 16 x i32> @llvm.aarch64.sve.tuple.create4.nxv16i32.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 16 x i32> @llvm.aarch64.sve.tuple.set.nxv16i32.nxv4i32(<vscale x 16 x i32>, i32, <vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv16i32(<vscale x 16 x i32>, i32)
