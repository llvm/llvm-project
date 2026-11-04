;; Check HVX vectorization.
; RUN: llc -mtriple hexagon < %s | FileCheck %s --check-prefixes=CHECK,CHECK-HVX

;; Check that there is no failure when compiling to scalar code, don't check the output.
; RUN: llc -mtriple hexagon -mattr=-hvx,-hvxv73,-hvx-length128b < %s | FileCheck %s --check-prefixes=CHECK,CHECK-NO-HVX

define <16 x i32> @partial_reduce_uu_64(<16 x i32> %acc, <64 x i8> %x, <64 x i8> %y) #0 {
; CHECK-LABEL: partial_reduce_uu_64:
; CHECK-HVX:    v0.uw += vrmpy(v1.ub,v2.ub)
; CHECK-NO-HVX: {{r[0-9]+}} += mpyi
  %x.ext = zext <64 x i8> %x to <64 x i32>
  %y.ext = zext <64 x i8> %y to <64 x i32>
  %m = mul nuw nsw <64 x i32> %x.ext, %y.ext
  %partial.reduce = tail call <16 x i32> @llvm.vector.partial.reduce.add.v32i32.v128i32(<16 x i32> %acc, <64 x i32> %m)
  ret <16 x i32> %partial.reduce
}

define <16 x i32> @partial_reduce_su_64(<16 x i32> %acc, <64 x i8> %x, <64 x i8> %y) #0 {
; CHECK-LABEL: partial_reduce_su_64:
; CHECK-HVX:    v0.w += vrmpy(v2.ub,v1.b)
; CHECK-NO-HVX: {{r[0-9]+}} += mpyi
  %x.ext = sext <64 x i8> %x to <64 x i32>
  %y.ext = zext <64 x i8> %y to <64 x i32>
  %m = mul nuw nsw <64 x i32> %x.ext, %y.ext
  %partial.reduce = tail call <16 x i32> @llvm.vector.partial.reduce.add.v32i32.v128i32(<16 x i32> %acc, <64 x i32> %m)
  ret <16 x i32> %partial.reduce
}

define <16 x i32> @partial_reduce_us_64(<16 x i32> %acc, <64 x i8> %x, <64 x i8> %y) #0 {
; CHECK-LABEL: partial_reduce_us_64:
; CHECK-HVX:    v0.w += vrmpy(v1.ub,v2.b)
; CHECK-NO-HVX: {{r[0-9]+}} += mpyi
  %x.ext = zext <64 x i8> %x to <64 x i32>
  %y.ext = sext <64 x i8> %y to <64 x i32>
  %m = mul nuw nsw <64 x i32> %x.ext, %y.ext
  %partial.reduce = tail call <16 x i32> @llvm.vector.partial.reduce.add.v32i32.v128i32(<16 x i32> %acc, <64 x i32> %m)
  ret <16 x i32> %partial.reduce
}

define <16 x i32> @partial_reduce_ss_64(<16 x i32> %acc, <64 x i8> %x, <64 x i8> %y) #0 {
; CHECK-LABEL: partial_reduce_ss_64:
; CHECK-HVX:    v0.w += vrmpy(v1.b,v2.b)
; CHECK-NO-HVX: {{r[0-9]+}} += mpyi
  %x.ext = sext <64 x i8> %x to <64 x i32>
  %y.ext = sext <64 x i8> %y to <64 x i32>
  %m = mul nuw nsw <64 x i32> %x.ext, %y.ext
  %partial.reduce = tail call <16 x i32> @llvm.vector.partial.reduce.add.v32i32.v128i32(<16 x i32> %acc, <64 x i32> %m)
  ret <16 x i32> %partial.reduce
}

define <32 x i32> @partial_reduce_uu_128(<32 x i32> %acc, <128 x i8> %x, <128 x i8> %y) #1 {
; CHECK-LABEL: partial_reduce_uu_128:
; CHECK-HVX:    v0.uw += vrmpy(v1.ub,v2.ub)
; CHECK-NO-HVX: {{r[0-9]+}} += mpyi
  %x.ext = zext <128 x i8> %x to <128 x i32>
  %y.ext = zext <128 x i8> %y to <128 x i32>
  %m = mul nuw nsw <128 x i32> %x.ext, %y.ext
  %partial.reduce = tail call <32 x i32> @llvm.vector.partial.reduce.add.v32i32.v128i32(<32 x i32> %acc, <128 x i32> %m)
  ret <32 x i32> %partial.reduce
}

define <32 x i32> @partial_reduce_su_128(<32 x i32> %acc, <128 x i8> %x, <128 x i8> %y) #1 {
; CHECK-LABEL: partial_reduce_su_128:
; CHECK-HVX:    v0.w += vrmpy(v2.ub,v1.b)
; CHECK-NO-HVX: {{r[0-9]+}} += mpyi
  %x.ext = sext <128 x i8> %x to <128 x i32>
  %y.ext = zext <128 x i8> %y to <128 x i32>
  %m = mul nuw nsw <128 x i32> %x.ext, %y.ext
  %partial.reduce = tail call <32 x i32> @llvm.vector.partial.reduce.add.v32i32.v128i32(<32 x i32> %acc, <128 x i32> %m)
  ret <32 x i32> %partial.reduce
}

define <32 x i32> @partial_reduce_us_128(<32 x i32> %acc, <128 x i8> %x, <128 x i8> %y) #1 {
; CHECK-LABEL: partial_reduce_us_128:
; CHECK-HVX:    v0.w += vrmpy(v1.ub,v2.b)
; CHECK-NO-HVX: {{r[0-9]+}} += mpyi
  %x.ext = zext <128 x i8> %x to <128 x i32>
  %y.ext = sext <128 x i8> %y to <128 x i32>
  %m = mul nuw nsw <128 x i32> %x.ext, %y.ext
  %partial.reduce = tail call <32 x i32> @llvm.vector.partial.reduce.add.v32i32.v128i32(<32 x i32> %acc, <128 x i32> %m)
  ret <32 x i32> %partial.reduce
}

define <32 x i32> @partial_reduce_ss_128(<32 x i32> %acc, <128 x i8> %x, <128 x i8> %y) #1 {
; CHECK-LABEL: partial_reduce_ss_128:
; CHECK-HVX:    v0.w += vrmpy(v1.b,v2.b)
; CHECK-NO-HVX: {{r[0-9]+}} += mpyi
  %x.ext = sext <128 x i8> %x to <128 x i32>
  %y.ext = sext <128 x i8> %y to <128 x i32>
  %m = mul nuw nsw <128 x i32> %x.ext, %y.ext
  %partial.reduce = tail call <32 x i32> @llvm.vector.partial.reduce.add.v32i32.v128i32(<32 x i32> %acc, <128 x i32> %m)
  ret <32 x i32> %partial.reduce
}

;; Multiple-size inputs, same output size.
define <32 x i32> @partial_reduce_uu_32xi32_256xi8(<32 x i32> %acc, <256 x i8> %x, <256 x i8> %y) #1 {
; CHECK-LABEL: partial_reduce_uu_32xi32_256xi8:
; CHECK-HVX:    [[R1:v[0-9]+]].uw += vrmpy({{v[0-9]+}}.ub,{{v[0-9]+}}.ub)
; CHECK-HVX:    [[R2:v[0-9]+]].uw += vrmpy({{v[0-9]+}}.ub,{{v[0-9]+}}.ub)
; CHECK-HVX:    [[R3:v[0-9]+]].w = vadd(v0.w,[[R1]].w)
; CHECK-HVX:    v0.w = vadd([[R2]].w,[[R3]].w)
; CHECK-NO-HVX: {{r[0-9]+}} += mpyi
  %x.ext = zext <256 x i8> %x to <256 x i32>
  %y.ext = zext <256 x i8> %y to <256 x i32>
  %m = mul nuw nsw <256 x i32> %x.ext, %y.ext
  %partial.reduce = tail call <32 x i32> @llvm.vector.partial.reduce.add.v32i32.v256i32(<32 x i32> %acc, <256 x i32> %m)
  ret <32 x i32> %partial.reduce
}

define <32 x i32> @partial_reduce_uu_32xi32_1024xi8(<32 x i32> %acc, <1024 x i8> %x, <1024 x i8> %y) #1 {
; CHECK-LABEL: partial_reduce_uu_32xi32_1024xi8:
; CHECK-HVX-DAG: vrmpy
; CHECK-HVX-DAG: vadd
; CHECK-HVX-DAG: vrmpy
; CHECK-HVX-DAG: vadd
; CHECK-HVX-DAG: vrmpy
; CHECK-HVX-DAG: vadd
; CHECK-HVX-DAG: vrmpy
; CHECK-HVX-DAG: vadd
; CHECK-HVX-DAG: vrmpy
; CHECK-HVX-DAG: vadd
; CHECK-HVX-DAG: vrmpy
; CHECK-HVX-DAG: vadd
; CHECK-HVX-DAG: vrmpy
; CHECK-HVX-DAG: vadd
; CHECK-HVX-DAG: vrmpy
; CHECK-HVX-DAG: vadd
; CHECK-NO-HVX: {{r[0-9]+}} += mpyi
  %x.ext = zext <1024 x i8> %x to <1024 x i32>
  %y.ext = zext <1024 x i8> %y to <1024 x i32>
  %m = mul nuw nsw <1024 x i32> %x.ext, %y.ext
  %partial.reduce = tail call <32 x i32> @llvm.vector.partial.reduce.add.v32i32.v1024i32(<32 x i32> %acc, <1024 x i32> %m)
  ret <32 x i32> %partial.reduce
}

define <256 x i32> @partial_reduce_uu_64xi32_1024xi8(<256 x i32> %acc, <1024 x i8> %x, <1024 x i8> %y) #1 {
; CHECK-LABEL: partial_reduce_uu_64xi32_1024xi8:
; CHECK-HVX-COUNT-8: vrmpy
; CHECK-HVX-NOT: vadd
; CHECK-NO-HVX: {{r[0-9]+}} += mpyi
; CHECK-HVX: dealloc_return
  %x.ext = zext <1024 x i8> %x to <1024 x i32>
  %y.ext = zext <1024 x i8> %y to <1024 x i32>
  %m = mul nuw nsw <1024 x i32> %x.ext, %y.ext
  %partial.reduce = tail call <256 x i32> @llvm.vector.partial.reduce.add.v32i32.v1024i32(<256 x i32> %acc, <1024 x i32> %m)
  ret <256 x i32> %partial.reduce
}

;; Check for vector size that do not match an available vrmpy (2x reduction).
define <64 x i32> @partial_reduce_unsupported(<64 x i32> %acc, <128 x i8> %x, <128 x i8> %y) #1 {
; CHECK-LABEL: partial_reduce_unsupported:
; CHECK-HVX: vmpy
; CHECK-HVX: vadd
  %x.ext = zext <128 x i8> %x to <128 x i32>
  %y.ext = zext <128 x i8> %y to <128 x i32>
  %m = mul nuw nsw <128 x i32> %x.ext, %y.ext
  %partial.reduce = tail call <64 x i32> @llvm.vector.partial.reduce.add.v64i32.v128i32(<64 x i32> %acc, <128 x i32> %m)
  ret <64 x i32> %partial.reduce
}

attributes #0 = { nounwind "target-cpu"="hexagonv73" "target-features"="+hvxv73,+hvx-length64b" }
attributes #1 = { nounwind "target-cpu"="hexagonv73" "target-features"="+hvxv73,+hvx-length128b" }
