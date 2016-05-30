; RUN: llc -march aarch64 %s -o - | FileCheck %s
; RUN: llc < %s -mtriple=aarch64-unknown-unknown  -mcpu=cyclone  | FileCheck %s --check-prefix=CYCLONE
; RUN: llc -mcpu cortex-a53 -march aarch64 %s -o - | FileCheck %s --check-prefix=A53

@g0 = external global <3 x float>, align 16
@g1 = external global <3 x float>, align 4

; CHECK: ldr s[[R0:[0-9]+]], {{\[}}[[R1:x[0-9]+]]{{\]}}, #4
; CHECK: ld1{{\.?s?}} { v[[R0]]{{\.?s?}} }[1], {{\[}}[[R1]]{{\]}}
; CHECK: str d[[R0]]

define void @blam() {
  %tmp4 = getelementptr inbounds <3 x float>, <3 x float>* @g1, i64 0, i64 0
  %tmp5 = load <3 x float>, <3 x float>* @g0, align 16
  %tmp6 = extractelement <3 x float> %tmp5, i64 0
  store float %tmp6, float* %tmp4
  %tmp7 = getelementptr inbounds float, float* %tmp4, i64 1
  %tmp8 = load <3 x float>, <3 x float>* @g0, align 16
  %tmp9 = extractelement <3 x float> %tmp8, i64 1
  store float %tmp9, float* %tmp7
  ret void;
}


; PR21711 - Merge vector stores into wider vector stores.

; On Cyclone, the stores should not get merged into a 16-byte store because
; unaligned 16-byte stores are slow. This test would infinite loop when
; the fastness of unaligned accesses was not specified correctly.

define void @merge_vec_extract_stores(<4 x float> %v1, <2 x float>* %ptr) {
  %idx0 = getelementptr inbounds <2 x float>, <2 x float>* %ptr, i64 3
  %idx1 = getelementptr inbounds <2 x float>, <2 x float>* %ptr, i64 4

  %shuffle0 = shufflevector <4 x float> %v1, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  %shuffle1 = shufflevector <4 x float> %v1, <4 x float> undef, <2 x i32> <i32 2, i32 3>

  store <2 x float> %shuffle0, <2 x float>* %idx0, align 8
  store <2 x float> %shuffle1, <2 x float>* %idx1, align 8
  ret void

; CHECK-LABEL:    merge_vec_extract_stores
; CHECK:          stur   q0, [x0, #24]
; CHECK-NEXT:     ret

; CYCLONE-LABEL:    merge_vec_extract_stores
; CYCLONE:          ext   v1.16b, v0.16b, v0.16b, #8
; CYCLONE-NEXT:     str   d0, [x0, #24]
; CYCLONE-NEXT:     str   d1, [x0, #32]
; CYCLONE-NEXT:     ret
}


; PR26827 - Merge stores causes wrong dependency.
%struct1 = type { %struct1*, %struct1*, i32, i32, i16, i16, void (i32, i32, i8*)*, i8* }
@gv0 = internal unnamed_addr global i32 0, align 4
@gv1 = internal unnamed_addr global %struct1** null, align 8

define void @test(%struct1* %fde, i32 %fd, void (i32, i32, i8*)* %func, i8* %arg)  {
;CHECK-LABEL: test
entry:
;A53: mov [[DATA:w[0-9]+]], w1
;A53: str q{{[0-9]+}}, {{.*}}
;A53: str q{{[0-9]+}}, {{.*}}
;A53: str [[DATA]], {{.*}}

  %0 = bitcast %struct1* %fde to i8*
  tail call void @llvm.memset.p0i8.i64(i8* %0, i8 0, i64 40, i32 8, i1 false)
  %state = getelementptr inbounds %struct1, %struct1* %fde, i64 0, i32 4
  store i16 256, i16* %state, align 8
  %fd1 = getelementptr inbounds %struct1, %struct1* %fde, i64 0, i32 2
  store i32 %fd, i32* %fd1, align 8
  %force_eof = getelementptr inbounds %struct1, %struct1* %fde, i64 0, i32 3
  store i32 0, i32* %force_eof, align 4
  %func2 = getelementptr inbounds %struct1, %struct1* %fde, i64 0, i32 6
  store void (i32, i32, i8*)* %func, void (i32, i32, i8*)** %func2, align 8
  %arg3 = getelementptr inbounds %struct1, %struct1* %fde, i64 0, i32 7
  store i8* %arg, i8** %arg3, align 8
  %call = tail call i32 (i32, i32, ...) @fcntl(i32 %fd, i32 4, i8* %0) #6
  %1 = load i32, i32* %fd1, align 8
  %cmp.i = icmp slt i32 %1, 0
  br i1 %cmp.i, label %if.then.i, label %while.body.i.preheader
if.then.i:
  unreachable

while.body.i.preheader:
  %2 = load i32, i32* @gv0, align 4
  %3 = icmp eq i32* %fd1, @gv0
  br i1 %3, label %while.body.i.split, label %while.body.i.split.ver.us.preheader

while.body.i.split.ver.us.preheader:
  br label %while.body.i.split.ver.us

while.body.i.split.ver.us:
  %.reg2mem21.0 = phi i32 [ %mul.i.ver.us, %while.body.i.split.ver.us ], [ %2, %while.body.i.split.ver.us.preheader ]
  %mul.i.ver.us = shl nsw i32 %.reg2mem21.0, 1
  %4 = icmp sgt i32 %mul.i.ver.us, %1
  br i1 %4, label %while.end.i, label %while.body.i.split.ver.us

while.body.i.split:
  br label %while.body.i.split

while.end.i:
  %call.i = tail call i8* @foo()
  store i8* %call.i, i8** bitcast (%struct1*** @gv1 to i8**), align 8
  br label %exit

exit:
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1)
declare i32 @fcntl(i32, i32, ...)
declare noalias i8* @foo()
