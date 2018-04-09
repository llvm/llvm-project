; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -verify-machineinstrs | FileCheck %s

; rdar://9428579

%type1 = type { <16 x i8> }
%type2 = type { <8 x i8> }
%type3 = type { <4 x i16> }


define hidden fastcc void @t1(%type1** %argtable) nounwind {
entry:
; CHECK-LABEL: t1:
; CHECK: ldr x[[REG:[0-9]+]], [x0]
; CHECK: str q0, [x[[REG]]]
  %tmp1 = load %type1*, %type1** %argtable, align 8
  %tmp2 = getelementptr inbounds %type1, %type1* %tmp1, i64 0, i32 0
  store <16 x i8> zeroinitializer, <16 x i8>* %tmp2, align 16
  ret void
}

define hidden fastcc void @t2(%type2** %argtable) nounwind {
entry:
; CHECK-LABEL: t2:
; CHECK: ldr x[[REG:[0-9]+]], [x0]
; CHECK: str d0, [x[[REG]]]
  %tmp1 = load %type2*, %type2** %argtable, align 8
  %tmp2 = getelementptr inbounds %type2, %type2* %tmp1, i64 0, i32 0
  store <8 x i8> zeroinitializer, <8 x i8>* %tmp2, align 8
  ret void
}

; add a bunch of tests for rdar://11246289

@globalArray64x2 = common global <2 x i64>* null, align 8
@globalArray32x4 = common global <4 x i32>* null, align 8
@globalArray16x8 = common global <8 x i16>* null, align 8
@globalArray8x16 = common global <16 x i8>* null, align 8
@globalArray64x1 = common global <1 x i64>* null, align 8
@globalArray32x2 = common global <2 x i32>* null, align 8
@globalArray16x4 = common global <4 x i16>* null, align 8
@globalArray8x8 = common global <8 x i8>* null, align 8
@floatglobalArray64x2 = common global <2 x double>* null, align 8
@floatglobalArray32x4 = common global <4 x float>* null, align 8
@floatglobalArray64x1 = common global <1 x double>* null, align 8
@floatglobalArray32x2 = common global <2 x float>* null, align 8

define void @fct1_64x2(<2 x i64>* nocapture %array, i64 %offset) nounwind ssp {
entry:
; CHECK-LABEL: fct1_64x2:
; CHECK: lsl [[SHIFTEDOFFSET:x[0-9]+]], x1, #4
; CHECK: ldr [[DEST:q[0-9]+]], [x0, [[SHIFTEDOFFSET]]
; CHECK: ldr [[BASE:x[0-9]+]],
; CHECK: str [[DEST]], {{\[}}[[BASE]], [[SHIFTEDOFFSET]]]
  %arrayidx = getelementptr inbounds <2 x i64>, <2 x i64>* %array, i64 %offset
  %tmp = load <2 x i64>, <2 x i64>* %arrayidx, align 16
  %tmp1 = load <2 x i64>*, <2 x i64>** @globalArray64x2, align 8
  %arrayidx1 = getelementptr inbounds <2 x i64>, <2 x i64>* %tmp1, i64 %offset
  store <2 x i64> %tmp, <2 x i64>* %arrayidx1, align 16
  ret void
}

define void @fct2_64x2(<2 x i64>* nocapture %array) nounwind ssp {
entry:
; CHECK-LABEL: fct2_64x2:
; CHECK: ldr [[DEST:q[0-9]+]], [x0, #48]
; CHECK: ldr [[BASE:x[0-9]+]],
; CHECK: str [[DEST]], {{\[}}[[BASE]], #80]
  %arrayidx = getelementptr inbounds <2 x i64>, <2 x i64>* %array, i64 3
  %tmp = load <2 x i64>, <2 x i64>* %arrayidx, align 16
  %tmp1 = load <2 x i64>*, <2 x i64>** @globalArray64x2, align 8
  %arrayidx1 = getelementptr inbounds <2 x i64>, <2 x i64>* %tmp1, i64 5
  store <2 x i64> %tmp, <2 x i64>* %arrayidx1, align 16
  ret void
}

define void @fct1_32x4(<4 x i32>* nocapture %array, i64 %offset) nounwind ssp {
entry:
; CHECK-LABEL: fct1_32x4:
; CHECK: lsl [[SHIFTEDOFFSET:x[0-9]+]], x1, #4
; CHECK: ldr [[DEST:q[0-9]+]], [x0, [[SHIFTEDOFFSET]]]
; CHECK: ldr [[BASE:x[0-9]+]],
; CHECK: str [[DEST]], {{\[}}[[BASE]], [[SHIFTEDOFFSET]]]
  %arrayidx = getelementptr inbounds <4 x i32>, <4 x i32>* %array, i64 %offset
  %tmp = load <4 x i32>, <4 x i32>* %arrayidx, align 16
  %tmp1 = load <4 x i32>*, <4 x i32>** @globalArray32x4, align 8
  %arrayidx1 = getelementptr inbounds <4 x i32>, <4 x i32>* %tmp1, i64 %offset
  store <4 x i32> %tmp, <4 x i32>* %arrayidx1, align 16
  ret void
}

define void @fct2_32x4(<4 x i32>* nocapture %array) nounwind ssp {
entry:
; CHECK-LABEL: fct2_32x4:
; CHECK: ldr [[DEST:q[0-9]+]], [x0, #48]
; CHECK: ldr [[BASE:x[0-9]+]],
; CHECK: str [[DEST]], {{\[}}[[BASE]], #80]
  %arrayidx = getelementptr inbounds <4 x i32>, <4 x i32>* %array, i64 3
  %tmp = load <4 x i32>, <4 x i32>* %arrayidx, align 16
  %tmp1 = load <4 x i32>*, <4 x i32>** @globalArray32x4, align 8
  %arrayidx1 = getelementptr inbounds <4 x i32>, <4 x i32>* %tmp1, i64 5
  store <4 x i32> %tmp, <4 x i32>* %arrayidx1, align 16
  ret void
}

define void @fct1_16x8(<8 x i16>* nocapture %array, i64 %offset) nounwind ssp {
entry:
; CHECK-LABEL: fct1_16x8:
; CHECK: lsl [[SHIFTEDOFFSET:x[0-9]+]], x1, #4
; CHECK: ldr [[DEST:q[0-9]+]], [x0, [[SHIFTEDOFFSET]]]
; CHECK: ldr [[BASE:x[0-9]+]],
; CHECK: str [[DEST]], {{\[}}[[BASE]], [[SHIFTEDOFFSET]]]
  %arrayidx = getelementptr inbounds <8 x i16>, <8 x i16>* %array, i64 %offset
  %tmp = load <8 x i16>, <8 x i16>* %arrayidx, align 16
  %tmp1 = load <8 x i16>*, <8 x i16>** @globalArray16x8, align 8
  %arrayidx1 = getelementptr inbounds <8 x i16>, <8 x i16>* %tmp1, i64 %offset
  store <8 x i16> %tmp, <8 x i16>* %arrayidx1, align 16
  ret void
}

define void @fct2_16x8(<8 x i16>* nocapture %array) nounwind ssp {
entry:
; CHECK-LABEL: fct2_16x8:
; CHECK: ldr [[DEST:q[0-9]+]], [x0, #48]
; CHECK: ldr [[BASE:x[0-9]+]],
; CHECK: str [[DEST]], {{\[}}[[BASE]], #80]
  %arrayidx = getelementptr inbounds <8 x i16>, <8 x i16>* %array, i64 3
  %tmp = load <8 x i16>, <8 x i16>* %arrayidx, align 16
  %tmp1 = load <8 x i16>*, <8 x i16>** @globalArray16x8, align 8
  %arrayidx1 = getelementptr inbounds <8 x i16>, <8 x i16>* %tmp1, i64 5
  store <8 x i16> %tmp, <8 x i16>* %arrayidx1, align 16
  ret void
}

define void @fct1_8x16(<16 x i8>* nocapture %array, i64 %offset) nounwind ssp {
entry:
; CHECK-LABEL: fct1_8x16:
; CHECK: lsl [[SHIFTEDOFFSET:x[0-9]+]], x1, #4
; CHECK: ldr [[DEST:q[0-9]+]], [x0, [[SHIFTEDOFFSET]]]
; CHECK: ldr [[BASE:x[0-9]+]],
; CHECK: str [[DEST]], {{\[}}[[BASE]], [[SHIFTEDOFFSET]]]
  %arrayidx = getelementptr inbounds <16 x i8>, <16 x i8>* %array, i64 %offset
  %tmp = load <16 x i8>, <16 x i8>* %arrayidx, align 16
  %tmp1 = load <16 x i8>*, <16 x i8>** @globalArray8x16, align 8
  %arrayidx1 = getelementptr inbounds <16 x i8>, <16 x i8>* %tmp1, i64 %offset
  store <16 x i8> %tmp, <16 x i8>* %arrayidx1, align 16
  ret void
}

define void @fct2_8x16(<16 x i8>* nocapture %array) nounwind ssp {
entry:
; CHECK-LABEL: fct2_8x16:
; CHECK: ldr [[DEST:q[0-9]+]], [x0, #48]
; CHECK: ldr [[BASE:x[0-9]+]],
; CHECK: str [[DEST]], {{\[}}[[BASE]], #80]
  %arrayidx = getelementptr inbounds <16 x i8>, <16 x i8>* %array, i64 3
  %tmp = load <16 x i8>, <16 x i8>* %arrayidx, align 16
  %tmp1 = load <16 x i8>*, <16 x i8>** @globalArray8x16, align 8
  %arrayidx1 = getelementptr inbounds <16 x i8>, <16 x i8>* %tmp1, i64 5
  store <16 x i8> %tmp, <16 x i8>* %arrayidx1, align 16
  ret void
}

define void @fct1_64x1(<1 x i64>* nocapture %array, i64 %offset) nounwind ssp {
entry:
; CHECK-LABEL: fct1_64x1:
; CHECK: lsl [[SHIFTEDOFFSET:x[0-9]+]], x1, #3
; CHECK: ldr [[DEST:d[0-9]+]], [x0, [[SHIFTEDOFFSET]]]
; CHECK: ldr [[BASE:x[0-9]+]],
; CHECK: str [[DEST]], {{\[}}[[BASE]], [[SHIFTEDOFFSET]]]
  %arrayidx = getelementptr inbounds <1 x i64>, <1 x i64>* %array, i64 %offset
  %tmp = load <1 x i64>, <1 x i64>* %arrayidx, align 8
  %tmp1 = load <1 x i64>*, <1 x i64>** @globalArray64x1, align 8
  %arrayidx1 = getelementptr inbounds <1 x i64>, <1 x i64>* %tmp1, i64 %offset
  store <1 x i64> %tmp, <1 x i64>* %arrayidx1, align 8
  ret void
}

define void @fct2_64x1(<1 x i64>* nocapture %array) nounwind ssp {
entry:
; CHECK-LABEL: fct2_64x1:
; CHECK: ldr [[DEST:d[0-9]+]], [x0, #24]
; CHECK: ldr [[BASE:x[0-9]+]],
; CHECK: str [[DEST]], {{\[}}[[BASE]], #40]
  %arrayidx = getelementptr inbounds <1 x i64>, <1 x i64>* %array, i64 3
  %tmp = load <1 x i64>, <1 x i64>* %arrayidx, align 8
  %tmp1 = load <1 x i64>*, <1 x i64>** @globalArray64x1, align 8
  %arrayidx1 = getelementptr inbounds <1 x i64>, <1 x i64>* %tmp1, i64 5
  store <1 x i64> %tmp, <1 x i64>* %arrayidx1, align 8
  ret void
}

define void @fct1_32x2(<2 x i32>* nocapture %array, i64 %offset) nounwind ssp {
entry:
; CHECK-LABEL: fct1_32x2:
; CHECK: lsl [[SHIFTEDOFFSET:x[0-9]+]], x1, #3
; CHECK: ldr [[DEST:d[0-9]+]], [x0, [[SHIFTEDOFFSET]]]
; CHECK: ldr [[BASE:x[0-9]+]],
; CHECK: str [[DEST]], {{\[}}[[BASE]], [[SHIFTEDOFFSET]]]
  %arrayidx = getelementptr inbounds <2 x i32>, <2 x i32>* %array, i64 %offset
  %tmp = load <2 x i32>, <2 x i32>* %arrayidx, align 8
  %tmp1 = load <2 x i32>*, <2 x i32>** @globalArray32x2, align 8
  %arrayidx1 = getelementptr inbounds <2 x i32>, <2 x i32>* %tmp1, i64 %offset
  store <2 x i32> %tmp, <2 x i32>* %arrayidx1, align 8
  ret void
}

define void @fct2_32x2(<2 x i32>* nocapture %array) nounwind ssp {
entry:
; CHECK-LABEL: fct2_32x2:
; CHECK: ldr [[DEST:d[0-9]+]], [x0, #24]
; CHECK: ldr [[BASE:x[0-9]+]],
; CHECK: str [[DEST]], {{\[}}[[BASE]], #40]
  %arrayidx = getelementptr inbounds <2 x i32>, <2 x i32>* %array, i64 3
  %tmp = load <2 x i32>, <2 x i32>* %arrayidx, align 8
  %tmp1 = load <2 x i32>*, <2 x i32>** @globalArray32x2, align 8
  %arrayidx1 = getelementptr inbounds <2 x i32>, <2 x i32>* %tmp1, i64 5
  store <2 x i32> %tmp, <2 x i32>* %arrayidx1, align 8
  ret void
}

define void @fct1_16x4(<4 x i16>* nocapture %array, i64 %offset) nounwind ssp {
entry:
; CHECK-LABEL: fct1_16x4:
; CHECK: lsl [[SHIFTEDOFFSET:x[0-9]+]], x1, #3
; CHECK: ldr [[DEST:d[0-9]+]], [x0, [[SHIFTEDOFFSET]]]
; CHECK: ldr [[BASE:x[0-9]+]],
; CHECK: str [[DEST]], {{\[}}[[BASE]], [[SHIFTEDOFFSET]]]
  %arrayidx = getelementptr inbounds <4 x i16>, <4 x i16>* %array, i64 %offset
  %tmp = load <4 x i16>, <4 x i16>* %arrayidx, align 8
  %tmp1 = load <4 x i16>*, <4 x i16>** @globalArray16x4, align 8
  %arrayidx1 = getelementptr inbounds <4 x i16>, <4 x i16>* %tmp1, i64 %offset
  store <4 x i16> %tmp, <4 x i16>* %arrayidx1, align 8
  ret void
}

define void @fct2_16x4(<4 x i16>* nocapture %array) nounwind ssp {
entry:
; CHECK-LABEL: fct2_16x4:
; CHECK: ldr [[DEST:d[0-9]+]], [x0, #24]
; CHECK: ldr [[BASE:x[0-9]+]],
; CHECK: str [[DEST]], {{\[}}[[BASE]], #40]
  %arrayidx = getelementptr inbounds <4 x i16>, <4 x i16>* %array, i64 3
  %tmp = load <4 x i16>, <4 x i16>* %arrayidx, align 8
  %tmp1 = load <4 x i16>*, <4 x i16>** @globalArray16x4, align 8
  %arrayidx1 = getelementptr inbounds <4 x i16>, <4 x i16>* %tmp1, i64 5
  store <4 x i16> %tmp, <4 x i16>* %arrayidx1, align 8
  ret void
}

define void @fct1_8x8(<8 x i8>* nocapture %array, i64 %offset) nounwind ssp {
entry:
; CHECK-LABEL: fct1_8x8:
; CHECK: lsl [[SHIFTEDOFFSET:x[0-9]+]], x1, #3
; CHECK: ldr [[DEST:d[0-9]+]], [x0, [[SHIFTEDOFFSET]]]
; CHECK: ldr [[BASE:x[0-9]+]],
; CHECK: str [[DEST]], {{\[}}[[BASE]], [[SHIFTEDOFFSET]]]
  %arrayidx = getelementptr inbounds <8 x i8>, <8 x i8>* %array, i64 %offset
  %tmp = load <8 x i8>, <8 x i8>* %arrayidx, align 8
  %tmp1 = load <8 x i8>*, <8 x i8>** @globalArray8x8, align 8
  %arrayidx1 = getelementptr inbounds <8 x i8>, <8 x i8>* %tmp1, i64 %offset
  store <8 x i8> %tmp, <8 x i8>* %arrayidx1, align 8
  ret void
}

; Add a bunch of tests for rdar://13258794: Match LDUR/STUR for D and Q
; registers for unscaled vector accesses

define <1 x i64> @fct0(i8* %str) nounwind readonly ssp {
entry:
; CHECK-LABEL: fct0:
; CHECK: ldur {{d[0-9]+}}, [{{x[0-9]+}}, #3]
  %p = getelementptr inbounds i8, i8* %str, i64 3
  %q = bitcast i8* %p to <1 x i64>*
  %0 = load <1 x i64>, <1 x i64>* %q, align 8
  ret <1 x i64> %0
}

define <2 x i32> @fct1(i8* %str) nounwind readonly ssp {
entry:
; CHECK-LABEL: fct1:
; CHECK: ldur {{d[0-9]+}}, [{{x[0-9]+}}, #3]
  %p = getelementptr inbounds i8, i8* %str, i64 3
  %q = bitcast i8* %p to <2 x i32>*
  %0 = load <2 x i32>, <2 x i32>* %q, align 8
  ret <2 x i32> %0
}

define <4 x i16> @fct2(i8* %str) nounwind readonly ssp {
entry:
; CHECK-LABEL: fct2:
; CHECK: ldur {{d[0-9]+}}, [{{x[0-9]+}}, #3]
  %p = getelementptr inbounds i8, i8* %str, i64 3
  %q = bitcast i8* %p to <4 x i16>*
  %0 = load <4 x i16>, <4 x i16>* %q, align 8
  ret <4 x i16> %0
}

define <8 x i8> @fct3(i8* %str) nounwind readonly ssp {
entry:
; CHECK-LABEL: fct3:
; CHECK: ldur {{d[0-9]+}}, [{{x[0-9]+}}, #3]
  %p = getelementptr inbounds i8, i8* %str, i64 3
  %q = bitcast i8* %p to <8 x i8>*
  %0 = load <8 x i8>, <8 x i8>* %q, align 8
  ret <8 x i8> %0
}

define <2 x i64> @fct4(i8* %str) nounwind readonly ssp {
entry:
; CHECK-LABEL: fct4:
; CHECK: ldur {{q[0-9]+}}, [{{x[0-9]+}}, #3]
  %p = getelementptr inbounds i8, i8* %str, i64 3
  %q = bitcast i8* %p to <2 x i64>*
  %0 = load <2 x i64>, <2 x i64>* %q, align 16
  ret <2 x i64> %0
}

define <4 x i32> @fct5(i8* %str) nounwind readonly ssp {
entry:
; CHECK-LABEL: fct5:
; CHECK: ldur {{q[0-9]+}}, [{{x[0-9]+}}, #3]
  %p = getelementptr inbounds i8, i8* %str, i64 3
  %q = bitcast i8* %p to <4 x i32>*
  %0 = load <4 x i32>, <4 x i32>* %q, align 16
  ret <4 x i32> %0
}

define <8 x i16> @fct6(i8* %str) nounwind readonly ssp {
entry:
; CHECK-LABEL: fct6:
; CHECK: ldur {{q[0-9]+}}, [{{x[0-9]+}}, #3]
  %p = getelementptr inbounds i8, i8* %str, i64 3
  %q = bitcast i8* %p to <8 x i16>*
  %0 = load <8 x i16>, <8 x i16>* %q, align 16
  ret <8 x i16> %0
}

define <16 x i8> @fct7(i8* %str) nounwind readonly ssp {
entry:
; CHECK-LABEL: fct7:
; CHECK: ldur {{q[0-9]+}}, [{{x[0-9]+}}, #3]
  %p = getelementptr inbounds i8, i8* %str, i64 3
  %q = bitcast i8* %p to <16 x i8>*
  %0 = load <16 x i8>, <16 x i8>* %q, align 16
  ret <16 x i8> %0
}

define void @fct8(i8* %str) nounwind ssp {
entry:
; CHECK-LABEL: fct8:
; CHECK: ldur [[DESTREG:d[0-9]+]], {{\[}}[[BASEREG:x[0-9]+]], #3]
; CHECK: stur [[DESTREG]], {{\[}}[[BASEREG]], #4]
  %p = getelementptr inbounds i8, i8* %str, i64 3
  %q = bitcast i8* %p to <1 x i64>*
  %0 = load <1 x i64>, <1 x i64>* %q, align 8
  %p2 = getelementptr inbounds i8, i8* %str, i64 4
  %q2 = bitcast i8* %p2 to <1 x i64>*
  store <1 x i64> %0, <1 x i64>* %q2, align 8
  ret void
}

define void @fct9(i8* %str) nounwind ssp {
entry:
; CHECK-LABEL: fct9:
; CHECK: ldur [[DESTREG:d[0-9]+]], {{\[}}[[BASEREG:x[0-9]+]], #3]
; CHECK: stur [[DESTREG]], {{\[}}[[BASEREG]], #4]
  %p = getelementptr inbounds i8, i8* %str, i64 3
  %q = bitcast i8* %p to <2 x i32>*
  %0 = load <2 x i32>, <2 x i32>* %q, align 8
  %p2 = getelementptr inbounds i8, i8* %str, i64 4
  %q2 = bitcast i8* %p2 to <2 x i32>*
  store <2 x i32> %0, <2 x i32>* %q2, align 8
  ret void
}

define void @fct10(i8* %str) nounwind ssp {
entry:
; CHECK-LABEL: fct10:
; CHECK: ldur [[DESTREG:d[0-9]+]], {{\[}}[[BASEREG:x[0-9]+]], #3]
; CHECK: stur [[DESTREG]], {{\[}}[[BASEREG]], #4]
  %p = getelementptr inbounds i8, i8* %str, i64 3
  %q = bitcast i8* %p to <4 x i16>*
  %0 = load <4 x i16>, <4 x i16>* %q, align 8
  %p2 = getelementptr inbounds i8, i8* %str, i64 4
  %q2 = bitcast i8* %p2 to <4 x i16>*
  store <4 x i16> %0, <4 x i16>* %q2, align 8
  ret void
}

define void @fct11(i8* %str) nounwind ssp {
entry:
; CHECK-LABEL: fct11:
; CHECK: ldur [[DESTREG:d[0-9]+]], {{\[}}[[BASEREG:x[0-9]+]], #3]
; CHECK: stur [[DESTREG]], {{\[}}[[BASEREG]], #4]
  %p = getelementptr inbounds i8, i8* %str, i64 3
  %q = bitcast i8* %p to <8 x i8>*
  %0 = load <8 x i8>, <8 x i8>* %q, align 8
  %p2 = getelementptr inbounds i8, i8* %str, i64 4
  %q2 = bitcast i8* %p2 to <8 x i8>*
  store <8 x i8> %0, <8 x i8>* %q2, align 8
  ret void
}

define void @fct12(i8* %str) nounwind ssp {
entry:
; CHECK-LABEL: fct12:
; CHECK: ldur [[DESTREG:q[0-9]+]], {{\[}}[[BASEREG:x[0-9]+]], #3]
; CHECK: stur [[DESTREG]], {{\[}}[[BASEREG]], #4]
  %p = getelementptr inbounds i8, i8* %str, i64 3
  %q = bitcast i8* %p to <2 x i64>*
  %0 = load <2 x i64>, <2 x i64>* %q, align 16
  %p2 = getelementptr inbounds i8, i8* %str, i64 4
  %q2 = bitcast i8* %p2 to <2 x i64>*
  store <2 x i64> %0, <2 x i64>* %q2, align 16
  ret void
}

define void @fct13(i8* %str) nounwind ssp {
entry:
; CHECK-LABEL: fct13:
; CHECK: ldur [[DESTREG:q[0-9]+]], {{\[}}[[BASEREG:x[0-9]+]], #3]
; CHECK: stur [[DESTREG]], {{\[}}[[BASEREG]], #4]
  %p = getelementptr inbounds i8, i8* %str, i64 3
  %q = bitcast i8* %p to <4 x i32>*
  %0 = load <4 x i32>, <4 x i32>* %q, align 16
  %p2 = getelementptr inbounds i8, i8* %str, i64 4
  %q2 = bitcast i8* %p2 to <4 x i32>*
  store <4 x i32> %0, <4 x i32>* %q2, align 16
  ret void
}

define void @fct14(i8* %str) nounwind ssp {
entry:
; CHECK-LABEL: fct14:
; CHECK: ldur [[DESTREG:q[0-9]+]], {{\[}}[[BASEREG:x[0-9]+]], #3]
; CHECK: stur [[DESTREG]], {{\[}}[[BASEREG]], #4]
  %p = getelementptr inbounds i8, i8* %str, i64 3
  %q = bitcast i8* %p to <8 x i16>*
  %0 = load <8 x i16>, <8 x i16>* %q, align 16
  %p2 = getelementptr inbounds i8, i8* %str, i64 4
  %q2 = bitcast i8* %p2 to <8 x i16>*
  store <8 x i16> %0, <8 x i16>* %q2, align 16
  ret void
}

define void @fct15(i8* %str) nounwind ssp {
entry:
; CHECK-LABEL: fct15:
; CHECK: ldur [[DESTREG:q[0-9]+]], {{\[}}[[BASEREG:x[0-9]+]], #3]
; CHECK: stur [[DESTREG]], {{\[}}[[BASEREG]], #4]
  %p = getelementptr inbounds i8, i8* %str, i64 3
  %q = bitcast i8* %p to <16 x i8>*
  %0 = load <16 x i8>, <16 x i8>* %q, align 16
  %p2 = getelementptr inbounds i8, i8* %str, i64 4
  %q2 = bitcast i8* %p2 to <16 x i8>*
  store <16 x i8> %0, <16 x i8>* %q2, align 16
  ret void
}

; Check the building of vector from a single loaded value.
; Part of <rdar://problem/14170854>
;
; Single loads with immediate offset.
define <8 x i8> @fct16(i8* nocapture %sp0) {
; CHECK-LABEL: fct16:
; CHECK: ldr b[[REGNUM:[0-9]+]], [x0, #1]
; CHECK-NEXT: mul.8b v0, v[[REGNUM]], v[[REGNUM]]
entry:
  %addr = getelementptr i8, i8* %sp0, i64 1
  %pix_sp0.0.copyload = load i8, i8* %addr, align 1
  %vec = insertelement <8 x i8> undef, i8 %pix_sp0.0.copyload, i32 0
  %vmull.i = mul <8 x i8> %vec, %vec
  ret <8 x i8> %vmull.i
}

define <16 x i8> @fct17(i8* nocapture %sp0) {
; CHECK-LABEL: fct17:
; CHECK: ldr b[[REGNUM:[0-9]+]], [x0, #1]
; CHECK-NEXT: mul.16b v0, v[[REGNUM]], v[[REGNUM]]
entry:
  %addr = getelementptr i8, i8* %sp0, i64 1
  %pix_sp0.0.copyload = load i8, i8* %addr, align 1
  %vec = insertelement <16 x i8> undef, i8 %pix_sp0.0.copyload, i32 0
  %vmull.i = mul <16 x i8> %vec, %vec
  ret <16 x i8> %vmull.i
}

define <4 x i16> @fct18(i16* nocapture %sp0) {
; CHECK-LABEL: fct18:
; CHECK: ldr h[[REGNUM:[0-9]+]], [x0, #2]
; CHECK-NEXT: mul.4h v0, v[[REGNUM]], v[[REGNUM]]
entry:
  %addr = getelementptr i16, i16* %sp0, i64 1
  %pix_sp0.0.copyload = load i16, i16* %addr, align 1
  %vec = insertelement <4 x i16> undef, i16 %pix_sp0.0.copyload, i32 0
  %vmull.i = mul <4 x i16> %vec, %vec
  ret <4 x i16> %vmull.i
}

define <8 x i16> @fct19(i16* nocapture %sp0) {
; CHECK-LABEL: fct19:
; CHECK: ldr h[[REGNUM:[0-9]+]], [x0, #2]
; CHECK-NEXT: mul.8h v0, v[[REGNUM]], v[[REGNUM]]
entry:
  %addr = getelementptr i16, i16* %sp0, i64 1
  %pix_sp0.0.copyload = load i16, i16* %addr, align 1
  %vec = insertelement <8 x i16> undef, i16 %pix_sp0.0.copyload, i32 0
  %vmull.i = mul <8 x i16> %vec, %vec
  ret <8 x i16> %vmull.i
}

define <2 x i32> @fct20(i32* nocapture %sp0) {
; CHECK-LABEL: fct20:
; CHECK: ldr s[[REGNUM:[0-9]+]], [x0, #4]
; CHECK-NEXT: mul.2s v0, v[[REGNUM]], v[[REGNUM]]
entry:
  %addr = getelementptr i32, i32* %sp0, i64 1
  %pix_sp0.0.copyload = load i32, i32* %addr, align 1
  %vec = insertelement <2 x i32> undef, i32 %pix_sp0.0.copyload, i32 0
  %vmull.i = mul <2 x i32> %vec, %vec
  ret <2 x i32> %vmull.i
}

define <4 x i32> @fct21(i32* nocapture %sp0) {
; CHECK-LABEL: fct21:
; CHECK: ldr s[[REGNUM:[0-9]+]], [x0, #4]
; CHECK-NEXT: mul.4s v0, v[[REGNUM]], v[[REGNUM]]
entry:
  %addr = getelementptr i32, i32* %sp0, i64 1
  %pix_sp0.0.copyload = load i32, i32* %addr, align 1
  %vec = insertelement <4 x i32> undef, i32 %pix_sp0.0.copyload, i32 0
  %vmull.i = mul <4 x i32> %vec, %vec
  ret <4 x i32> %vmull.i
}

define <1 x i64> @fct22(i64* nocapture %sp0) {
; CHECK-LABEL: fct22:
; CHECK: ldr d0, [x0, #8]
entry:
  %addr = getelementptr i64, i64* %sp0, i64 1
  %pix_sp0.0.copyload = load i64, i64* %addr, align 1
  %vec = insertelement <1 x i64> undef, i64 %pix_sp0.0.copyload, i32 0
   ret <1 x i64> %vec
}

define <2 x i64> @fct23(i64* nocapture %sp0) {
; CHECK-LABEL: fct23:
; CHECK: ldr d[[REGNUM:[0-9]+]], [x0, #8]
entry:
  %addr = getelementptr i64, i64* %sp0, i64 1
  %pix_sp0.0.copyload = load i64, i64* %addr, align 1
  %vec = insertelement <2 x i64> undef, i64 %pix_sp0.0.copyload, i32 0
  ret <2 x i64> %vec
}

;
; Single loads with register offset.
define <8 x i8> @fct24(i8* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: fct24:
; CHECK: ldr b[[REGNUM:[0-9]+]], [x0, x1]
; CHECK-NEXT: mul.8b v0, v[[REGNUM]], v[[REGNUM]]
entry:
  %addr = getelementptr i8, i8* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i8, i8* %addr, align 1
  %vec = insertelement <8 x i8> undef, i8 %pix_sp0.0.copyload, i32 0
  %vmull.i = mul <8 x i8> %vec, %vec
  ret <8 x i8> %vmull.i
}

define <16 x i8> @fct25(i8* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: fct25:
; CHECK: ldr b[[REGNUM:[0-9]+]], [x0, x1]
; CHECK-NEXT: mul.16b v0, v[[REGNUM]], v[[REGNUM]]
entry:
  %addr = getelementptr i8, i8* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i8, i8* %addr, align 1
  %vec = insertelement <16 x i8> undef, i8 %pix_sp0.0.copyload, i32 0
  %vmull.i = mul <16 x i8> %vec, %vec
  ret <16 x i8> %vmull.i
}

define <4 x i16> @fct26(i16* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: fct26:
; CHECK: ldr h[[REGNUM:[0-9]+]], [x0, x1, lsl #1]
; CHECK-NEXT: mul.4h v0, v[[REGNUM]], v[[REGNUM]]
entry:
  %addr = getelementptr i16, i16* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i16, i16* %addr, align 1
  %vec = insertelement <4 x i16> undef, i16 %pix_sp0.0.copyload, i32 0
  %vmull.i = mul <4 x i16> %vec, %vec
  ret <4 x i16> %vmull.i
}

define <8 x i16> @fct27(i16* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: fct27:
; CHECK: ldr h[[REGNUM:[0-9]+]], [x0, x1, lsl #1]
; CHECK-NEXT: mul.8h v0, v[[REGNUM]], v[[REGNUM]]
entry:
  %addr = getelementptr i16, i16* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i16, i16* %addr, align 1
  %vec = insertelement <8 x i16> undef, i16 %pix_sp0.0.copyload, i32 0
  %vmull.i = mul <8 x i16> %vec, %vec
  ret <8 x i16> %vmull.i
}

define <2 x i32> @fct28(i32* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: fct28:
; CHECK: ldr s[[REGNUM:[0-9]+]], [x0, x1, lsl #2]
; CHECK-NEXT: mul.2s v0, v[[REGNUM]], v[[REGNUM]]
entry:
  %addr = getelementptr i32, i32* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i32, i32* %addr, align 1
  %vec = insertelement <2 x i32> undef, i32 %pix_sp0.0.copyload, i32 0
  %vmull.i = mul <2 x i32> %vec, %vec
  ret <2 x i32> %vmull.i
}

define <4 x i32> @fct29(i32* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: fct29:
; CHECK: ldr s[[REGNUM:[0-9]+]], [x0, x1, lsl #2]
; CHECK-NEXT: mul.4s v0, v[[REGNUM]], v[[REGNUM]]
entry:
  %addr = getelementptr i32, i32* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i32, i32* %addr, align 1
  %vec = insertelement <4 x i32> undef, i32 %pix_sp0.0.copyload, i32 0
  %vmull.i = mul <4 x i32> %vec, %vec
  ret <4 x i32> %vmull.i
}

define <1 x i64> @fct30(i64* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: fct30:
; CHECK: ldr d0, [x0, x1, lsl #3]
entry:
  %addr = getelementptr i64, i64* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i64, i64* %addr, align 1
  %vec = insertelement <1 x i64> undef, i64 %pix_sp0.0.copyload, i32 0
   ret <1 x i64> %vec
}

define <2 x i64> @fct31(i64* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: fct31:
; CHECK: ldr d0, [x0, x1, lsl #3]
entry:
  %addr = getelementptr i64, i64* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i64, i64* %addr, align 1
  %vec = insertelement <2 x i64> undef, i64 %pix_sp0.0.copyload, i32 0
  ret <2 x i64> %vec
}
