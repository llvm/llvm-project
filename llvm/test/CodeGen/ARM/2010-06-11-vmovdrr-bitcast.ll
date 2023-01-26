; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o /dev/null
; rdar://8084742

%struct.__int8x8x2_t = type { [2 x <8 x i8>] }

define void @foo(ptr nocapture %a, ptr %b) nounwind {
entry:
 %srcval = load i128, ptr %a, align 8                ; <i128> [#uses=2]
 %tmp6 = trunc i128 %srcval to i64               ; <i64> [#uses=1]
 %tmp8 = lshr i128 %srcval, 64                   ; <i128> [#uses=1]
 %tmp9 = trunc i128 %tmp8 to i64                 ; <i64> [#uses=1]
 %tmp16.i = bitcast i64 %tmp6 to <8 x i8>        ; <<8 x i8>> [#uses=1]
 %tmp20.i = bitcast i64 %tmp9 to <8 x i8>        ; <<8 x i8>> [#uses=1]
 tail call void @llvm.arm.neon.vst2.p0.v8i8(ptr %b, <8 x i8> %tmp16.i, <8 x i8> %tmp20.i, i32 1) nounwind
 ret void
}

declare void @llvm.arm.neon.vst2.p0.v8i8(ptr, <8 x i8>, <8 x i8>, i32) nounwind
