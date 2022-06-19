; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -S < %s 2>&1 | FileCheck %s
;
; We weren't properly considering the args in callsites in equality or hashing.

target triple = "armv7-dcg-linux-gnueabi"

; CHECK-LABEL: define <8 x i16> @vpx_idct32_32_neon
define <8 x i16> @vpx_idct32_32_neon(i8* %p, <8 x i16> %v) {
entry:
; CHECK: MemoryUse(liveOnEntry)
  %load1 = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %p, i32 2) #4 ; load CSE replacement

; CHECK: 1 = MemoryDef(liveOnEntry)
  call void @llvm.arm.neon.vst1.p0i8.v8i16(i8* %p, <8 x i16> %v, i32 2) #4 ; clobber

  %p_next = getelementptr inbounds i8, i8* %p, i32 16
; CHECK: MemoryUse(liveOnEntry)
  %load2 = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %p_next, i32 2) #4 ; non-aliasing load needed to trigger bug

; CHECK: MemoryUse(1)
  %load3 = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %p, i32 2) #4 ; load CSE removed

  %add = add <8 x i16> %load1, %load2
  %ret = add <8 x i16> %add, %load3
  ret <8 x i16> %ret
}

; Function Attrs: argmemonly nounwind readonly
declare <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8*, i32) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.arm.neon.vst1.p0i8.v8i16(i8*, <8 x i16>, i32) #1

attributes #1 = { argmemonly nounwind }
attributes #2 = { argmemonly nounwind readonly }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind }
