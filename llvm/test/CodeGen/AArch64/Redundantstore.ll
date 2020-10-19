; RUN: llc < %s -O3 -mtriple=aarch64-eabi 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
@end_of_array = common global i8* null, align 8

; The tests in this file should not produce a TypeSize warning.
; CHECK-NOT: warning: {{.*}}TypeSize is not scalable

; CHECK-LABEL: @test
; CHECK: stur
; CHECK-NOT: stur
define i8* @test(i32 %size) {
entry:
  %0 = load i8*, i8** @end_of_array, align 8
  %conv = sext i32 %size to i64
  %and = and i64 %conv, -8
  %conv2 = trunc i64 %and to i32
  %add.ptr.sum = add nsw i64 %and, -4
  %add.ptr3 = getelementptr inbounds i8, i8* %0, i64 %add.ptr.sum
  %size4 = bitcast i8* %add.ptr3 to i32*
  store i32 %conv2, i32* %size4, align 4
  %add.ptr.sum9 = add nsw i64 %and, -4
  %add.ptr5 = getelementptr inbounds i8, i8* %0, i64 %add.ptr.sum9
  %size6 = bitcast i8* %add.ptr5 to i32*
  store i32 %conv2, i32* %size6, align 4
  ret i8* %0
}

; #include <arm_sve.h>
; #include <stdint.h>
;
; void redundant_store(uint32_t *x) {
;     *x = 1;
;     *(svint32_t *)x = svdup_s32(0);
; }

; CHECK-LABEL: @redundant_store
define void @redundant_store(i32* nocapture %x) local_unnamed_addr #0 {
  %1 = bitcast i32* %x to <vscale x 4 x i32>*
  store i32 1, i32* %x, align 4
  %2 = tail call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 0)
  store <vscale x 4 x i32> %2, <vscale x 4 x i32>* %1, align 16
  ret void
}

declare <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32)

attributes #0 = { "target-cpu"="generic" "target-features"="+neon,+sve,+v8.2a" }
