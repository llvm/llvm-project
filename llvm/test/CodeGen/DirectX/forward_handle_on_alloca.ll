; RUN: opt -S -dxil-forward-handle-accesses  %s | FileCheck %s  --check-prefixes=CHECK,FHCHECK
; RUN: opt -S -mtriple=dxil--shadermodel6.3-compute -passes='function(dxil-forward-handle-accesses),dse' %s | FileCheck %s --check-prefix=CHECK

; Note: test to confirm fix for issues: 140819 & 151764

%"class.hlsl::RWStructuredBuffer" = type { target("dx.RawBuffer", i32, 1, 0) }
@global = internal unnamed_addr global %"class.hlsl::RWStructuredBuffer" poison, align 4
@name = private unnamed_addr constant [5 x i8] c"dest\00", align 1


; NOTE: intent of this test is to confirm load target("dx.RawBuffer", i32, 1, 0)
;       is replaced with call @llvm.dx.resource.getpointer
define void @CSMain() local_unnamed_addr {
; CHECK-LABEL: define void @CSMain() local_unnamed_addr {
; CHECK-NEXT:  [[ENTRY:.*:]]
; FHCHECK-NEXT:    [[AGG_TMP_I1_SROA_0:%.*]] = alloca target("dx.RawBuffer", i32, 1, 0), align 8
; CHECK-NEXT:    [[TMP0:%.*]] = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 3, i32 1, i32 0, ptr nonnull @name)
; CHECK-NEXT:    store target("dx.RawBuffer", i32, 1, 0) [[TMP0]], ptr @global, align 4
; FHCHECK-NEXT:    [[TMP2:%.*]] = load i32, ptr @global, align 4
; FHCHECK-NEXT:    store i32 [[TMP2]], ptr [[AGG_TMP_I1_SROA_0]], align 8
; CHECK-NEXT:    [[TMP3:%.*]] = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) [[TMP0]], i32 0)
; CHECK-NEXT:    store i32 0, ptr [[TMP3]], align 4
; CHECK-NEXT:    ret void
;
entry:
  %alloca = alloca target("dx.RawBuffer", i32, 1, 0), align 8
  %handle  = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 3, i32 1, i32 0, ptr nonnull @name)
  store target("dx.RawBuffer", i32, 1, 0) %handle , ptr @global, align 4
  %val  = load i32, ptr @global, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %alloca)
  store i32 %val , ptr  %alloca, align 8
  %indirect = load target("dx.RawBuffer", i32, 1, 0), ptr  %alloca, align 8
  %buff = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %indirect, i32 0)
  store i32 0, ptr %buff, align 4
  call void @llvm.lifetime.end.p0(ptr nonnull %alloca)
  ret void
}
