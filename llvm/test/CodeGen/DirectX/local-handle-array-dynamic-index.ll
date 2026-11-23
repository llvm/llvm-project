; RUN: llc %s -o - | FileCheck %s

; A dynamically indexed local array of handles used to assert in the DXIL
; legalizer.

target triple = "dxil-unknown-shadermodel6.0-compute"

%"class.hlsl::RWByteAddressBuffer" = type { target("dx.RawBuffer", i8, 1, 0) }

@gBufArray.str = internal unnamed_addr constant [10 x i8] c"gBufArray\00", align 1

; CHECK-NOT: alloca [2 x %"class.hlsl::RWByteAddressBuffer"]
; CHECK-LABEL: define void @main() {{.*}} {
; CHECK:         %[[ARR:.*]] = alloca [2 x i32], align 4
; CHECK-NEXT:    store [2 x i32] zeroinitializer, ptr %[[ARR]], align 4
; CHECK-NEXT:    %[[TID:.*]] = call i32 @dx.op.threadId.i32(i32 93, i32 0)
; CHECK:         %[[VAL:.*]] = load i32, ptr %{{.*}}, align 4
; CHECK-NEXT:    %[[HANDLE:.*]] = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 0, i32 %[[VAL]], i1 false)
; CHECK-NEXT:    call void @dx.op.bufferStore.i32(i32 69, %dx.types.Handle %[[HANDLE]], i32 0, i32 undef, i32 42, i32 undef, i32 undef, i32 undef, i8 1)
; CHECK:         ret void
define void @main() local_unnamed_addr #0 {
entry:
  %arr.i = alloca [2 x %"class.hlsl::RWByteAddressBuffer"], align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %arr.i)
  %0 = tail call i32 @llvm.dx.thread.id(i32 0)
  %1 = tail call target("dx.RawBuffer", i8, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_0t(i32 0, i32 0, i32 4, i32 0, ptr nonnull @gBufArray.str)
  store target("dx.RawBuffer", i8, 1, 0) %1, ptr %arr.i, align 4
  %2 = tail call target("dx.RawBuffer", i8, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_0t(i32 0, i32 0, i32 4, i32 1, ptr nonnull @gBufArray.str)
  %arrayidx2.i = getelementptr inbounds nuw i8, ptr %arr.i, i32 4
  store target("dx.RawBuffer", i8, 1, 0) %2, ptr %arrayidx2.i, align 4
  %and.i = and i32 %0, 1
  %arrayidx4.i = getelementptr inbounds nuw [4 x i8], ptr %arr.i, i32 %and.i
  %3 = load target("dx.RawBuffer", i8, 1, 0), ptr %arrayidx4.i, align 4
  %4 = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) %3, i32 0)
  store i32 42, ptr %4, align 4
  call void @llvm.lifetime.end.p0(ptr nonnull %arr.i)
  ret void
}

declare i32 @llvm.dx.thread.id(i32 immarg)
declare void @llvm.lifetime.start.p0(ptr captures(none))
declare void @llvm.lifetime.end.p0(ptr captures(none))
declare target("dx.RawBuffer", i8, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_0t(i32, i32, i32, i32, ptr)
declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0), i32)

attributes #0 = { convergent mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, inaccessiblemem: none, target_mem: none) "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
