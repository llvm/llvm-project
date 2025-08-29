; RUN: opt -S -dxil-forward-handle-accesses -mtriple=dxil--shadermodel6.3-library %s | FileCheck %s

%__cblayout_CB = type <{ float, i32, i32 }>
%__cblayout_CB2 = type <{ float }>
%struct.Scalars = type { float, i32, i32 }

@CB.cb = local_unnamed_addr global target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 12, 0, 4, 8)) poison
@CB2.cb = local_unnamed_addr global target("dx.CBuffer", target("dx.Layout", %__cblayout_CB2, 4, 0)) poison

define void @main() local_unnamed_addr #1 {
entry:
  ; CHECK: [[CB:%.*]] = tail call target({{.*}}) @llvm.dx.resource.handlefrombinding
  %h = tail call target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 12, 0, 4, 8)) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  store target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 12, 0, 4, 8)) %h, ptr @CB.cb, align 4
  %_ZL3Out_h.i.i = tail call target("dx.RawBuffer", %struct.Scalars, 1, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  ; CHECK-NOT: load target({{.*}}), ptr @CB.cb
  %cb = load target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 12, 0, 4, 8)), ptr @CB.cb, align 4
  ; CHECK: call { float, float, float, float } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target({{.*}}) [[CB]], i32 0)
  %0 = call { float, float, float, float } @llvm.dx.resource.load.cbufferrow.4(target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 12, 0, 4, 8)) %cb, i32 0)
  %1 = extractvalue { float, float, float, float } %0, 0
  call void @llvm.dx.resource.store.rawbuffer(target("dx.RawBuffer", %struct.Scalars, 1, 0) %_ZL3Out_h.i.i, i32 0, i32 0, float %1)
  
  ; CHECK: [[CB2:%.*]] = tail call target({{.*}}) @llvm.dx.resource.handlefromimplicitbinding
  %h2 = tail call target("dx.CBuffer", target("dx.Layout", %__cblayout_CB2, 4, 0)) @llvm.dx.resource.handlefromimplicitbinding(i32 100, i32 0, i32 1, i32 0, ptr null)
  store target("dx.CBuffer", target("dx.Layout", %__cblayout_CB2, 4, 0)) %h2, ptr @CB2.cb, align 4
  ; CHECK-NOT: load target({{.*}}), ptr @CB2.cb
  %cb2 = load target("dx.CBuffer", target("dx.Layout", %__cblayout_CB2, 4, 0)), ptr @CB2.cb, align 4

  ret void
}

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none) "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
