; RUN: opt -S -passes='early-cse<memssa>' %s -o %t
; RUN: FileCheck --check-prefixes=CSE,CHECK %s < %t
; Finish compiling to verify that dxil-op-lower removes the globals entirely.
; RUN: opt -mtriple=dxil-pc-shadermodel6.0-compute -S -dxil-op-lower %t -o - | FileCheck --check-prefixes=DXOP,CHECK %s
; RUN: opt -mtriple=dxil-pc-shadermodel6.6-compute -S -dxil-op-lower %t -o - | FileCheck --check-prefixes=DXOP,CHECK %s
; RUN: llc -mtriple=dxil-pc-shadermodel6.0-compute --filetype=asm -o - %t | FileCheck --check-prefixes=DXOP,CHECK %s
; RUN: llc -mtriple=dxil-pc-shadermodel6.6-compute --filetype=asm -o - %t | FileCheck --check-prefixes=DXOP,CHECK %s

; Ensure that EarlyCSE is able to eliminate unneeded loads of resource globals across typedBufferLoad.
; Also that DXILOpLowering eliminates the globals entirely.

%"class.hlsl::RWBuffer" = type { target("dx.TypedBuffer", <4 x float>, 1, 0, 0) }

; DXOP-NOT: @In = global
; DXOP-NOT: @Out = global
@In = global %"class.hlsl::RWBuffer" zeroinitializer, align 4
@Out = global %"class.hlsl::RWBuffer" zeroinitializer, align 4

; CHECK-LABEL define void @main()
define void @main() local_unnamed_addr #0 {
entry:
  ; DXOP: [[In_h_i:%.*]] = call %dx.types.Handle @dx.op.createHandle
  ; DXOP: [[Out_h_i:%.*]] = call %dx.types.Handle @dx.op.createHandle
  %In_h.i = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v4f32_1_0_0t(i32 0, i32 0, i32 1, i32 0, ptr null)
  store target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %In_h.i, ptr @In, align 4
  %Out_h.i = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v4f32_1_0_0t(i32 4, i32 1, i32 1, i32 0, ptr null)
  store target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %Out_h.i, ptr @Out, align 4
  ; CSE: call i32 @llvm.dx.flattened.thread.id.in.group()
  %0 = call i32 @llvm.dx.flattened.thread.id.in.group()
  ; CHECK-NOT: load {{.*}} ptr @In
  %1 = load target("dx.TypedBuffer", <4 x float>, 1, 0, 0), ptr @In, align 4
  ; CSE: call noundef { <4 x float>, i1 } @llvm.dx.resource.load.typedbuffer.v4f32.tdx.TypedBuffer_v4f32_1_0_0t
  %load = call noundef {<4 x float>, i1} @llvm.dx.resource.load.typedbuffer.v4f32.tdx.TypedBuffer_v4f32_1_0_0t(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %1, i32 %0)
  %2 = extractvalue {<4 x float>, i1} %load, 0
  ; CHECK-NOT: load {{.*}} ptr @In
  %3 = load target("dx.TypedBuffer", <4 x float>, 1, 0, 0), ptr @In, align 4
  %load2 = call noundef {<4 x float>, i1} @llvm.dx.resource.load.typedbuffer.v4f32.tdx.TypedBuffer_v4f32_1_0_0t(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %3, i32 %0)
  %4 = extractvalue {<4 x float>, i1} %load2, 0
  %add.i = fadd <4 x float> %2, %4
  call void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_v4f32_1_0_0t.v4f32(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %Out_h.i, i32 %0, <4 x float> %add.i)
  ; CHECK: ret void
  ret void
}

; CSE-DAG: declare { <4 x float>, i1 } @llvm.dx.resource.load.typedbuffer.v4f32.tdx.TypedBuffer_v4f32_1_0_0t(target("dx.TypedBuffer", <4 x float>, 1, 0, 0), i32) [[ROAttr:#[0-9]+]]
; CSE-DAG: declare void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_v4f32_1_0_0t.v4f32(target("dx.TypedBuffer", <4 x float>, 1, 0, 0), i32, <4 x float>) [[WOAttr:#[0-9]+]]

attributes #0 = { convergent noinline norecurse "frame-pointer"="all" "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

; Just need to split up the DAG searches.
; CSE: attributes #0

; CSE-DAG: attributes [[ROAttr]] = { {{.*}} memory(read) }
; CSE-DAG: attributes [[WOAttr]] = { {{.*}} memory(write) }
