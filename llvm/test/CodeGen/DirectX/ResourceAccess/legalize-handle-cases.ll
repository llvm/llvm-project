; RUN: opt -S -dxil-resource-type -dxil-resource-access -disable-verify \
; RUN:  -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; The file contains examples of hlsl snippets that will generate invalid dxil
; resource access, either through code-gen or by an InstCombine/GVN sink
; optimization

; NOTE: The below resources are generated with:
;
;   RWBuffer<int> In : register(u0);
;   RWStructuredBuffer<int> Out0 : register(u1);
;   RWStructuredBuffer<int> Out1 : register(u2);
;   RWStructuredBuffer<int> OutArr[];

;   cbuffer c {
;       bool cond;
;   };

%__cblayout_c = type <{ i32 }>

@.str = internal unnamed_addr constant [3 x i8] c"In\00", align 1
@.str.2 = internal unnamed_addr constant [5 x i8] c"Out0\00", align 1
@.str.3 = internal unnamed_addr constant [5 x i8] c"Out1\00", align 1
@c.cb = local_unnamed_addr global target("dx.CBuffer", %__cblayout_c) poison
@c.str = internal unnamed_addr constant [2 x i8] c"c\00", align 1
@OutArr.str = internal unnamed_addr constant [7 x i8] c"OutArr\00", align 1

; Local select into global resource array:
;
;   RWStructuredBuffer<int> Out = cond ? OutArr[0] : OutArr[1];
;   Out[GI] = WaveActiveMax(In[GI]);
;
; CHECK-LABEL: @select_global_resource_array()
define void @select_global_resource_array() {
entry:
  %c.cb_h.i.i = tail call target("dx.CBuffer", %__cblayout_c) @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_cst(i32 4, i32 0, i32 1, i32 0, ptr nonnull @c.str)
  store target("dx.CBuffer", %__cblayout_c) %c.cb_h.i.i, ptr @c.cb, align 4
  %c.cb = load target("dx.CBuffer", %__cblayout_c), ptr @c.cb, align 4
  %0 = call ptr addrspace(2) @llvm.dx.resource.getpointer.p2.tdx.CBuffer_s___cblayout_cst(target("dx.CBuffer", %__cblayout_c) %c.cb, i32 0)
  %1 = load i32, ptr addrspace(2) %0, align 4
  %loadedv.i = trunc nuw i32 %1 to i1
  br i1 %loadedv.i, label %cond.true.i, label %cond.false.i

cond.true.i:
; CHECK:      cond.true.i:
; CHECK-NEXT:   br label %cond.end.i
  %2 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_i32_1_0t(i32 2, i32 0, i32 -1, i32 0, ptr nonnull @OutArr.str)
  br label %cond.end.i

cond.false.i:
; CHECK:      cond.false.i:
; CHECK-NEXT:   br label %cond.end.i
  %3 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_i32_1_0t(i32 2, i32 0, i32 -1, i32 1, ptr nonnull @OutArr.str)
  br label %cond.end.i

cond.end.i:
; CHECK:     cond.end.i
; CHECK-NEXT:  %[[HANDLE_IDX:.*]] = phi i32 [ 0, %cond.true.i ], [ 1, %cond.false.i ]
; CHECK:       %[[TID:.*]] = tail call i32 @llvm.dx.flattened.thread.id.in.group()
; CHECK:       %[[WAVE_MAX:.*]] = tail call i32 @llvm.dx.wave.reduce.max.i32(i32 %{{.*}})
; CHECK-NEXT:  %[[HANDLE:.*]] = tail call target("dx.RawBuffer", i32, 1, 0)
; CHECK-SAME:    @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_i32_1_0t(i32 2, i32 0, i32 -1, i32 %[[HANDLE_IDX]], ptr nonnull @OutArr.str)
; CHECK-NEXT:  call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_i32_1_0t.i32(target("dx.RawBuffer", i32, 1, 0) %[[HANDLE]], i32 %[[TID]], i32 0, i32 %[[WAVE_MAX]])
; CHECK-NEXT:  ret void
  %cond.i.sroa.speculated = phi target("dx.RawBuffer", i32, 1, 0) [ %2, %cond.true.i ], [ %3, %cond.false.i ]
  %4 = tail call target("dx.TypedBuffer", i32, 1, 0, 1) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0_1t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %5 = tail call i32 @llvm.dx.flattened.thread.id.in.group()
  %6 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_1_0_1t(target("dx.TypedBuffer", i32, 1, 0, 1) %4, i32 %5)
  %7 = load i32, ptr %6, align 4
  %hlsl.wave.active.max.i = tail call i32 @llvm.dx.wave.reduce.max.i32(i32 %7)
  %8 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %cond.i.sroa.speculated, i32 %5)
  store i32 %hlsl.wave.active.max.i, ptr %8, align 4
  ret void
}

; GVN Sink of handle ptr
;
;   if (cond) {
;     Out0[GI] = WaveActiveSum(In[GI]);
;   } else {
;     Out0[0] = In[GI];
;   }
;   Out0[GI] = WaveActiveSum(In[GI]);
;
; CHECK-LABEL: @gvn_sink()
define void @gvn_sink() {
entry:
  %0 = tail call target("dx.TypedBuffer", i32, 1, 0, 1) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0_1t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %1 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str.2)
  %c.cb_h.i.i = tail call target("dx.CBuffer", %__cblayout_c) @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_cst(i32 4, i32 0, i32 1, i32 0, ptr nonnull @c.str)
  store target("dx.CBuffer", %__cblayout_c) %c.cb_h.i.i, ptr @c.cb, align 4
  %2 = tail call i32 @llvm.dx.flattened.thread.id.in.group()
  %c.cb = load target("dx.CBuffer", %__cblayout_c), ptr @c.cb, align 4
  %3 = call ptr addrspace(2) @llvm.dx.resource.getpointer.p2.tdx.CBuffer_s___cblayout_cst(target("dx.CBuffer", %__cblayout_c) %c.cb, i32 0)
  %4 = load i32, ptr addrspace(2) %3, align 4
  %loadedv.i = trunc nuw i32 %4 to i1
  %5 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_1_0_1t(target("dx.TypedBuffer", i32, 1, 0, 1) %0, i32 %2)
  %6 = load i32, ptr %5, align 4
  br i1 %loadedv.i, label %if.then.i, label %if.else.i

if.then.i:
  %hlsl.wave.active.sum.i = tail call i32 @llvm.dx.wave.reduce.sum.i32(i32 %6)
  %7 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %1, i32 %2)
  store i32 %hlsl.wave.active.sum.i, ptr %7, align 4
  br label %_Z4mainj.exit

if.else.i:
  %8 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %1, i32 0)
  store i32 %6, ptr %8, align 4
  %.pre = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %1, i32 %2)
  br label %_Z4mainj.exit

_Z4mainj.exit:
; CHECK:     _Z4mainj.exit:
; CHECK-NEXT:  %[[TID:.*]] = phi i32 [ %2, %if.then.i ], [ %2, %if.else.i ]
; CHECK-NEXT:  %[[HANDLE_IDX:.*]] = phi i32 [ 0, %if.then.i ], [ 0, %if.else.i ]
; CHECK-NEXT:  %[[HANDLE:.*]] = tail call target("dx.RawBuffer", i32, 1, 0)
; CHECK-SAME:    @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 1, i32 1, i32 %[[HANDLE_IDX]], ptr nonnull @.str.2)
; CHECK:       %[[WAVE_SUM:.*]] = tail call i32 @llvm.dx.wave.reduce.sum.i32(i32 {{.*}})
; CHECK-NEXT:  call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_i32_1_0t.i32(
; CHECK-SAME:    target("dx.RawBuffer", i32, 1, 0) %[[HANDLE]], i32 %[[TID]], i32 0, i32 %[[WAVE_SUM]])
; CHECK-NEXT:  ret void
  %.pre-phi1 = phi ptr [ %7, %if.then.i ], [ %.pre, %if.else.i ]
  %9 = load i32, ptr %5, align 4
  %hlsl.wave.active.sum5.i = tail call i32 @llvm.dx.wave.reduce.sum.i32(i32 %9)
  store i32 %hlsl.wave.active.sum5.i, ptr %.pre-phi1, align 4
  ret void
}

; Using a local array of global resources
;
;   RWStructuredBuffer<int> Outs[2] = {OutArr[0], OutArr[1]};
;   Outs[cond ? 0 : 1][GI] = In[GI];
;
; CHECK-LABEL: @local_array_of_global_resources()
define void @local_array_of_global_resources() {
entry:
  %0 = tail call target("dx.TypedBuffer", i32, 1, 0, 1) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0_1t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %c.cb_h.i.i = tail call target("dx.CBuffer", %__cblayout_c) @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_cst(i32 4, i32 0, i32 1, i32 0, ptr nonnull @c.str)
  store target("dx.CBuffer", %__cblayout_c) %c.cb_h.i.i, ptr @c.cb, align 4
  %1 = tail call i32 @llvm.dx.flattened.thread.id.in.group()
  %2 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_i32_1_0t(i32 2, i32 0, i32 -1, i32 0, ptr nonnull @OutArr.str)
  %3 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_i32_1_0t(i32 2, i32 0, i32 -1, i32 1, ptr nonnull @OutArr.str)
  %4 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_1_0_1t(target("dx.TypedBuffer", i32, 1, 0, 1) %0, i32 %1)
  %5 = load i32, ptr %4, align 4
  %c.cb = load target("dx.CBuffer", %__cblayout_c), ptr @c.cb, align 4
  %6 = call ptr addrspace(2) @llvm.dx.resource.getpointer.p2.tdx.CBuffer_s___cblayout_cst(target("dx.CBuffer", %__cblayout_c) %c.cb, i32 0)
  %7 = load i32, ptr addrspace(2) %6, align 4
  %loadedv.i = trunc nuw i32 %7 to i1

; CHECK:       %[[TID:.*]] = tail call i32 @llvm.dx.flattened.thread.id.in.group()
; CHECK:       %[[HANDLE_IDX:.*]] = select i1 %loadedv.i, i32 0, i32 1
; CHECK-NEXT:  %[[HANDLE:.*]] = tail call target("dx.RawBuffer", i32, 1, 0)
; CHECK-SAME:    @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_i32_1_0t(i32 2, i32 0, i32 -1, i32 %[[HANDLE_IDX]], ptr nonnull @OutArr.str)
; CHECK-NEXT:  call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_i32_1_0t.i32(target("dx.RawBuffer", i32, 1, 0) %[[HANDLE]], i32 %[[TID]], i32 0, i32 {{.*}})
  %.sroa.speculated = select i1 %loadedv.i, target("dx.RawBuffer", i32, 1, 0) %2, target("dx.RawBuffer", i32, 1, 0) %3
  %8 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %.sroa.speculated, i32 %1)
  store i32 %5, ptr %8, align 4
  ret void
}

; Sink of a load/store
;
;   if (cond) {
;     Out0[GI] += In[GI];
;   } else {
;     Out1[GI] += In[GI];
;   }
;
; CHECK-LABEL: @sink_load_store()
define void @sink_load_store() {
entry:
; CHECK: %[[IN_HANDLE:.*]] = tail call target("dx.TypedBuffer", i32, 1, 0, 1) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0_1t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %0 = tail call target("dx.TypedBuffer", i32, 1, 0, 1) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0_1t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %c.cb_h.i.i = tail call target("dx.CBuffer", %__cblayout_c) @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_cst(i32 4, i32 0, i32 1, i32 0, ptr nonnull @c.str)
  store target("dx.CBuffer", %__cblayout_c) %c.cb_h.i.i, ptr @c.cb, align 4
; CHECK: %[[TID:.*]] = tail call i32 @llvm.dx.flattened.thread.id.in.group()
  %1 = tail call i32 @llvm.dx.flattened.thread.id.in.group()
  %c.cb = load target("dx.CBuffer", %__cblayout_c), ptr @c.cb, align 4
  %2 = call ptr addrspace(2) @llvm.dx.resource.getpointer.p2.tdx.CBuffer_s___cblayout_cst(target("dx.CBuffer", %__cblayout_c) %c.cb, i32 0)
  %3 = load i32, ptr addrspace(2) %2, align 4
  %loadedv.i = trunc nuw i32 %3 to i1
; CHECK: %[[IN_LOAD:.*]] = call { i32, i1 } @llvm.dx.resource.load.typedbuffer.i32.tdx.TypedBuffer_i32_1_0_1t(target("dx.TypedBuffer", i32, 1, 0, 1) %[[IN_HANDLE]], i32 %[[TID]])
; CHECK: %[[IN_X:.*]] = extractvalue { i32, i1 } %[[IN_LOAD]], 0
  %4 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_1_0_1t(target("dx.TypedBuffer", i32, 1, 0, 1) %0, i32 %1)
  %5 = load i32, ptr %4, align 4
  br i1 %loadedv.i, label %if.then.i, label %if.else.i

if.then.i:
; CHECK:     if.then.i:
; CHECK-NEXT:  %[[HANDLE0:.*]] = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str.2)
; CHECK-NEXT:  %[[LOAD0:.*]] = call { i32, i1 } @llvm.dx.resource.load.rawbuffer.i32.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %[[HANDLE0]], i32 %[[TID]], i32 0)
; CHECK-NEXT:  %[[X0:.*]] = extractvalue { i32, i1 } %[[LOAD0]], 0
; CHECK-NEXT:  %[[ADD0:.*]] = add i32 %[[X0]], %[[IN_X]]
; CHECK-NEXT:  call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_i32_1_0t.i32(target("dx.RawBuffer", i32, 1, 0) %[[HANDLE0]], i32 %[[TID]], i32 0, i32 %[[ADD0]])
; CHECK-NEXT:  br label %_Z4mainj.exit
  %6 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str.2)
  %7 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %6, i32 %1)
  br label %_Z4mainj.exit

if.else.i:
; CHECK:     if.else.i:
; CHECK-NEXT:  %[[HANDLE1:.*]] = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 2, i32 1, i32 0, ptr nonnull @.str.3)
; CHECK-NEXT:  %[[LOAD1:.*]] = call { i32, i1 } @llvm.dx.resource.load.rawbuffer.i32.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %[[HANDLE1]], i32 %[[TID]], i32 0)
; CHECK-NEXT:  %[[X1:.*]] = extractvalue { i32, i1 } %[[LOAD1]], 0
; CHECK-NEXT:  %[[ADD1:.*]] = add i32 %[[X1]], %[[IN_X]]
; CHECK-NEXT:  call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_i32_1_0t.i32(target("dx.RawBuffer", i32, 1, 0) %[[HANDLE1]], i32 %[[TID]], i32 0, i32 %[[ADD1]])
; CHECK-NEXT:  br label %_Z4mainj.exit
  %8 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 2, i32 1, i32 0, ptr nonnull @.str.3)
  %9 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %8, i32 %1)
  br label %_Z4mainj.exit

_Z4mainj.exit:
  %.sink = phi ptr [ %7, %if.then.i ], [ %9, %if.else.i ]
  %10 = load i32, ptr %.sink, align 4
  %add.i = add i32 %10, %5
  store i32 %add.i, ptr %.sink, align 4
  ret void
}
