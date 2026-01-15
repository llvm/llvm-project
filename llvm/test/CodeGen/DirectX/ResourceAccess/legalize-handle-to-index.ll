; RUN: opt -S -dxil-resource-type -dxil-resource-access -disable-verify \
; RUN:  -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

@OutArr.str = internal unnamed_addr constant [7 x i8] c"OutArr\00", align 1

; CHECK-LABEL: handle_phi_load(
; CHECK-SAME:   i1 %[[COND:.*]], i32 %[[A:.*]], i32 %[[B:.*]])
define i32 @handle_phi_load(i1 %cond, i32 %a, i32 %b) {
; CHECK-NOT: handlefromimplicitbinding
; CHECK:     main:
; CHECK-NEXT:  %[[IDX:.*]] = phi i32 [ 0, %entry ], [ 1, %if.then.i ]
; CHECK-NEXT:  %[[C:.*]] = phi i32 [ %[[A]], %entry ], [ %[[B]], %if.then.i ]
; CHECK-NEXT:  %[[HANDLE:.*]] = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_i32_1_0t(i32 2, i32 0, i32 -1, i32 %[[IDX]], ptr nonnull @OutArr.str)
; CHECK-NEXT:  %[[LOAD:.*]] = call { i32, i1 } @llvm.dx.resource.load.rawbuffer.i32.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %[[HANDLE]], i32 %[[C]], i32 0)
; CHECK-NEXT:  %[[X:.*]] = extractvalue { i32, i1 } %[[LOAD]], 0
; CHECK-NEXT:  ret i32 %[[X]]
entry:
  %handle0 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_i32_1_0t(i32 2, i32 0, i32 -1, i32 0, ptr nonnull @OutArr.str)
  br i1 %cond, label %if.then.i, label %main

if.then.i:
  %handle1 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_i32_1_0t(i32 2, i32 0, i32 -1, i32 1, ptr nonnull @OutArr.str)
  br label %main

main:
  %handle_phi = phi target("dx.RawBuffer", i32, 1, 0) [ %handle0, %entry ], [ %handle1, %if.then.i ]
  %c = phi i32 [ %a, %entry ], [ %b, %if.then.i ]
  %ptr = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %handle_phi, i32 %c)
  %x = load i32, ptr %ptr, align 4
  ret i32 %x
}

; CHECK-LABEL: handle_select_store(
; CHECK-SAME:   i32 %[[X:.*]], i1 %[[COND:.*]], i32 %[[A:.*]], i32 %[[B:.*]])
define void @handle_select_store(i32 %x, i1 %cond, i32 %a, i32 %b) {
; CHECK-NOT: handlefromimplicitbinding
; CHECK:     entry:
; CHECK-NEXT:  %[[IDX:.*]] = select i1 %[[COND]], i32 0, i32 1
; CHECK-NEXT:  %[[C:.*]] = select i1 %cond, i32 %[[A]], i32 %[[B]]
; CHECK-NEXT:  %[[HANDLE:.*]] = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_i32_1_0t(i32 2, i32 0, i32 -1, i32 %[[IDX]], ptr nonnull @OutArr.str)
; CHECK-NEXT:  call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_i32_1_0t.i32(target("dx.RawBuffer", i32, 1, 0) %[[HANDLE]], i32 %[[C]], i32 0, i32 %[[X]])
; CHECK-NEXT:  ret void
entry:
  %handle0 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_i32_1_0t(i32 2, i32 0, i32 -1, i32 0, ptr nonnull @OutArr.str)
  %handle1 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_i32_1_0t(i32 2, i32 0, i32 -1, i32 1, ptr nonnull @OutArr.str)
  %handle = select i1 %cond, target("dx.RawBuffer", i32, 1, 0) %handle0, target("dx.RawBuffer", i32, 1, 0) %handle1
  %c = select i1 %cond, i32 %a, i32 %b
  %ptr = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %handle, i32 %c)
  store i32 %x, ptr %ptr, align 4
  ret void
}

; CHECK-LABEL: ptr_phi_store(
; CHECK-SAME:   i32 %[[X:.*]], i1 %[[COND:.*]], i32 %[[A:.*]], i32 %[[B:.*]])
define void @ptr_phi_store(i32 %x, i1 %cond, i32 %a, i32 %b) {
; CHECK-NOT: handlefromimplicitbinding
; CHECK:     main:
; CHECK-NEXT:  %[[C:.*]] = phi i32 [ %[[A]], %entry ], [ %[[B]], %if.then.i ]
; CHECK-NEXT:  %[[IDX:.*]] = phi i32 [ 0, %entry ], [ 1, %if.then.i ]
; CHECK-NEXT:  %[[HANDLE:.*]] = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_i32_1_0t(i32 2, i32 0, i32 -1, i32 %[[IDX]], ptr nonnull @OutArr.str)
; CHECK-NEXT:  call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_i32_1_0t.i32(target("dx.RawBuffer", i32, 1, 0) %[[HANDLE]], i32 %[[C]], i32 0, i32 %[[X]])
; CHECK-NEXT:  ret void
entry:
  %handle0 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_i32_1_0t(i32 2, i32 0, i32 -1, i32 0, ptr nonnull @OutArr.str)
  %ptr0 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %handle0, i32 %a)
  br i1 %cond, label %if.then.i, label %main

if.then.i:
  %handle1 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_i32_1_0t(i32 2, i32 0, i32 -1, i32 1, ptr nonnull @OutArr.str)
  %ptr1 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %handle1, i32 %b)
  br label %main

main:
  %ptr_phi = phi ptr [ %ptr0, %entry ], [ %ptr1, %if.then.i ]
  store i32 %x, ptr %ptr_phi, align 4
  ret void
}

; CHECK-LABEL: ptr_select_load(
; CHECK-SAME:   i1 %[[COND:.*]], i32 %[[A:.*]], i32 %[[B:.*]])
define i32 @ptr_select_load(i1 %cond, i32 %a, i32 %b) {
; CHECK-NOT: handlefromimplicitbinding
; CHECK:     entry:
; CHECK-NEXT:  %[[C:.*]] = select i1 %[[COND]], i32 %[[A]], i32 %[[B]]
; CHECK-NEXT:  %[[IDX:.*]] = select i1 %[[COND]], i32 0, i32 1
; CHECK-NEXT:  %[[HANDLE:.*]] = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_i32_1_0t(i32 2, i32 0, i32 -1, i32 %[[IDX]], ptr nonnull @OutArr.str)
; CHECK-NEXT:  %[[LOAD:.*]] = call { i32, i1 } @llvm.dx.resource.load.rawbuffer.i32.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %[[HANDLE]], i32 %[[C]], i32 0)
; CHECK-NEXT:  %[[X:.*]] = extractvalue { i32, i1 } %[[LOAD]], 0
; CHECK-NEXT:  ret i32 %[[X]]
entry:
  %handle0 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_i32_1_0t(i32 2, i32 0, i32 -1, i32 0, ptr nonnull @OutArr.str)
  %ptr0 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %handle0, i32 %a)
  %handle1 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_i32_1_0t(i32 2, i32 0, i32 -1, i32 1, ptr nonnull @OutArr.str)
  %ptr1 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %handle1, i32 %b)
  %ptr = select i1 %cond, ptr %ptr0, ptr %ptr1
  %x = load i32, ptr %ptr, align 4
  ret i32 %x
}

; CHECK-LABEL: gvn_ptr_store
; CHECK-SAME:   i32 %[[X:.*]], i1 %[[COND:.*]], i32 %[[A:.*]], i32 %[[B:.*]])
define void @gvn_ptr_store(i32 %x, i1 %cond, i32 %a, i32 %b) {
; CHECK-NOT: handlefromimplicitbinding
; CHECK:     main:
; CHECK-NEXT:  %[[C:.*]] = phi i32 [ %a, %entry ], [ %b, %if.then.i ]
; CHECK-NEXT:  %[[IDX:.*]] = phi i32 [ 0, %entry ], [ 0, %if.then.i ]
; CHECK-NEXT:  %[[HANDLE:.*]] = tail call target("dx.TypedBuffer", i32, 1, 0, 1) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0_1t(i32 2, i32 0, i32 1, i32 %[[IDX]], ptr nonnull @OutArr.str)
; CHECK-NEXT:  call void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_i32_1_0_1t.i32(target("dx.TypedBuffer", i32, 1, 0, 1) %[[HANDLE]], i32 %[[C]], i32 %[[X]])
; CHECK-NEXT:  ret void
entry:
  %handle0 = tail call target("dx.TypedBuffer", i32, 1, 0, 1) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0t(i32 2, i32 0, i32 1, i32 0, ptr nonnull @OutArr.str)
  %ptr0 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_1_0t(target("dx.TypedBuffer", i32, 1, 0, 1) %handle0, i32 %a)
  br i1 %cond, label %if.then.i, label %main

if.then.i:
  %handle1 = tail call target("dx.TypedBuffer", i32, 1, 0, 1) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0t(i32 2, i32 0, i32 1, i32 0, ptr nonnull @OutArr.str)
  %ptr1 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_1_0t(target("dx.TypedBuffer", i32, 1, 0, 1) %handle1, i32 %b)
  br label %main

main:
  %ptr = phi ptr [ %ptr0, %entry ], [ %ptr1, %if.then.i ]
  store i32 %x, ptr %ptr, align 4
  ret void
}

; CHECK-LABEL: multiple_use_handle
; CHECK-SAME:   i32 %[[X:.*]], i1 %[[COND:.*]], i32 %[[A:.*]], i32 %[[B:.*]])
define void @multiple_use_handle(i32 %x, i1 %cond, i32 %a, i32 %b) {
;   %3 = call { i32, i1 } @llvm.dx.resource.load.typedbuffer.i32.tdx.TypedBuffer_i32_1_0_1t(target("dx.TypedBuffer", i32, 1, 0, 1) %2, i32 %1)
;   %4 = extractvalue { i32, i1 } %3, 0
;   %add = add i32 %4, %x
;   call void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_i32_1_0_1t.i32(target("dx.TypedBuffer", i32, 1, 0, 1) %handle0, i32 %a, i32 %add)
;   ret void
; CHECK:     entry:
; CHECK-NEXT:  %[[HANDLE0:.*]] = tail call target("dx.TypedBuffer", i32, 1, 0, 1) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0_1t(i32 2, i32 0, i32 1, i32 0, ptr nonnull @OutArr.str)
; CHECK:     main:
; CHECK-NEXT:  %[[C:.*]] = phi i32 [ %[[A]], %entry ], [ %[[B]], %if.then.i ]
; CHECK-NEXT:  %[[IDX:.*]] = phi i32 [ 0, %entry ], [ 0, %if.then.i ]
; CHECK-NEXT:  %[[HANDLE1:.*]] = tail call target("dx.TypedBuffer", i32, 1, 0, 1) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0_1t(i32 2, i32 0, i32 1, i32 %[[IDX]], ptr nonnull @OutArr.str)
; CHECK-NEXT:  %[[LOAD:.*]] = call { i32, i1 } @llvm.dx.resource.load.typedbuffer.i32.tdx.TypedBuffer_i32_1_0_1t(target("dx.TypedBuffer", i32, 1, 0, 1) %[[HANDLE1]], i32 %[[C]])
; CHECK-NEXT:  %[[Y:.*]] = extractvalue { i32, i1 } %[[LOAD]], 0
; CHECK-NEXT:  %[[ADD:.*]] = add i32 %[[Y]], %[[X]]
; CHECK-NEXT:  call void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_i32_1_0_1t.i32(target("dx.TypedBuffer", i32, 1, 0, 1) %[[HANDLE0]], i32 %[[A]], i32 %[[ADD]])
; CHECK-NEXT:  ret void
entry:
  %handle0 = tail call target("dx.TypedBuffer", i32, 1, 0, 1) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0t(i32 2, i32 0, i32 1, i32 0, ptr nonnull @OutArr.str)
  %ptr0 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_1_0t(target("dx.TypedBuffer", i32, 1, 0, 1) %handle0, i32 %a)
  br i1 %cond, label %if.then.i, label %main

if.then.i:
  %handle1 = tail call target("dx.TypedBuffer", i32, 1, 0, 1) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0t(i32 2, i32 0, i32 1, i32 0, ptr nonnull @OutArr.str)
  %ptr1 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_1_0t(target("dx.TypedBuffer", i32, 1, 0, 1) %handle1, i32 %b)
  br label %main

main:
  %ptr = phi ptr [ %ptr0, %entry ], [ %ptr1, %if.then.i ]
  %y = load i32, ptr %ptr, align 4
  %add = add i32 %y, %x
  store i32 %add, ptr %ptr0, align 4
  ret void
}
