; RUN: opt -S -dxil-resource-access -mtriple=dxil %s | FileCheck %s
;
; Test for when we have indices into both the array and the vector: ie, s[1][3]

; cbuffer CB : register(b0) {
;   uint4 s[3]; // offset   0,  size 16        * 3
; }
%__cblayout_CB = type <{ [2 x <4 x i32>] }>

@CB.cb = local_unnamed_addr global target("dx.CBuffer", %__cblayout_CB) poison

; CHECK: define void @f
define void @f(ptr %dst) {
entry:
  %CB.cb_h = tail call target("dx.CBuffer", %__cblayout_CB) @llvm.dx.resource.handlefromimplicitbinding(i32 1, i32 0, i32 1, i32 0, ptr null)
  store target("dx.CBuffer", %__cblayout_CB) %CB.cb_h, ptr @CB.cb, align 4

  ; CHECK: [[CB:%.*]] = load target("dx.CBuffer", %__cblayout_CB), ptr @CB.cb
  %CB.cb = load target("dx.CBuffer", %__cblayout_CB), ptr @CB.cb, align 4

  ;; s[1][3]
  ;
  ; CHECK: [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", %__cblayout_CB) [[CB]], i32 1)
  ; CHECK: [[X:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 3
  ; CHECK: store i32 [[X]], ptr %dst
  %i8_ptr = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %__cblayout_CB) %CB.cb, i32 0)
  %i8_gep = getelementptr inbounds nuw i8, ptr addrspace(2) %i8_ptr, i32 28
  %i8_vecext = load i32, ptr addrspace(2) %i8_gep, align 4
  store i32 %i8_vecext, ptr %dst, align 4

  ;; s[2].w
  ;
  ; CHECK: [[LOAD:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.resource.load.cbufferrow.4.{{.*}}(target("dx.CBuffer", %__cblayout_CB) [[CB]], i32 2)
  ; CHECK: [[X:%.*]] = extractvalue { i32, i32, i32, i32 } [[LOAD]], 3
  ;;
  ;; It would be nice to avoid the redundant vector creation here, but that's
  ;; outside of the scope of this pass.
  ;;
  ; CHECK: [[X_VEC:%.*]] = insertelement <4 x i32> {{%.*}}, i32 [[X]], i32 3
  ; CHECK: [[X_EXT:%.*]] = extractelement <4 x i32> [[X_VEC]], i32 3
  ; CHECK: store i32 [[X_EXT]], ptr %dst
  %typed_ptr = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %__cblayout_CB) %CB.cb, i32 0)
  %typed_gep = getelementptr <4 x i32>, ptr addrspace(2) %typed_ptr, i32 2
  %typed_load = load <4 x i32>, ptr addrspace(2) %typed_gep, align 16
  %typed_vecext = extractelement <4 x i32> %typed_load, i32 3
  store i32 %typed_vecext, ptr %dst, align 4

  ret void
}
