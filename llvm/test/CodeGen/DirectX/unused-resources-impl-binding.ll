; RUN: llc -stop-after=dxil-resource-access %s -o - | FileCheck %s
; RUN: llc %s -o - | FileCheck %s --check-prefix=CHECK-PRINT

target triple = "dxil-pc-shadermodel6.6-compute"

; This test makes sure that unused resource removed in dxil-resource-access
; pass do not affect implicit resource bindings.
; For constant buffers declared in the order {CB_Unused, CB, $Globals} where only
; CB is used, the CB should be the only resource that remains in the resource bindings
; table and should be assigned slot cb0.
; Similarly, for RWBuffer resources declared in order {Buf_Unused, Buf} where only Buf
; is used, the Buf resource should be the only UAV resource in the resource bindings table 
; and should be assigned slot u0.

%__cblayout_CB_Unused = type <{ i32 }>
%__cblayout_CB = type <{ i32 }>
%"__cblayout_$Globals" = type <{ i32 }>

@CB_Unused.cb = internal global target("dx.CBuffer", %__cblayout_CB_Unused) poison
@CB_Unused.str = private unnamed_addr constant [10 x i8] c"CB_Unused\00", align 1
@CB.cb = internal global target("dx.CBuffer", %__cblayout_CB) poison
@c = external hidden local_unnamed_addr addrspace(2) global i32, align 4
@CB.str = private unnamed_addr constant [3 x i8] c"CB\00", align 1
@_ZL10Buf_Unused.0 = internal unnamed_addr global target("dx.TypedBuffer", i32, 1, 0, 1) poison, align 4
@.str = private unnamed_addr constant [11 x i8] c"Buf_Unused\00", align 1
@_ZL3Buf.0 = internal unnamed_addr global target("dx.TypedBuffer", i32, 1, 0, 1) poison, align 4
@.str.2 = private unnamed_addr constant [4 x i8] c"Buf\00", align 1
@a = external hidden local_unnamed_addr addrspace(2) global i32, align 4
@"$Globals.cb" = internal global target("dx.CBuffer", %"__cblayout_$Globals") poison
@"$Globals.str" = private unnamed_addr constant [9 x i8] c"$Globals\00", align 1

; Unused function foo
define hidden void @_Z3foov() {
entry:
  ; Use constant @a from $Globals and buffer Buf_Unused
  %0 = load i32, ptr addrspace(2) @a, align 4
  %1 = load target("dx.TypedBuffer", i32, 1, 0, 1), ptr @_ZL10Buf_Unused.0, align 4
  %2 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_1_0_1t.i32(target("dx.TypedBuffer", i32, 1, 0, 1) %1, i32 0)
  store i32 %0, ptr %2, align 4
  ret void
}

define void @main() {
entry:
  %cb_handle0 = tail call target("dx.CBuffer", %__cblayout_CB_Unused) @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_CB_Unusedst(i32 0, i32 0, i32 1, i32 0, ptr nonnull @CB_Unused.str)
  store target("dx.CBuffer", %__cblayout_CB_Unused) %cb_handle0, ptr @CB_Unused.cb, align 4

  %cb_handle1 = tail call target("dx.CBuffer", %__cblayout_CB) @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_CBst(i32 1, i32 0, i32 1, i32 0, ptr nonnull @CB.str)
  store target("dx.CBuffer", %__cblayout_CB) %cb_handle1, ptr @CB.cb, align 4

  %uav_handle0 = tail call target("dx.TypedBuffer", i32, 1, 0, 1) @llvm.dx.resource.handlefromimplicitbinding.tdx.TypedBuffer_i32_1_0_1t(i32 2, i32 0, i32 1, i32 0, ptr nonnull @.str)
  store target("dx.TypedBuffer", i32, 1, 0, 1) %uav_handle0, ptr @_ZL10Buf_Unused.0, align 4

  %uav_handle1 = tail call target("dx.TypedBuffer", i32, 1, 0, 1) @llvm.dx.resource.handlefromimplicitbinding.tdx.TypedBuffer_i32_1_0_1t(i32 3, i32 0, i32 1, i32 0, ptr nonnull @.str.2)
  store target("dx.TypedBuffer", i32, 1, 0, 1) %uav_handle1, ptr @_ZL3Buf.0, align 4

  %cb_handle2 = tail call target("dx.CBuffer", %"__cblayout_$Globals") @"llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_$Globalsst"(i32 4, i32 0, i32 1, i32 0, ptr nonnull @"$Globals.str")
  store target("dx.CBuffer", %"__cblayout_$Globals") %cb_handle2, ptr @"$Globals.cb", align 4

  ; Use constant @c from CB and buffer Buf
  %1 = load i32, ptr addrspace(2) @c, align 4
  %2 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_1_0_1t.i32(target("dx.TypedBuffer", i32, 1, 0, 1) %uav_handle1, i32 0)
  store i32 %1, ptr %2, align 4
  ret void
}

!hlsl.cbs = !{!0, !1, !2}

; This is a list of all the constant buffers in the shader.
; The null values in the list indicate that the corresponding constant is not used in the shader.
!0 = distinct !{ptr @CB_Unused.cb, null}
!1 = !{ptr @CB.cb, ptr addrspace(2) @c}
!2 = !{ptr @"$Globals.cb", ptr addrspace(2) @a}

; Make sure the unused resource globals and initialization calls are removed from the module
; and the used ones are retained.
; CHECK-NOT: @CB_Unused.cb
; CHECK-NOT: @CB_Unused.str
; CHECK-NOT: @_ZL10Buf_Unused.0
; CHECK-NOT: @.str
; CHECK-NOT: @"$Globals.cb"
; CHECK-NOT: @"$Globals.str" 

; CHECK: @CB.cb = internal global target("dx.CBuffer", %__cblayout_CB) poison
; CHECK: @CB.str = internal unnamed_addr constant [3 x i8] c"CB\00", align 1
; CHECK: @_ZL3Buf.0 = internal unnamed_addr global target("dx.TypedBuffer", i32, 1, 0, 1) poison, align 4
; CHECK: @.str.2 = internal unnamed_addr constant [4 x i8] c"Buf\00", align 1

; CHECK-NOT: {{.*}} call {{.*}} @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_CB_Unusedst({{.*}}, ptr nonnull @CB_unused.str)
; CHECK: call {{.*}} @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_CBst({{.*}}, ptr nonnull @CB.str)
; CHECK-NOT: call {{.*}} @"llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_$Globalsst"({{.*}}, ptr nonnull @"$Globals.str")

; CHECK-NOT: call {{.*}} @llvm.dx.resource.handlefromimplicitbinding.tdx.TypedBuffer_i32_1_0_1t({{.*}}, ptr nonnull @.str)
; CHECK: call {{.*}} @llvm.dx.resource.handlefromimplicitbinding.tdx.TypedBuffer_i32_1_0_1t({{.*}}, ptr nonnull @.str.2)
  
; Make sure the resource bindings table contain only the used resources with correct binding information.

; CHECK-PRINT: ; Resource Bindings:
; CHECK-PRINT-NEXT: ;
; CHECK-PRINT-NEXT: ; Name                                 Type  Format         Dim      ID      HLSL Bind     Count
; CHECK-PRINT-NEXT: ; ------------------------------ ---------- ------- ----------- ------- -------------- ---------
; CHECK-PRINT-NEXT: ; Buf                                   UAV     i32         buf      U0             u0         1
; CHECK-PRINT-NEXT: ; CB                                cbuffer      NA          NA     CB0            cb0         1
; CHECK-PRINT-NEXT: ;
; CHECK-PRINT-NEXT: ; ModuleID

; CHECK-PRINT-NOT: ; Buf_Unused
; CHECK-PRINT-NOT: ; CB_Unused
; CHECK-PRINT-NOT: ; $Globals
