; RUN: llc %s -stop-after=dxil-resource-access -o - | FileCheck %s
; RUN: llc %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-PRINT

target triple = "dxil-pc-shadermodel6.6-compute"

; This test makes sure that if resources are referenced in a function that
; is not used, then after the function is optimized out we also remove:
; - unused resource initialization calls
; - unused associated global resource variables
; - resource name strings

%__cblayout_CB = type <{ i32, float }>
%"__cblayout_$Globals" = type <{ i32, float }>

@CB.cb = internal global target("dx.CBuffer", %__cblayout_CB) poison
@c = external hidden local_unnamed_addr addrspace(2) global i32, align 4
@CB.str = private unnamed_addr constant [3 x i8] c"CB\00", align 1
@_ZL3Buf.0 = internal unnamed_addr global target("dx.TypedBuffer", float, 1, 0, 0) poison, align 4
@.str = private unnamed_addr constant [4 x i8] c"Buf\00", align 1
@a = external hidden local_unnamed_addr addrspace(2) global i32, align 4
@"$Globals.cb" = internal global target("dx.CBuffer", %"__cblayout_$Globals") poison
@"$Globals.str" = private unnamed_addr constant [9 x i8] c"$Globals\00", align 1

; Unused function that references resources
define hidden void @_Z3foov() local_unnamed_addr #1 {
entry:
  %0 = load i32, ptr addrspace(2) @a, align 4
  %1 = load i32, ptr addrspace(2) @c, align 4
  %add = add nsw i32 %1, %0
  %conv = sitofp i32 %add to float
  %2 = load target("dx.TypedBuffer", float, 1, 0, 0), ptr @_ZL3Buf.0, align 4
  %3 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_f32_1_0_0t.i32(target("dx.TypedBuffer", float, 1, 0, 0) %2, i32 0)
  store float %conv, ptr %3, align 4
  ret void
}

define void @main() {
entry:

; CBuffers
  %cb_handle0 = tail call target("dx.CBuffer", %__cblayout_CB) @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_CBst(i32 0, i32 0, i32 1, i32 0, ptr nonnull @CB.str)
  store target("dx.CBuffer", %__cblayout_CB) %cb_handle0, ptr @CB.cb, align 4

  %uav_handle = tail call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  store target("dx.TypedBuffer", float, 1, 0, 0) %uav_handle, ptr @_ZL3Buf.0, align 4
  
  %cb_handle1 = tail call target("dx.CBuffer", %"__cblayout_$Globals") @"llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_$Globalsst"(i32 1, i32 0, i32 1, i32 0, ptr nonnull @"$Globals.str")
  store target("dx.CBuffer", %"__cblayout_$Globals") %cb_handle1, ptr @"$Globals.cb", align 4

  ret void
}

!hlsl.cbs = !{!0, !1}

; This is a list of all the constant buffers in the shader.
; The null values in the list indicate that the corresponding constant is not used in the shader.
!0 = distinct !{ptr @CB.cb, ptr addrspace(2) @c, null}
!1 = distinct !{ptr @"$Globals.cb", ptr addrspace(2) @a, null}

; Make sure the resource globals and initialization calls are removed from the module
; CHECK-NOT: @CB.cb
; CHECK-NOT: @c
; CHECK-NOT: @CB.str
; CHECK-NOT: @_ZL3Buf.0
; CHECK-NOT: @.str
; CHECK-NOT: @a
; CHECK-NOT: @"$Globals.cb"
; CHECK-NOT: @"$Globals.str" 
; CHECK-NOT: call {{.*}} llvm.dx.resource.handlefrombinding
; CHECK-NOT: call {{.*}} llvm.dx.resource.handlefromimplicitbinding

; Make sure the resource bindings table is empty
; CHECK-PRINT: ; Resource Bindings:
; CHECK-PRINT-NEXT: ;
; CHECK-PRINT-NEXT: ; Name                                  Type  Format         Dim      ID      HLSL Bind     Count
; CHECK-PRINT-NEXT: ; ------------------------------ ---------- ------- ----------- ------- -------------- ---------
; CHECK-PRINT-NEXT: ;
; CHECK-PRINT-NEXT: ; ModuleID
; CHECK-PRINT-NOT: ; Buf
; CHECK-PRINT-NOT: ; CB
; CHECK-PRINT-NOT: ; $Globals
