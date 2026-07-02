; RUN: opt -S -dxil-remove-unused-resources %s -o - | FileCheck %s
; RUN: llc %s -o - | FileCheck %s --check-prefix=CHECK-PRINT

target triple = "dxil-pc-shadermodel6.6-compute"

; This test makes sure that we remove:
; - unused resource initialization calls
; - unused cbuffers and associated global variables
; - resource name strings

%__cblayout_CB = type <{ i32, float }>
%"__cblayout_$Globals" = type <{ i32, float }>

@CB.cb = internal global target("dx.CBuffer", %__cblayout_CB) poison
@CB.str = private unnamed_addr constant [3 x i8] c"CB\00", align 1
@"$Globals.cb" = internal global target("dx.CBuffer", %"__cblayout_$Globals") poison
@"$Globals.str" = private unnamed_addr constant [9 x i8] c"$Globals\00", align 1
@_ZL3Buf = internal unnamed_addr global target("dx.RawBuffer", i16, 1, 0) poison, align 4
@Buf.str = private unnamed_addr constant [4 x i8] c"Buf\00", align 1

define void @main() {
entry:

; CBuffers
  %cb_handle0 = tail call target("dx.CBuffer", %__cblayout_CB) @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_CBst(i32 0, i32 0, i32 1, i32 0, ptr nonnull @CB.str)
  store target("dx.CBuffer", %__cblayout_CB) %cb_handle0, ptr @CB.cb, align 4

  %cb_handle1 = tail call target("dx.CBuffer", %"__cblayout_$Globals") @"llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_$Globalsst"(i32 2, i32 0, i32 1, i32 0, ptr nonnull @"$Globals.str")
  store target("dx.CBuffer", %"__cblayout_$Globals") %cb_handle1, ptr @"$Globals.cb", align 4

; SrvBuffer
  %srv_handle = call target("dx.TypedBuffer", i64, 0, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 5, i32 20, i32 1, i32 0, ptr null)

; UavBuffer
  %uav_handle = call target("dx.RawBuffer", i16, 1, 0)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 6, i32 1, i32 0, ptr @Buf.str)
  store target("dx.RawBuffer", i16, 1, 0) %uav_handle, ptr @_ZL3Buf, align 4

  ret void
}

!hlsl.cbs = !{!0, !1}

; This is a list of all the constant buffers in the shader.
; The null values in the list indicate that the corresponding constant is not used in the shader.
!0 = distinct !{ptr @CB.cb, null, null}
!1 = distinct !{ptr @"$Globals.cb", null, null}

; CHECK-NOT: @CB.cb
; CHECK-NOT: @CB.str
; CHECK-NOT: @"$Globals.cb"
; CHECK-NOT: @"$Globals.str" 
; CHECK-NOT: @_ZL3Buf
; CHECK-NOT: @Buf.str  
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
