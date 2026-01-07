; RUN: opt -S -dxil-finalize-linkage -mtriple=dxil-unknown-shadermodel6.5-compute %s | FileCheck %s
; RUN: llc %s --filetype=asm -o - | FileCheck %s --check-prefixes=CHECK-LLC

target triple = "dxilv1.5-pc-shadermodel6.5-compute"

; DXILFinalizeLinkage changes linkage of all functions that are hidden to
; internal, converts private globals to internal linkage, and converts external globals 
; with no usage to internal linkage.

; CHECK: @switch.table = internal unnamed_addr constant [4 x i32]
@switch.table = private unnamed_addr constant [4 x i32] [i32 1, i32 257, i32 65793, i32 16843009], align 4

; CHECK: @private_array = internal constant [3 x float]
@private_array = private constant [3 x float] [float 1.0, float 2.0, float 3.0], align 4

; CHECK: @private_var = internal global i32
@private_var = private global i32 1, align 4

; Internal global should remain internal
; CHECK: @internal_var = internal global i32
@internal_var = internal global i32 1, align 4

; External global should remain external
; CHECK: @external_var = external global i32
@external_var = external global i32, align 4

; Hidden global should remain hidden
; CHECK: @hidden_var = hidden global i32
@hidden_var = hidden global i32 1, align 4

; Running the whole pipeline should remove unused global variables

; CHECK: @aTile = internal addrspace(3) global
; CHECK-LLC-NOT: @aTile
@aTile = hidden addrspace(3) global [4 x [1 x i32]] zeroinitializer, align 4

; CHECK: @bTile = internal addrspace(3) global
; CHECK-LLC-NOT: @bTile
@bTile = hidden addrspace(3) global [1 x [1 x i32]] zeroinitializer, align 4

define void @anchor_function() #0 {
entry:
  %0 = load i32, ptr @switch.table, align 4
  %1 = load [3 x float], ptr @private_array, align 4
  %2 = load i32, ptr @private_var, align 4
  %3 = load i32, ptr @internal_var, align 4
  %4 = load i32, ptr @external_var, align 4
  %5 = load i32, ptr @hidden_var, align 4
  ret void
}

; CHECK-NOT: define internal void @"?f1@@YAXXZ"()
define void @"?f1@@YAXXZ"() #0 {
entry:
  ret void
}

; CHECK: define internal void @"?f2@@YAXXZ"()
define hidden void @"?f2@@YAXXZ"() #0 {
entry:
  ret void
}

; CHECK: define internal void @"?f3@@YAXXZ"()
define hidden void @"?f3@@YAXXZ"() #0 {
entry:
  ret void
}

; CHECK: define internal void @"?foo@@YAXXZ"()
define hidden void @"?foo@@YAXXZ"() #0 {
entry:
  call void @"?f2@@YAXXZ"() #3
  ret void
}

; Exported function - do not change linkage
; CHECK: define void @"?bar@@YAXXZ"()
define void @"?bar@@YAXXZ"() #0 {
entry:
  call void @"?f3@@YAXXZ"() #3
  ret void
}

; CHECK: define internal void @"?main@@YAXXZ"() #0
define internal void @"?main@@YAXXZ"() #0 {
entry:
  call void @"?foo@@YAXXZ"() #2
  call void @"?bar@@YAXXZ"() #2
  ret void
}

; Entry point function - do not change linkage
; CHECK: define void @main() #1
define void @main() #1 {
entry:
  call void @"?main@@YAXXZ"()
  ret void
}

attributes #0 = { convergent noinline nounwind optnone}
attributes #1 = { convergent "hlsl.numthreads"="4,1,1" "hlsl.shader"="compute"}
attributes #2 = { convergent }

; Make sure "hlsl.export" attribute is stripped by llc
; CHECK-LLC-NOT: "hlsl.export"
