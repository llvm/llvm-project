; RUN: opt -S -dxil-finalize-linkage -mtriple=dxil-unknown-shadermodel6.5-compute %s | FileCheck %s
; RUN: llc %s --filetype=asm -o - | FileCheck %s --check-prefixes=CHECK-LLC

target triple = "dxilv1.5-pc-shadermodel6.5-compute"

; DXILFinalizeLinkage changes linkage of all functions that are not
; entry points or exported function to internal.

; CHECK: define internal void @"?f1@@YAXXZ"()
define void @"?f1@@YAXXZ"() #0 {
entry:
  ret void
}

; CHECK: define internal void @"?f2@@YAXXZ"()
define void @"?f2@@YAXXZ"() #0 {
entry:
  ret void
}

; CHECK: define internal void @"?f3@@YAXXZ"()
define void @"?f3@@YAXXZ"() #0 {
entry:
  ret void
}

; CHECK: define internal void @"?foo@@YAXXZ"()
define void @"?foo@@YAXXZ"() #0 {
entry:
  call void @"?f2@@YAXXZ"() #3
  ret void
}

; Exported function - do not change linkage
; CHECK: define void @"?bar@@YAXXZ"()
define void @"?bar@@YAXXZ"() #1 {
entry:
  call void @"?f3@@YAXXZ"() #3
  ret void
}

; CHECK: define internal void @"?main@@YAXXZ"() #0
define internal void @"?main@@YAXXZ"() #0 {
entry:
  call void @"?foo@@YAXXZ"() #3
  call void @"?bar@@YAXXZ"() #3
  ret void
}

; Entry point function - do not change linkage
; CHECK: define void @main() #2
define void @main() #2 {
entry:
  call void @"?main@@YAXXZ"()
  ret void
}

attributes #0 = { convergent noinline nounwind optnone}
attributes #1 = { convergent noinline nounwind optnone "hlsl.export"}
attributes #2 = { convergent "hlsl.numthreads"="4,1,1" "hlsl.shader"="compute"}
attributes #3 = { convergent }

; Make sure "hlsl.export" attribute is stripped by llc
; CHECK-LLC-NOT: "hlsl.export"
