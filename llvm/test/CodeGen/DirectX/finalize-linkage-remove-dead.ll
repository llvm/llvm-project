; RUN: opt -S -dxil-finalize-linkage -mtriple=dxil-unknown-shadermodel6.5-compute %s | FileCheck %s
; RUN: llc %s --filetype=asm -o - | FileCheck %s

target triple = "dxilv1.5-pc-shadermodel6.5-compute"

; Confirm that DXILFinalizeLinkage will remove functions that have compatible
; linkage and are not called from anywhere. This should be any function that
; is not explicitly marked noinline or export and is not an entry point.

; Not called nor marked with any linking or inlining attributes.
; CHECK-NOT: define {{.*}}doNothingNothing
define void @"?doNothingNothing@@YAXXZ"() #0 {
entry:
  ret void
}

; Marked internal, this should be removed.
; CHECK-NOT: define {{.*}}doNothingInternally
define internal void @"?doNothingInternally@@YAXXZ"() #0 {
entry:
  ret void
}

; Marked external, which should become internal and be removed.
; CHECK-NOT: define {{.*}}doNothingExternally
define external void @"?doNothingExternally@@YAXXZ"() #0 {
entry:
  ret void
}

; Not called nor marked with any linking or inlining attributes.
; CHECK: define internal void @"?doSomethingSomething@@YAXXZ"() #0
define void @"?doSomethingSomething@@YAXXZ"() #0 {
entry:
  ret void
}

; Marked internal, this should be removed.
; CHECK: define internal void @"?doSomethingInternally@@YAXXZ"() #0
define internal void @"?doSomethingInternally@@YAXXZ"() #0 {
entry:
  ret void
}

; Marked external, which should become internal and be removed.
; CHECK: define internal void @"?doSomethingExternally@@YAXXZ"() #0
define external void @"?doSomethingExternally@@YAXXZ"() #0 {
entry:
  ret void
}

; Lacks alwaysinline attribute. Should remain.
; CHECK: define internal void @"?doNothingDefault@@YAXXZ"() #1
define void @"?doNothingDefault@@YAXXZ"() #1 {
entry:
  ret void
}

; Has noinline attribute. Should remain.
; CHECK: define {{.*}}doNothingNoinline
define void @"?doNothingNoinline@@YAXXZ"() #2 {
entry:
  ret void
}

; Entry point function should stay.
; CHECK: define void @main() #3
define void @main() #3 {
entry:
  call void @"?doSomethingSomething@@YAXXZ"() #4
  call void @"?doSomethingInternally@@YAXXZ"() #4
  call void @"?doSomethingExternally@@YAXXZ"() #4
  ret void
}

attributes #0 = { alwaysinline convergent norecurse nounwind }
attributes #1 = { convergent norecurse nounwind }
attributes #2 = { convergent noinline norecurse nounwind }
attributes #3 = { convergent noinline norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
attributes #4 = { convergent }
