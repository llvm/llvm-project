; RUN: opt -S -dxil-finalize-linkage -mtriple=dxil-unknown-shadermodel6.5-compute %s | FileCheck %s
; RUN: llc %s --filetype=asm -o - | FileCheck %s

target triple = "dxilv1.5-pc-shadermodel6.5-compute"

; Confirm that DXILFinalizeLinkage will remove functions that have compatible
; linkage and are not called from anywhere. This should be any function that
; is not an entry point.

; Has no specified inlining/linking behavior and is uncalled, this should be removed.
; CHECK-NOT: define {{.*}}doNothingUncalled
define void @"?doNothingUncalled@@YAXXZ"() #1 {
entry:
  ret void
}

; Alwaysinline and uncalled, this should be removed.
; CHECK-NOT: define {{.*}}doAlwaysInlineUncalled
define void @"?doAlwaysInlineUncalled@@YAXXZ"() #0 {
entry:
  ret void
}

; Noinline and uncalled, this should be removed.
; CHECK-NOT: define {{.*}}doNoinlineUncalled
define void @"?doNoinlineUncalled@@YAXXZ"() #3 {
entry:
  ret void
}

; No inlining attribute, internal, and uncalled; this should be removed.
; CHECK-NOT: define {{.*}}doInternalUncalled
define internal void @"?doInternalUncalled@@YAXXZ"() #1 {
entry:
  ret void
}

; Alwaysinline, internal, and uncalled; this should be removed.
; CHECK-NOT: define {{.*}}doAlwaysInlineInternalUncalled
define internal void @"?doAlwaysInlineInternalUncalled@@YAXXZ"() #0 {
entry:
  ret void
}

; Noinline, internal, and uncalled; this should be removed.
; CHECK-NOT: define {{.*}}doNoinlineInternalUncalled
define internal void @"?doNoinlineInternalUncalled@@YAXXZ"() #3 {
entry:
  ret void
}

; Marked external and uncalled, this should become internal and be removed.
; CHECK-NOT: define {{.*}}doExternalUncalled
define external void @"?doExternalUncalled@@YAXXZ"() #1 {
entry:
  ret void
}

; Alwaysinline, external and uncalled, this should become internal and be removed.
; CHECK-NOT: define {{.*}}doAlwaysInlineExternalUncalled
define external void @"?doAlwaysInlineExternalUncalled@@YAXXZ"() #0 {
entry:
  ret void
}

; Noinline, external and uncalled, this should become internal and be removed.
; CHECK-NOT: define {{.*}}doNoinlineExternalUncalled
define external void @"?doNoinlineExternalUncalled@@YAXXZ"() #3 {
entry:
  ret void
}

; No inlining attribute and called, this should stay.
; CHECK: define {{.*}}doNothingCalled
define void @"?doNothingCalled@@YAXXZ"() #1 {
entry:
  ret void
}

; Alwaysinline and called, this should stay.
; CHECK: define {{.*}}doAlwaysInlineCalled
define void @"?doAlwaysInlineCalled@@YAXXZ"() #0 {
entry:
  ret void
}

; Noinline and called, this should stay.
; CHECK: define {{.*}}doNoinlineCalled
define void @"?doNoinlineCalled@@YAXXZ"() #3 {
entry:
  ret void
}

; No inlining attribute, internal, and called; this should stay.
; CHECK: define {{.*}}doInternalCalled
define internal void @"?doInternalCalled@@YAXXZ"() #1 {
entry:
  ret void
}

; Alwaysinline, internal, and called; this should stay.
; CHECK: define {{.*}}doAlwaysInlineInternalCalled
define internal void @"?doAlwaysInlineInternalCalled@@YAXXZ"() #0 {
entry:
  ret void
}

; Noinline, internal, and called; this should stay.
; CHECK: define {{.*}}doNoinlineInternalCalled
define internal void @"?doNoinlineInternalCalled@@YAXXZ"() #3 {
entry:
  ret void
}

; Marked external and called, this should become internal and stay.
; CHECK: define {{.*}}doExternalCalled
define external void @"?doExternalCalled@@YAXXZ"() #1 {
entry:
  ret void
}

; Always inlined, external and called, this should become internal and stay.
; CHECK: define {{.*}}doAlwaysInlineExternalCalled
define external void @"?doAlwaysInlineExternalCalled@@YAXXZ"() #0 {
entry:
  ret void
}

; Noinline, external and called, this should become internal and stay.
; CHECK: define {{.*}}doNoinlineExternalCalled
define external void @"?doNoinlineExternalCalled@@YAXXZ"() #3 {
entry:
  ret void
}

; Entry point function, this should stay.
; CHECK: define void @main()
define void @main() #4 {
entry:
  call void @"?doNothingCalled@@YAXXZ"() #5
  call void @"?doAlwaysInlineCalled@@YAXXZ"() #5
  call void @"?doNoinlineCalled@@YAXXZ"() #5
  call void @"?doInternalCalled@@YAXXZ"() #5
  call void @"?doAlwaysInlineInternalCalled@@YAXXZ"() #5
  call void @"?doNoinlineInternalCalled@@YAXXZ"() #5
  call void @"?doExternalCalled@@YAXXZ"() #5
  call void @"?doAlwaysInlineExternalCalled@@YAXXZ"() #5
  call void @"?doNoinlineExternalCalled@@YAXXZ"() #5
  ret void
}

attributes #0 = { alwaysinline convergent norecurse nounwind }
attributes #1 = { convergent norecurse nounwind }
attributes #3 = { convergent noinline norecurse nounwind }
attributes #4 = { convergent noinline norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
attributes #5 = { convergent }
