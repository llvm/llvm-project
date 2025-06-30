; RUN: opt -S -dxil-finalize-linkage -mtriple=dxil-unknown-shadermodel6.5-library %s | FileCheck %s
; RUN: llc %s --filetype=asm -o - | FileCheck %s

target triple = "dxilv1.5-pc-shadermodel6.5-compute"

; Confirm that DXILFinalizeLinkage will remove functions that have compatible
; linkage and are not called from anywhere. This should be any function that
; is marked hidden or internal.

; Is hidden, and uncalled, this should be removed.
; CHECK-NOT: define {{.*}}doNothingUncalled
define hidden void @"?doNothingUncalled@@YAXXZ"() #2 {
entry:
  ret void
}

; Alwaysinline, hidden and uncalled, this should be removed.
; CHECK-NOT: define {{.*}}doAlwaysInlineUncalled
define hidden void @"?doAlwaysInlineUncalled@@YAXXZ"() #0 {
entry:
  ret void
}

; Noinline, hidden and uncalled, this should be removed.
; CHECK-NOT: define {{.*}}doNoinlineUncalled
define hidden void @"?doNoinlineUncalled@@YAXXZ"() #4 {
entry:
  ret void
}

; No inlining attribute, internal, and uncalled; this should be removed.
; CHECK-NOT: define {{.*}}doInternalUncalled
define internal void @"?doInternalUncalled@@YAXXZ"() #2 {
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
define internal void @"?doNoinlineInternalUncalled@@YAXXZ"() #4 {
entry:
  ret void
}

; Marked external, hidden and uncalled, this should become internal and be removed.
; CHECK-NOT: define {{.*}}doExternalUncalled
define external hidden void @"?doExternalUncalled@@YAXXZ"() #2 {
entry:
  ret void
}

; Alwaysinline, external, hidden and uncalled, this should become internal and be removed.
; CHECK-NOT: define {{.*}}doAlwaysInlineExternalUncalled
define external hidden void @"?doAlwaysInlineExternalUncalled@@YAXXZ"() #0 {
entry:
  ret void
}

; Noinline, external, hidden and uncalled, this should become internal and be removed.
; CHECK-NOT: define {{.*}}doNoinlineExternalUncalled
define external hidden void @"?doNoinlineExternalUncalled@@YAXXZ"() #4 {
entry:
  ret void
}

; No inlining attribute, hidden and called, this should stay.
; CHECK: define {{.*}}doNothingCalled
define hidden void @"?doNothingCalled@@YAXXZ"() #2 {
entry:
  ret void
}

; Alwaysinline, hidden and called, this should stay.
; CHECK: define {{.*}}doAlwaysInlineCalled
define hidden void @"?doAlwaysInlineCalled@@YAXXZ"() #0 {
entry:
  ret void
}

; Noinline, hidden and called, this should stay.
; CHECK: define {{.*}}doNoinlineCalled
define hidden void @"?doNoinlineCalled@@YAXXZ"() #4 {
entry:
  ret void
}

; No inlining attribute, internal, and called; this should stay.
; CHECK: define {{.*}}doInternalCalled
define internal void @"?doInternalCalled@@YAXXZ"() #2 {
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
define internal void @"?doNoinlineInternalCalled@@YAXXZ"() #4 {
entry:
  ret void
}

; Marked external, hidden and called, this should become internal and stay.
; CHECK: define {{.*}}doExternalCalled
define external hidden void @"?doExternalCalled@@YAXXZ"() #2 {
entry:
  ret void
}

; Always inlined, external, hidden and called, this should become internal and stay.
; CHECK: define {{.*}}doAlwaysInlineExternalCalled
define external hidden void @"?doAlwaysInlineExternalCalled@@YAXXZ"() #0 {
entry:
  ret void
}

; Noinline, external, hidden and called, this should become internal and stay.
; CHECK: define {{.*}}doNoinlineExternalCalled
define external hidden void @"?doNoinlineExternalCalled@@YAXXZ"() #4 {
entry:
  ret void
}

; No inlining attribute and exported, this should stay.
; CHECK: define {{.*}}doNothingExported
define void @"?doNothingExported@@YAXXZ"() #3 {
entry:
  ret void
}

; Alwaysinline and exported, this should stay.
; CHECK: define {{.*}}doAlwaysInlineExported
define void @"?doAlwaysInlineExported@@YAXXZ"() #1 {
entry:
  ret void
}

; Noinline attribute and exported, this should stay.
; CHECK: define {{.*}}doNoinlineExported
define void @"?doNoinlineExported@@YAXXZ"() #5 {
entry:
  ret void
}

; Marked external and exported, this should stay.
; CHECK: define {{.*}}doExternalExported
define external void @"?doExternalExported@@YAXXZ"() #3 {
entry:
  ret void
}

; Alwaysinline, external and exported, this should stay.
; CHECK: define {{.*}}doAlwaysInlineExternalExported
define external void @"?doAlwaysInlineExternalExported@@YAXXZ"() #1 {
entry:
  ret void
}

; Noinline, external and exported, this should stay.
; CHECK: define {{.*}}doNoinlineExternalExported
define external void @"?doNoinlineExternalExported@@YAXXZ"() #5 {
entry:
  ret void
}

; Entry point function, this should stay.
; CHECK: define void @main()
define void @main() #6 {
entry:
  call void @"?doNothingCalled@@YAXXZ"() #7
  call void @"?doAlwaysInlineCalled@@YAXXZ"() #7
  call void @"?doNoinlineCalled@@YAXXZ"() #7
  call void @"?doInternalCalled@@YAXXZ"() #7
  call void @"?doAlwaysInlineInternalCalled@@YAXXZ"() #7
  call void @"?doNoinlineInternalCalled@@YAXXZ"() #7
  call void @"?doExternalCalled@@YAXXZ"() #7
  call void @"?doAlwaysInlineExternalCalled@@YAXXZ"() #7
  call void @"?doNoinlineExternalCalled@@YAXXZ"() #7
  ret void
}

attributes #0 = { alwaysinline convergent norecurse nounwind }
attributes #1 = { alwaysinline convergent norecurse nounwind }
attributes #2 = { convergent norecurse nounwind }
attributes #3 = { convergent norecurse nounwind }
attributes #4 = { convergent noinline norecurse nounwind }
attributes #5 = { convergent noinline norecurse nounwind }
attributes #6 = { convergent noinline norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
attributes #7 = { convergent }
