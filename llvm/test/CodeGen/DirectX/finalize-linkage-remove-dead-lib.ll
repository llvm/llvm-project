; RUN: opt -S -dxil-finalize-linkage -mtriple=dxil-unknown-shadermodel6.5-library %s | FileCheck %s
; RUN: llc %s --filetype=asm -o - | FileCheck %s

target triple = "dxilv1.5-pc-shadermodel6.5-compute"

; Confirm that DXILFinalizeLinkage will remove functions that have compatible
; linkage and are not called from anywhere. This should be any function that
; is not explicitly marked export and is not an entry point.

; Has no specified inlining/linking behavior and is uncalled, this should be removed.
; CHECK-NOT: define {{.*}}doNothingUncalled
define void @"?doNothingUncalled@@YAXXZ"() #2 {
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
define void @"?doNoinlineUncalled@@YAXXZ"() #4 {
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

; Marked external and uncalled, this should become internal and be removed.
; CHECK-NOT: define {{.*}}doExternalUncalled
define external void @"?doExternalUncalled@@YAXXZ"() #2 {
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
define external void @"?doNoinlineExternalUncalled@@YAXXZ"() #4 {
entry:
  ret void
}

; No inlining attribute and called, this should stay.
; CHECK: define {{.*}}doNothingCalled
define void @"?doNothingCalled@@YAXXZ"() #2 {
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
define void @"?doNoinlineCalled@@YAXXZ"() #4 {
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

; Marked external and called, this should become internal and stay.
; CHECK: define {{.*}}doExternalCalled
define external void @"?doExternalCalled@@YAXXZ"() #2 {
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
define external void @"?doNoinlineExternalCalled@@YAXXZ"() #4 {
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

; No inlining attribute, internal, and exported; this should stay.
; CHECK: define {{.*}}doInternalExported
define internal void @"?doInternalExported@@YAXXZ"() #3 {
entry:
  ret void
}

; Alwaysinline, internal, and exported; this should stay.
; CHECK: define {{.*}}doAlwaysInlineInternalExported
define internal void @"?doAlwaysInlineInternalExported@@YAXXZ"() #1 {
entry:
  ret void
}

; Noinline, internal, and exported; this should stay.
; CHECK: define {{.*}}doNoinlineInternalExported
define internal void @"?doNoinlineInternalExported@@YAXXZ"() #5 {
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
attributes #1 = { alwaysinline convergent norecurse nounwind "hlsl.export"}
attributes #2 = { convergent norecurse nounwind }
attributes #3 = { convergent norecurse nounwind "hlsl.export"}
attributes #4 = { convergent noinline norecurse nounwind }
attributes #5 = { convergent noinline norecurse nounwind "hlsl.export"}
attributes #6 = { convergent noinline norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
attributes #7 = { convergent }
