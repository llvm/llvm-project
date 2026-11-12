; RUN: llc < %s -stop-after=finalize-isel -mtriple=x86_64-unknown-linux - | FileCheck %s --implicit-check-not FAKE_USE
; Fake uses following tail calls should be pulled in front
; of the TCRETURN instruction. Fake uses using something defined by
; the tail call or after it should be suppressed.

; CHECK: name:{{ +}}bar
; CHECK: body:
; CHECK: bb.0.{{.*}}:
; CHECK: %0:{{.*}}= COPY
; CHECK: FAKE_USE %0
; CHECK: TCRETURN

; CHECK: name:{{ +}}baz
; CHECK: body:
; CHECK: bb.0.{{.*}}:
; CHECK: %0:{{.*}}= COPY
; CHECK: FAKE_USE %0
; CHECK: TCRETURN

define void @bar(i32 %v) optdebug {
entry:
  %call = tail call i32 @_Z3fooi(i32 %v)
  %mul = mul nsw i32 %call, 3
  notail call void (...) @llvm.fake.use(i32 %mul)
  notail call void (...) @llvm.fake.use(i32 %call)
  notail call void (...) @llvm.fake.use(i32 %v)
  ret void
}

define i32 @baz(i32 %v) optdebug {
entry:
  %call = tail call i32 @_Z3fooi(i32 %v)
  notail call void (...) @llvm.fake.use(i32 %v)
  ret i32 %call
}

declare i32 @_Z3fooi(i32) local_unnamed_addr
