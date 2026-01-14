; RUN: llc < %s | llvm-mc -triple=wasm32-unknown-unknown

target triple = "wasm32-unknown-unknown"

define void @fake_use() {
  %t = call i32 @foo()
  tail call void (...) @llvm.fake.use(i32 %t)
  ret void
}

; %t shouldn't be converted to TEE in RegStackify, because the FAKE_USE will be
; deleted in the beginning of ExplicitLocals.
define void @fake_use_no_tee() {
  %t = call i32 @foo()
  tail call void (...) @llvm.fake.use(i32 %t)
  call void @use(i32 %t)
  ret void
}

declare i32 @foo()
declare void @use(i32 %t)
; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.fake.use(...) #0

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
