; RUN: llc < %s | llvm-mc -triple=wasm32-unknown-unknown

target triple = "wasm32-unknown-unknown"

define void @fake_use_test() {
  %t = call i32 @foo()
  tail call void (...) @llvm.fake.use(i32 %t)
  ret void
}

declare void @foo()
; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.fake.use(...) #0

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
