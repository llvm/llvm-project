; RUN: opt < %s -passes=normalize -verify-each

define fastcc void @foo.resume_musttail(ptr %FramePtr) {
entry:
  %0 = tail call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  musttail call fastcc void undef(ptr null)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
