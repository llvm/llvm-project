; RUN: llc -mtriple=hexagon -O0 < %s | FileCheck %s

; CHECK: r0 = add(r0,#-1)

define fastcc void @os.linux.tls.initStatic(i32 %x) {
  %1 = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %x, i32 1)
  br label %2

  2:                                                ; preds = %0
  %3 = extractvalue { i32, i1 } %1, 0
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare { i32, i1 } @llvm.usub.with.overflow.i32(i32, i32) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

