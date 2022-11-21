; RUN: opt < %s -S -rewrite-statepoints-for-gc | FileCheck %s

; CHECK: Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
; CHECK: declare i64 @llvm.x86.sse2.cvttsd2si64(<2 x double>)
declare i64 @llvm.x86.sse2.cvttsd2si64(<2 x double>)

define i64 @test(<2 x double> %arg) {
  %ret = call i64 @llvm.x86.sse2.cvttsd2si64(<2 x double> %arg)
  ret i64 %ret
}

