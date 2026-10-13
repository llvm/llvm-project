; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=instructions --test FileCheck --test-arg --check-prefixes=CHECK,INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefixes=CHECK,RESULT %s < %t

; CHECK-LABEL: define <1024 x i8> @test(
; INTERESTING: call <1024 x i8> @llvm.x86.cast.tile.to.vector

; RESULT: %amx = call x86_amx @llvm.x86.cast.vector.to.tile.v1024i8(<1024 x i8> %vec)
; RESULT: %vec2 = call <1024 x i8> @llvm.x86.cast.tile.to.vector.v1024i8(x86_amx %amx)
define <1024 x i8> @test(<1024 x i8> %vec) {
  %amx = call x86_amx @llvm.x86.cast.vector.to.tile.v1024i8(<1024 x i8> %vec)
  %vec2 = call <1024 x i8> @llvm.x86.cast.tile.to.vector.v1024i8(x86_amx %amx)
  ret <1024 x i8> %vec2
}
