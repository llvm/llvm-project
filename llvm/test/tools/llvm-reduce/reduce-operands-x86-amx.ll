; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operands-zero --test FileCheck --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK,ZERO %s < %t

; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operands-one --test FileCheck --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK,ONE %s < %t

; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operands-poison --test FileCheck --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK,POISON %s < %t

; CHECK-LABEL: @test(
; ZERO: call <1024 x i8> @llvm.x86.cast.tile.to.vector.v1024i8(x86_amx %amx)
; ONE: call <1024 x i8> @llvm.x86.cast.tile.to.vector.v1024i8(x86_amx %amx)
; POISON: call <1024 x i8> @llvm.x86.cast.tile.to.vector.v1024i8(x86_amx %amx)
define <1024 x i8> @test(<1024 x i8> %vec) {
  %amx = call x86_amx @llvm.x86.cast.vector.to.tile.v1024i8(<1024 x i8> %vec)
  %vec2 = call <1024 x i8> @llvm.x86.cast.tile.to.vector.v1024i8(x86_amx %amx)
  ret <1024 x i8> %vec2
}
