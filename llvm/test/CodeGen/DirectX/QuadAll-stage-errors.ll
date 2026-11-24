; RUN: split-file %s %t
; RUN: not opt -S -dxil-intrinsic-expansion \
; RUN:   -mtriple=dxil-pc-shadermodel6.0-vertex %t/direct.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=VERTEX
; RUN: not opt -S -dxil-intrinsic-expansion \
; RUN:   -mtriple=dxil-pc-shadermodel6.7-vertex %t/direct.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=VERTEX
; RUN: not opt -S -dxil-intrinsic-expansion \
; RUN:   -mtriple=dxil-pc-shadermodel6.6-library %t/library.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=LIBRARY
; RUN: not opt -S -dxil-intrinsic-expansion \
; RUN:   -mtriple=dxil-pc-shadermodel6.7-library %t/library.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=LIBRARY
; RUN: opt -S -dxil-intrinsic-expansion \
; RUN:   -mtriple=dxil-pc-shadermodel6.6-library %t/allowed.ll \
; RUN:   | FileCheck %s --check-prefix=EMULATED \
; RUN:     --implicit-check-not=llvm.dx.quad.all
; RUN: opt -S -dxil-intrinsic-expansion \
; RUN:   -mtriple=dxil-pc-shadermodel6.7-library %t/allowed.ll \
; RUN:   | FileCheck %s --check-prefix=NATIVE
; RUN: opt -S -dxil-intrinsic-expansion \
; RUN:   -mtriple=dxil-pc-shadermodel6.6-vertex %t/unused.ll \
; RUN:   | FileCheck %s --check-prefix=UNUSED

;--- direct.ll
; VERTEX: error: intrinsic 'QuadAll' is not supported in 'vertex' shaders for DirectX; supported stages: 'compute', 'amplification', 'mesh', 'pixel', 'node' (function 'vote')
define i1 @vote(i1 %cond) {
  %ret = call i1 @llvm.dx.quad.all(i1 %cond)
  ret i1 %ret
}

declare i1 @llvm.dx.quad.all(i1)

;--- unused.ll
define i1 @unused(i1 %cond) {
  %ret = call i1 @llvm.dx.quad.all(i1 %cond)
  ret i1 %ret
}

; UNUSED: define void @vertex()
define void @vertex() #0 {
  ret void
}

attributes #0 = { "hlsl.shader"="vertex" }

declare i1 @llvm.dx.quad.all(i1)

;--- library.ll
define i1 @shared(i1 %cond) {
  %ret = call i1 @llvm.dx.quad.all(i1 %cond)
  ret i1 %ret
}

define i1 @compute(i1 %cond) #0 {
  %ret = call i1 @shared(i1 %cond)
  ret i1 %ret
}

; LIBRARY: error: intrinsic 'QuadAll' is not supported in 'vertex' shaders for DirectX; supported stages: 'compute', 'amplification', 'mesh', 'pixel', 'node' (function 'vertex')
define i1 @vertex(i1 %cond) #1 {
  %ret = call i1 @shared(i1 %cond)
  ret i1 %ret
}

attributes #0 = { "hlsl.shader"="compute" }
attributes #1 = { "hlsl.shader"="vertex" }

declare i1 @llvm.dx.quad.all(i1)

;--- allowed.ll
define i1 @exported(i1 %cond) {
  ; EMULATED: call i32 @llvm.dx.quad.read.across.x.i32
  ; EMULATED: call i32 @llvm.dx.quad.read.across.y.i32
  ; EMULATED: call i32 @llvm.dx.quad.read.across.diagonal.i32
  ; NATIVE: call i1 @llvm.dx.quad.all
  %ret = call i1 @llvm.dx.quad.all(i1 %cond)
  ret i1 %ret
}

define i1 @pixel(i1 %cond) #0 {
  %ret = call i1 @exported(i1 %cond)
  ret i1 %ret
}

define i1 @compute(i1 %cond) #1 {
  %ret = call i1 @exported(i1 %cond)
  ret i1 %ret
}

define i1 @amplification(i1 %cond) #2 {
  %ret = call i1 @exported(i1 %cond)
  ret i1 %ret
}

define i1 @mesh(i1 %cond) #3 {
  %ret = call i1 @exported(i1 %cond)
  ret i1 %ret
}

define i1 @node(i1 %cond) #4 {
  %ret = call i1 @exported(i1 %cond)
  ret i1 %ret
}

attributes #0 = { "hlsl.shader"="pixel" }
attributes #1 = { "hlsl.shader"="compute" }
attributes #2 = { "hlsl.shader"="amplification" }
attributes #3 = { "hlsl.shader"="mesh" }
attributes #4 = { "hlsl.shader"="node" }

declare i1 @llvm.dx.quad.all(i1)
