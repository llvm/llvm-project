; RUN: opt -S -dxil-intrinsic-expansion \
; RUN:   -mtriple=dxil-pc-shadermodel6.0-library %s \
; RUN:   | FileCheck %s --check-prefix=EMULATED \
; RUN:     --implicit-check-not=llvm.dx.quad.all
; RUN: opt -S -dxil-intrinsic-expansion \
; RUN:   -mtriple=dxil-pc-shadermodel6.6-library %s \
; RUN:   | FileCheck %s --check-prefix=EMULATED \
; RUN:     --implicit-check-not=llvm.dx.quad.all
; RUN: opt -S -dxil-intrinsic-expansion \
; RUN:   -mtriple=dxil-pc-shadermodel6.7-library %s \
; RUN:   | FileCheck %s --check-prefix=NATIVE \
; RUN:     --implicit-check-not=llvm.dx.quad.read

; EMULATED-LABEL: define i1 @quad_all(
; EMULATED-NEXT: %[[TOKEN:.*]] = call token @llvm.experimental.convergence.entry()
; EMULATED-NEXT: %[[COND:.*]] = sext i1 %cond to i32
; EMULATED-NEXT: %[[X:.*]] = call i32 @llvm.dx.quad.read.across.x.i32(i32 %[[COND]]) [ "convergencectrl"(token %[[TOKEN]]) ]
; EMULATED-NEXT: %[[ANDX:.*]] = and i32 %[[COND]], %[[X]]
; EMULATED-NEXT: %[[Y:.*]] = call i32 @llvm.dx.quad.read.across.y.i32(i32 %[[COND]]) [ "convergencectrl"(token %[[TOKEN]]) ]
; EMULATED-NEXT: %[[ANDY:.*]] = and i32 %[[ANDX]], %[[Y]]
; EMULATED-NEXT: %[[D:.*]] = call i32 @llvm.dx.quad.read.across.diagonal.i32(i32 %[[COND]]) [ "convergencectrl"(token %[[TOKEN]]) ]
; EMULATED-NEXT: %[[ANDD:.*]] = and i32 %[[ANDY]], %[[D]]
; EMULATED-NEXT: %[[RET:.*]] = trunc i32 %[[ANDD]] to i1
; EMULATED-NEXT: ret i1 %[[RET]]

; NATIVE-LABEL: define i1 @quad_all(
; NATIVE-NEXT: %[[TOKEN:.*]] = call token @llvm.experimental.convergence.entry()
; NATIVE-NEXT: %ret = call i1 @llvm.dx.quad.all(i1 %cond) [ "convergencectrl"(token %[[TOKEN]]) ]
; NATIVE-NEXT: ret i1 %ret
define i1 @quad_all(i1 %cond) convergent {
  %tok = call token @llvm.experimental.convergence.entry()
  %ret = call i1 @llvm.dx.quad.all(i1 %cond) [ "convergencectrl"(token %tok) ]
  ret i1 %ret
}

declare i1 @llvm.dx.quad.all(i1) convergent
