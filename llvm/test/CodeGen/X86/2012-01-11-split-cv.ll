; RUN: llc < %s -march=x86 -mcpu=corei7-avx -mattr=+avx -mtriple=i686-pc-win32 | FileCheck %s

;CHECK: add18i16
define void @add18i16(<18 x i16>* nocapture sret %ret, <18 x i16>* %bp) nounwind {
;CHECK: vmovups
  %b = load <18 x i16>* %bp, align 16
  %x = add <18 x i16> zeroinitializer, %b
  store <18 x i16> %x, <18 x i16>* %ret, align 16
;CHECK: ret
  ret void
}

