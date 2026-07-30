; RUN: llc < %s -mtriple=x86_64-apple-darwin -stop-after=finalize-isel | FileCheck --check-prefixes=CHECK,NOAVX %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin -stop-after=finalize-isel -mattr=avx | FileCheck --check-prefixes=CHECK,AVX %s

@x = common global float zeroinitializer, align 4
@z = common global <4 x float> zeroinitializer, align 16

define void @zero32() nounwind ssp {
  store float zeroinitializer, ptr @x, align 4
  ret void
}

define void @zero128() nounwind ssp {
  store <4 x float> zeroinitializer, ptr @z, align 16
  ret void
}

