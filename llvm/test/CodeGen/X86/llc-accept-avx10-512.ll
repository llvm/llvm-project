; avx10.x-512 is just avx10.x -- 512 is kept for compatibility purposes.

; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=avx10.1-512 2>&1 | grep -v "is not a recognized feature"

; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=avx10.2-512 2>&1 | grep -v "is not a recognized feature"

define float @foo(float %f) {
  ret float %f
}

