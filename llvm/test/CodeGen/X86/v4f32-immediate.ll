; RUN: llc < %s -march=x86 -mattr=+sse | grep movaps

define <4 x float> @foo() {
  ret <4 x float> <float 0x4009C9D0A0000000, float 0x4002666660000000, float 0x3FF3333340000000, float 0x3FB99999A0000000>
}
