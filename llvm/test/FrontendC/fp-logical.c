// RUN: %llvmgcc %s -S -o - | grep bitcast | count 14

typedef float vFloat __attribute__ ((__vector_size__ (16)));
typedef unsigned int vUInt32 __attribute__ ((__vector_size__ (16)));
void foo(vFloat *X) {
  vFloat NoSignBit = (vFloat) ~ (vUInt32) (vFloat) { -0.f, -0.f, -0.f, -0.f };
  vFloat ExtremeValue = *X & NoSignBit;
  *X = ExtremeValue;
}

void bar(vFloat *X) {
  vFloat NoSignBit = (vFloat) ~ (vUInt32) (vFloat) { -0.f, -0.f, -0.f, -0.f };
  vFloat ExtremeValue = *X & ~NoSignBit;
  *X = ExtremeValue;
}
