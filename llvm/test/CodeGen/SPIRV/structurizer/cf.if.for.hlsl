// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}
// RUN: %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-sim --function=_Z7processv --wave=1 --expects=6

int process() {
  int color = 0;

  int val = 0;

  if (color < 0) {
    val = 1;
  }

  // for-stmt following if-stmt
  for (int i = 0; i < 10; ++i) {
    if (color < 0) { // if-stmt nested in for-stmt
      val = val + 1;
      for (int j = 0; j < 15; ++j) { // for-stmt deeply nested in if-then
        val = val * 2;
      } // end for (int j
      val = val + 3;
    }

    if (color < 1) { // if-stmt following if-stmt
      val = val * 4;
    } else {
      for (int k = 0; k < 20; ++k) { // for-stmt deeply nested in if-else
        val = val - 5;
        if (val < 0) { // deeply nested if-stmt
          val = val + 100;
        }
      } // end for (int k
    } // end elsek
  } // end for (int i

  // if-stmt following for-stmt
  if (color < 2) {
    val = val + 6;
  }

  return val;
}

[numthreads(1, 1, 1)]
void main() {
  process();
}
