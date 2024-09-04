// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}
// RUN: %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-sim --function=_Z7processv --wave=1 --expects=2563170

int process() {
  int val = 0;

  for (int i = 0; i < 10; ++i) {
    val = val + i;

    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        val = val + k;
      }

      val = val * 2;

    }
  }
  return val;
}

[numthreads(1, 1, 1)]
void main() {
  process();
}
