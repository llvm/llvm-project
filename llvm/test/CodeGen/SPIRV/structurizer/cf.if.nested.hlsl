// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}
// RUN: %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-sim --function=_Z7processv --wave=1 --expects=3

int process() {
  int c1 = 0;
  int c2 = 1;
  int c3 = 0;
  int c4 = 1;
  int val = 0;

  if (c1) {
    if (c2)
      val = 1;
  } else {
    if (c3) {
      val = 2;
    } else {
      if (c4) {
        val = 3;
      }
    }
  }
  return val;
}

[numthreads(1, 1, 1)]
void main() {
  process();
}
