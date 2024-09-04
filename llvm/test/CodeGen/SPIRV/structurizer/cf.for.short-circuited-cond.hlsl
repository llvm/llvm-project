// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}
// RUN: %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-sim --function=_Z7processv --wave=1 --expects=9

int process() {
  int a = 0;
  int b = 1;
  int val = 0;

  for (int i = 0; a && b; ++i) {
    val += 1;
  }

  for (int i = 0; a || b; ++i) {
    val += 1;
    b = 0;
  }

  b = 1;
  for (int i = 0; a && ((a || b) && b); ++i) {
    val += 4;
    b = 0;
  }

  b = 1;
  for (int i = 0; a ? a : b; ++i) {
    val += 8;
    b = 0;
  }

  int x = 0;
  int y = 0;
  for (int i = 0; x + (x && y); ++i) {
    val += 16;
  }

  return val;
}

[numthreads(1, 1, 1)]
void main() {
  process();
}
