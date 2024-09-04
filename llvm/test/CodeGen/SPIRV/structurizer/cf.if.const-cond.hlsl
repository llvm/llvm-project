// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}
// RUN: %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-sim --function=_Z7processv --wave=1 --expects=3

int process() {
  int a = 0;
  int b = 0;

  if (3 + 5) {
    a = 1;
  } else {
    a = 0;
  }

  if (4 + 3 > 7 || 4 + 3 < 8) {
    b = 2;
  }

  if (4 + 3 > 7 && true) {
    b = 0;
  }

  if (true)
    ;

  if (false) {}

  return a + b;
}

[numthreads(1, 1, 1)]
void main() {
  process();
}
