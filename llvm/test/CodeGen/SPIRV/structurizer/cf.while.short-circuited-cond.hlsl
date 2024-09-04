// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}

[numthreads(1, 1, 1)]
void main() {
  int a, b;
  while (a && b) {
  }

  while (a || b) {
  }
  while (a && ((a || b) && b)) {
  }

  while (a ? a : b) {
  }

  int x, y;
  while (x + (x && y)) {
  }
}
