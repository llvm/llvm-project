// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}
// RUN: %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-sim --function=_Z7processv --wave=1 --expects=2

int process() {
  int c = 0;
  int val = 0;

  // Both then and else
  if (c) {
    val = val + 1;
  } else {
    val = val + 2;
  }

  // No else
  if (c)
    val = 1;

  // Empty then
  if (c) {
  } else {
    val = 2;
  }

  // Null body
  if (c)
    ;

  if (int d = val) {
    c = true;
  }

  return val;
}

[numthreads(1, 1, 1)]
void main() {
  process();
}
