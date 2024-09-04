// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}
// RUN: %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-sim --function=_Z7processv --wave=1 --expects=5

int process() {
  int b = 0;
  const int t = 50;

  switch(int d = 5) {
    case t:
      b = t;
    case 4:
    case 5:
      b = 5;
      break;
    default:
      break;
  }

  return b;
}

[numthreads(1, 1, 1)]
void main() {
  process();
}
