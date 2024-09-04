// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}
// RUN: %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-sim --function=_Z7processv --wave=1 --expects=1

int fn() { return true; }

int process() {
  // Use in control flow
  int a = 0;
  int b = 0;
  int val = 0;
  if (a && b) val++;

  // Operand with side effects
  if (fn() && fn()) val++;

  if (a && fn())
    val++;

  if (fn() && b)
    val++;
  return val;
}

[numthreads(1, 1, 1)]
void main() {
  process();
}
