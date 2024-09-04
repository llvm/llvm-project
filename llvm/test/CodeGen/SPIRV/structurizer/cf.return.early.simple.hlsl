// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}

[numthreads(1, 1, 1)]
void main() {
  int a, b;
  int cond = 1;

  while(cond) {
    switch(b) {
      default:
        if (cond) {
          if (cond)
            return;
          else
            return;
        }
    }
    return;
  }
}
