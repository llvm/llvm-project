// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}

[numthreads(1, 1, 1)]
void main() {
  int val=0, i=0, j=0, k=0;

  while (i < 10) {
    val = val + i;
    while (j < 20) {
      while (k < 30) {
        val = val + k;
        ++k;
      }

      val = val * 2;
      ++j;
    }

    ++i;
  }
}
