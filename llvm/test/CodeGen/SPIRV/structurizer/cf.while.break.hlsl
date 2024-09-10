// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}

int foo() { return true; }

[numthreads(1, 1, 1)]
void main() {
  int val = 0;
  int i = 0;

  while (i < 10) {
    val = i;
    if (val > 5) {
      break;
    }

    if (val > 6) {
      break;
      break;       // No SPIR-V should be emitted for this statement.
      val++;       // No SPIR-V should be emitted for this statement.
      while(true); // No SPIR-V should be emitted for this statement.
      --i;         // No SPIR-V should be emitted for this statement.
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Nested while loops with break statements                                   //
  // Each break statement should branch to the corresponding loop's break block //
  ////////////////////////////////////////////////////////////////////////////////

  while (true) {
    i++;
    while(i<20) {
      val = i;
      {{break;}}
    }
    --i;
    break;
  }
}
