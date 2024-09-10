// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}

int foo() { return true; }

[numthreads(1, 1, 1)]
void main() {
  int val = 0;
  int i = 0;

  while (i < 10) {
    val = i;
    if (val > 5) {
      continue;
    }

    if (val > 6) {
      {{continue;}}
      val++;       // No SPIR-V should be emitted for this statement.
      continue;    // No SPIR-V should be emitted for this statement.
      while(true); // No SPIR-V should be emitted for this statement.
      --i;         // No SPIR-V should be emitted for this statement.
    }

  }

  //////////////////////////////////////////////////////////////////////////////////////
  // Nested while loops with continue statements                                      //
  // Each continue statement should branch to the corresponding loop's continue block //
  //////////////////////////////////////////////////////////////////////////////////////

  while (true) {
    i++;

    while(i<20) {
      val = i;
      continue;
    }
    --i;
    continue;
    continue;  // No SPIR-V should be emitted for this statement.

  }
}
