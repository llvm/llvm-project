// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}
// RUN: %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-sim --function=_Z7processv --wave=1 --expects=19

int process() {
  int val = 0;

  for (int i = 0; i < 10; ++i) {
    if (i < 5) {
      continue;
    }
    val = i;

    {
      continue;
    }
    val++;       // No SPIR-V should be emitted for this statement.
    continue;    // No SPIR-V should be emitted for this statement.
    while(true); // No SPIR-V should be emitted for this statement.
  }

  //////////////////////////////////////////////////////////////////////////////////////
  // Nested for loops with continue statements                                        //
  // Each continue statement should branch to the corresponding loop's continue block //
  //////////////////////////////////////////////////////////////////////////////////////

  for (int j = 0; j < 10; ++j) {
    val = j+5;

    for ( ; val < 20; ++val) {
      int k = val + j;
      continue;
      k++;      // No SPIR-V should be emitted for this statement.
    }

    val -= 1;
    continue;
    continue;     // No SPIR-V should be emitted for this statement.
    val = val*10; // No SPIR-V should be emitted for this statement.
  }

  return val;
}

[numthreads(1, 1, 1)]
void main() {
  process();
}
