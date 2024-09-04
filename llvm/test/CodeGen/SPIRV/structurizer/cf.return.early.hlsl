// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}
// RUN: %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-sim --function=_Z7processv --wave=1 --expects=0

int process() {
  int cond = 1;
  int value = 0;

  while(value < 10) {
    switch(value) {
      case 1:
        value = 1;
        return value;
      case 2: {
        value = 3;
        {return value;}   // Return from function.
        value = 4;      // No SPIR-V should be emitted for this statement.
        break;      // No SPIR-V should be emitted for this statement.
      }
      case 5 : {
        value = 5;
        {{return value;}} // Return from function.
        value = 6;      // No SPIR-V should be emitted for this statement.
      }
      default:
        for (int i=0; i<10; ++i) {
          if (cond) {
            return value;    // Return from function.
            return value;    // No SPIR-V should be emitted for this statement.
            continue;  // No SPIR-V should be emitted for this statement.
            break;     // No SPIR-V should be emitted for this statement.
            ++value;       // No SPIR-V should be emitted for this statement.
          } else {
            return value;   // Return from function
            continue; // No SPIR-V should be emitted for this statement.
            break;    // No SPIR-V should be emitted for this statement.
            ++value;      // No SPIR-V should be emitted for this statement.
          }
        }

        // Return from function.
        // Even though this statement will never be executed [because both "if" and "else" above have return statements],
        // SPIR-V code should be emitted for it as we do not analyze the logic.
        return value;
    }

    // Return from function.
    // Even though this statement will never be executed [because all "case" statements above contain a return statement],
    // SPIR-V code should be emitted for it as we do not analyze the logic.
    return value;
  }

  return value;
}

[numthreads(1, 1, 1)]
void main() {
  process();
}
