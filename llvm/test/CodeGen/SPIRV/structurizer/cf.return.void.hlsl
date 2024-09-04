// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}

void A() {
}

[numthreads(1, 1, 1)]
void main() {
  // CHECK: [[type:%[0-9]+]] = OpTypeFunction %void
  // CHECK:        %src_main = OpFunction %void None [[type]]
  // CHECK:      {{%[0-9]+}} = OpFunctionCall %void %A
  // CHECK:                    OpReturn
  // CHECK:                    OpFunctionEnd
  return A();
}
