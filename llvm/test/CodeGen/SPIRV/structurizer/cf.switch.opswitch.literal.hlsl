// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}

[numthreads(1, 1, 1)]
void main() {
///////////////////////////////
// 32-bit int literal switch //
///////////////////////////////
  switch (0) {
  case 0:
    return;
  default:
    return;
  }

///////////////////////////////
// 64-bit int literal switch //
///////////////////////////////
  switch (12345678910) {
  case 12345678910:
    return;
  }

  return;
}
