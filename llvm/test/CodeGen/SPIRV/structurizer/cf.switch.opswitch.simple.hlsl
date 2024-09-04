// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}

int foo() { return 200; }

[numthreads(1, 1, 1)]
void main() {
  int result;

  int a = 0;
  switch(a) {
    case -3:
      result = -300;
      break;
    case 0:
      result = 0;
      break;
    case 1:
      result = 100;
      break;
    case 2:
      result = foo();
      break;
    default:
      result = 777;
      break;
  }

  switch(int c = a) {
    case -4:
      result = -400;
      break;
    case 4:
      result = 400;
      break;
  }
}
