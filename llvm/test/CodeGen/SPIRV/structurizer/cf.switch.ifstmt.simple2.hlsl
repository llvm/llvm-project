// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}
// RUN: %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-sim --function=_Z7processv --wave=1 --expects=5

int foo() { return 200; }

int process() {
  int a = 0;
  int b = 0;
  int c = 0;
  const int r = 20;
  const int s = 40;
  const int t = 3*r+2*s;

  switch(int d = 5) {
    case 1:
      b += 1;
      c += foo();
    case 2:
      b += 2;
      break;
    case 3:
    {
      b += 3;
      break;
    }
    case t:
      b += t;
    case 4:
    case 5:
      b += 5;
      break;
    case 6: {
    case 7:
      break;}
    default:
      break;
  }

  return a + b + c;
}

[numthreads(1, 1, 1)]
void main() {
  process();
}
