// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}

int foo() { return true; }

[numthreads(1, 1, 1)]
void main() {
  int val = 0;
  int i = 0;

  //////////////////////////
  //// Basic while loop ////
  //////////////////////////
  while (i < 10) {
      val = i;
  }

  //////////////////////////
  ////  infinite loop   ////
  //////////////////////////
  while (true) {
      val = 0;
  }

  //////////////////////////
  ////    Null Body     ////
  //////////////////////////
  while (val < 20)
    ;

  ////////////////////////////////////////////////////////////////
  //// Condition variable has VarDecl                         ////
  //// foo() returns an integer which must be cast to boolean ////
  ////////////////////////////////////////////////////////////////
  while (int a = foo()) {
    val = a;
  }

}
