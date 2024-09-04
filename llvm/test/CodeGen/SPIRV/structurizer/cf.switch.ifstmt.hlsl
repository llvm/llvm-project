// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}
// RUN: %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-sim --function=_Z7processv --wave=1 --expects=308

int foo() { return 200; }

int process() {
  int a = 0;
  int b = 0;
  int c = 0;
  const int r = 20;
  const int s = 40;
  const int t = 3*r+2*s;


  ////////////////////////////////////////
  // DefaultStmt is the first statement //
  ////////////////////////////////////////
  switch(a) {
    default:
      b += 0;
    case 1:
      b += 1;
      break;
    case 2:
      b += 2;
  }


  //////////////////////////////////////////////
  // DefaultStmt in the middle of other cases //
  //////////////////////////////////////////////
  switch(a) {
    case 10:
      b += 1;
    default:
      b += 0;
    case 20:
      b += 2;
      break;
  }

  ///////////////////////////////////////////////
  // Various CaseStmt and BreakStmt topologies //
  // DefaultStmt is the last statement         //
  ///////////////////////////////////////////////
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


  //////////////////////////
  // No Default statement //
  //////////////////////////
  switch(a) {
    case 100:
      b += 100;
      break;
  }


  /////////////////////////////////////////////////////////
  // No cases. Only a default                            //
  // This means the default body will always be executed //
  /////////////////////////////////////////////////////////
  switch(a) {
    default:
      b += 100;
      c += 200;
      break;
  }


  ////////////////////////////////////////////////////////////
  // Nested Switch with branching                           //
  // The two inner switch statements should be executed for //
  // both cases of the outer switch (case 300 and case 400) //
  ////////////////////////////////////////////////////////////
  switch(a) {
    case 300:
      b += 300;
    case 400:
      switch(c) {
        case 500:
          b += 500;
          break;
        case 600:
          switch(b) {
            default:
            a += 600;
            b += 600;
          }
      }
  }

  return a + b + c;
}

[numthreads(1, 1, 1)]
void main() {
  process();
}
