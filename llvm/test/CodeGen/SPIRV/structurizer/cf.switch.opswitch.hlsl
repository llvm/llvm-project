// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}

int foo() { return 200; }

[numthreads(1, 1, 1)]
void main() {
  int result;

  ////////////////////////////
  // The most basic case    //
  // Has a 'default' case   //
  // All cases have 'break' //
  ////////////////////////////
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

  ////////////////////////////////////
  // The selector is a statement    //
  // Does not have a 'default' case //
  // All cases have 'break'         //
  ////////////////////////////////////

  switch(int c = a) {
    case -4:
      result = -400;
      break;
    case 4:
      result = 400;
      break;
  }

  ///////////////////////////////////
  // All cases are fall-through    //
  // The last case is fall-through //
  ///////////////////////////////////
  switch(a) {
    case -5:
      result = -500;
    case 5:
      result = 500;
  }

  ///////////////////////////////////////
  // Some cases are fall-through       //
  // The last case is not fall-through //
  ///////////////////////////////////////

  switch(a) {
    case 6:
      result = 600;
    case 7:
      result = 700;
    case 8:
      result = 800;
      break;
    default:
      result = 777;
      break;
  }

  ///////////////////////////////////////
  // Fall-through cases with no body   //
  ///////////////////////////////////////

  switch(a) {
    case 10:
    case 11:
    default:
    case 12:
      result = 12;
  }

  ////////////////////////////////////////////////
  // No-op. Two nested cases and a nested break //
  ////////////////////////////////////////////////

  switch(a) {
    case 15:
    case 16:
      break;
  }

  ////////////////////////////////////////////////////////////////
  // Using braces (compound statements) in various parts        //
  // Using breaks such that each AST configuration is different //
  // Also uses 'forcecase' attribute                            //
  ////////////////////////////////////////////////////////////////

  switch(a) {
    case 20: {
      result = 20;
      break;
    }
    case 21:
      result = 21;
      break;
    case 22:
    case 23:
      break;
    case 24:
    case 25: { result = 25; }
      break;
    case 26:
    case 27: {
      break;
    }
    case 28: {
      result = 28;
      {{break;}}
    }
    case 29: {
      {
        result = 29;
        {break;}
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////
  // Nested Switch statements with mixed use of fall-through and braces //
  ////////////////////////////////////////////////////////////////////////

  switch(a) {
    case 30: {
        result = 30;
        switch(result) {
          default:
            a = 55;
          case 50:
            a = 50;
            break;
          case 51:
          case 52:
            a = 52;
          case 53:
            a = 53;
            break;
          case 54 : {
            a = 54;
            break;
          }
        }
    }
  }

  ///////////////////////////////////////////////
  // Constant integer variables as case values //
  ///////////////////////////////////////////////

  const int r = 35;
  const int s = 45;
  const int t = 2*r + s;  // evaluates to 115.

  switch(a) {
    case r:
      result = r;
    case t:
      result = t;
      break;
  }


  //////////////////////////////////////////////////////////////////
  // Using float as selector results in multiple casts in the AST //
  //////////////////////////////////////////////////////////////////
  float sel;
  switch ((int)sel) {
  case 0:
    result = 0;
    break;
  }
}
