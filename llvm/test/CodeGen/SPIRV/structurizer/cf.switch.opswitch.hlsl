// RUN: %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val

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
// CHECK: [[a:%[0-9]+]] = OpLoad %int %a
// CHECK-NEXT: OpSelectionMerge %switch_merge None
// CHECK-NEXT: OpSwitch [[a]] %switch_default -3 %switch_n3 0 %switch_0 1 %switch_1 2 %switch_2
  switch(a) {
// CHECK-NEXT: %switch_n3 = OpLabel
// CHECK-NEXT: OpStore %result %int_n300
// CHECK-NEXT: OpBranch %switch_merge
    case -3:
      result = -300;
      break;
// CHECK-NEXT: %switch_0 = OpLabel
// CHECK-NEXT: OpStore %result %int_0
// CHECK-NEXT: OpBranch %switch_merge
    case 0:
      result = 0;
      break;
// CHECK-NEXT: %switch_1 = OpLabel
// CHECK-NEXT: OpStore %result %int_100
// CHECK-NEXT: OpBranch %switch_merge
    case 1:
      result = 100;
      break;
// CHECK-NEXT: %switch_2 = OpLabel
// CHECK-NEXT: [[foo:%[0-9]+]] = OpFunctionCall %int %foo
// CHECK-NEXT: OpStore %result [[foo]]
// CHECK-NEXT: OpBranch %switch_merge
    case 2:
      result = foo();
      break;
// CHECK-NEXT: %switch_default = OpLabel
// CHECK-NEXT: OpStore %result %int_777
// CHECK-NEXT: OpBranch %switch_merge
    default:
      result = 777;
      break;
  }
// CHECK-NEXT: %switch_merge = OpLabel



  ////////////////////////////////////
  // The selector is a statement    //
  // Does not have a 'default' case //
  // All cases have 'break'         //
  ////////////////////////////////////

// CHECK-NEXT: [[a1:%[0-9]+]] = OpLoad %int %a
// CHECK-NEXT: OpStore %c [[a1]]
// CHECK-NEXT: [[c:%[0-9]+]] = OpLoad %int %c
// CHECK-NEXT: OpSelectionMerge %switch_merge_0 None
// CHECK-NEXT: OpSwitch [[c]] %switch_merge_0 -4 %switch_n4 4 %switch_4
  switch(int c = a) {
// CHECK-NEXT: %switch_n4 = OpLabel
// CHECK-NEXT: OpStore %result %int_n400
// CHECK-NEXT: OpBranch %switch_merge_0
    case -4:
      result = -400;
      break;
// CHECK-NEXT: %switch_4 = OpLabel
// CHECK-NEXT: OpStore %result %int_400
// CHECK-NEXT: OpBranch %switch_merge_0
    case 4:
      result = 400;
      break;
  }
// CHECK-NEXT: %switch_merge_0 = OpLabel



  ///////////////////////////////////
  // All cases are fall-through    //
  // The last case is fall-through //
  ///////////////////////////////////

// CHECK-NEXT: [[a2:%[0-9]+]] = OpLoad %int %a
// CHECK-NEXT: OpSelectionMerge %switch_merge_1 None
// CHECK-NEXT: OpSwitch [[a2]] %switch_merge_1 -5 %switch_n5 5 %switch_5
  switch(a) {
// CHECK-NEXT: %switch_n5 = OpLabel
// CHECK-NEXT: OpStore %result %int_n500
// CHECK-NEXT: OpBranch %switch_5
    case -5:
      result = -500;
// CHECK-NEXT: %switch_5 = OpLabel
// CHECK-NEXT: OpStore %result %int_500
// CHECK-NEXT: OpBranch %switch_merge_1
    case 5:
      result = 500;
  }
// CHECK-NEXT: %switch_merge_1 = OpLabel



  ///////////////////////////////////////
  // Some cases are fall-through       //
  // The last case is not fall-through //
  ///////////////////////////////////////

// CHECK-NEXT: [[a3:%[0-9]+]] = OpLoad %int %a
// CHECK-NEXT: OpSelectionMerge %switch_merge_2 None
// CHECK-NEXT: OpSwitch [[a3]] %switch_default_0 6 %switch_6 7 %switch_7 8 %switch_8
  switch(a) {
// CHECK-NEXT: %switch_6 = OpLabel
// CHECK-NEXT: OpStore %result %int_600
// CHECK-NEXT: OpBranch %switch_7
    case 6:
      result = 600;
    case 7:
// CHECK-NEXT: %switch_7 = OpLabel
// CHECK-NEXT: OpStore %result %int_700
// CHECK-NEXT: OpBranch %switch_8
      result = 700;
// CHECK-NEXT: %switch_8 = OpLabel
// CHECK-NEXT: OpStore %result %int_800
// CHECK-NEXT: OpBranch %switch_merge_2
    case 8:
      result = 800;
      break;
// CHECK-NEXT: %switch_default_0 = OpLabel
// CHECK-NEXT: OpStore %result %int_777
// CHECK-NEXT: OpBranch %switch_merge_2
    default:
      result = 777;
      break;
  }
// CHECK-NEXT: %switch_merge_2 = OpLabel



  ///////////////////////////////////////
  // Fall-through cases with no body   //
  ///////////////////////////////////////

// CHECK-NEXT: [[a4:%[0-9]+]] = OpLoad %int %a
// CHECK-NEXT: OpSelectionMerge %switch_merge_3 None
// CHECK-NEXT: OpSwitch [[a4]] %switch_default_1 10 %switch_10 11 %switch_11 12 %switch_12
  switch(a) {
// CHECK-NEXT: %switch_10 = OpLabel
// CHECK-NEXT: OpBranch %switch_11
    case 10:
// CHECK-NEXT: %switch_11 = OpLabel
// CHECK-NEXT: OpBranch %switch_default_1
    case 11:
// CHECK-NEXT: %switch_default_1 = OpLabel
// CHECK-NEXT: OpBranch %switch_12
    default:
// CHECK-NEXT: %switch_12 = OpLabel
// CHECK-NEXT: OpStore %result %int_12
// CHECK-NEXT: OpBranch %switch_merge_3
    case 12:
      result = 12;
  }
// CHECK-NEXT: %switch_merge_3 = OpLabel



  ////////////////////////////////////////////////
  // No-op. Two nested cases and a nested break //
  ////////////////////////////////////////////////

// CHECK-NEXT: [[a5:%[0-9]+]] = OpLoad %int %a
// CHECK-NEXT: OpSelectionMerge %switch_merge_4 None
// CHECK-NEXT: OpSwitch [[a5]] %switch_merge_4 15 %switch_15 16 %switch_16
  switch(a) {
// CHECK-NEXT: %switch_15 = OpLabel
// CHECK-NEXT: OpBranch %switch_16
    case 15:
// CHECK-NEXT: %switch_16 = OpLabel
// CHECK-NEXT: OpBranch %switch_merge_4
    case 16:
      break;
  }
// CHECK-NEXT: %switch_merge_4 = OpLabel



  ////////////////////////////////////////////////////////////////
  // Using braces (compound statements) in various parts        //
  // Using breaks such that each AST configuration is different //
  // Also uses 'forcecase' attribute                            //
  ////////////////////////////////////////////////////////////////

// CHECK-NEXT: [[a6:%[0-9]+]] = OpLoad %int %a
// CHECK-NEXT: OpSelectionMerge %switch_merge_5 None
// CHECK-NEXT: OpSwitch [[a6]] %switch_merge_5 20 %switch_20 21 %switch_21 22 %switch_22 23 %switch_23 24 %switch_24 25 %switch_25 26 %switch_26 27 %switch_27 28 %switch_28 29 %switch_29
  switch(a) {
// CHECK-NEXT: %switch_20 = OpLabel
// CHECK-NEXT: OpStore %result %int_20
// CHECK-NEXT: OpBranch %switch_merge_5
    case 20: {
      result = 20;
      break;
    }
// CHECK-NEXT: %switch_21 = OpLabel
// CHECK-NEXT: OpStore %result %int_21
// CHECK-NEXT: OpBranch %switch_merge_5
    case 21:
      result = 21;
      break;
// CHECK-NEXT: %switch_22 = OpLabel
// CHECK-NEXT: OpBranch %switch_23
// CHECK-NEXT: %switch_23 = OpLabel
// CHECK-NEXT: OpBranch %switch_merge_5
    case 22:
    case 23:
      break;
// CHECK-NEXT: %switch_24 = OpLabel
// CHECK-NEXT: OpBranch %switch_25
// CHECK-NEXT: %switch_25 = OpLabel
// CHECK-NEXT: OpStore %result %int_25
// CHECK-NEXT: OpBranch %switch_merge_5
    case 24:
    case 25: { result = 25; }
      break;
// CHECK-NEXT: %switch_26 = OpLabel
// CHECK-NEXT: OpBranch %switch_27
// CHECK-NEXT: %switch_27 = OpLabel
// CHECK-NEXT: OpBranch %switch_merge_5
    case 26:
    case 27: {
      break;
    }
// CHECK-NEXT: %switch_28 = OpLabel
// CHECK-NEXT: OpStore %result %int_28
// CHECK-NEXT: OpBranch %switch_merge_5
    case 28: {
      result = 28;
      {{break;}}
    }
// CHECK-NEXT: %switch_29 = OpLabel
// CHECK-NEXT: OpStore %result %int_29
// CHECK-NEXT: OpBranch %switch_merge_5
    case 29: {
      {
        result = 29;
        {break;}
      }
    }
  }
// CHECK-NEXT: %switch_merge_5 = OpLabel



  ////////////////////////////////////////////////////////////////////////
  // Nested Switch statements with mixed use of fall-through and braces //
  ////////////////////////////////////////////////////////////////////////

// CHECK-NEXT: [[a7:%[0-9]+]] = OpLoad %int %a
// CHECK-NEXT: OpSelectionMerge %switch_merge_7 None
// CHECK-NEXT: OpSwitch [[a7]] %switch_merge_7 30 %switch_30
  switch(a) {
// CHECK-NEXT: %switch_30 = OpLabel
    case 30: {
// CHECK-NEXT: OpStore %result %int_30
        result = 30;
// CHECK-NEXT: [[result:%[0-9]+]] = OpLoad %int %result
// CHECK-NEXT: OpSelectionMerge %switch_merge_6 None
// CHECK-NEXT: OpSwitch [[result]] %switch_default_2 50 %switch_50 51 %switch_51 52 %switch_52 53 %switch_53 54 %switch_54
        switch(result) {
// CHECK-NEXT: %switch_default_2 = OpLabel
// CHECK-NEXT: OpStore %a %int_55
// CHECK-NEXT: OpBranch %switch_50
          default:
            a = 55;
// CHECK-NEXT: %switch_50 = OpLabel
// CHECK-NEXT: OpStore %a %int_50
// CHECK-NEXT: OpBranch %switch_merge_6
          case 50:
            a = 50;
            break;
// CHECK-NEXT: %switch_51 = OpLabel
// CHECK-NEXT: OpBranch %switch_52
          case 51:
// CHECK-NEXT: %switch_52 = OpLabel
// CHECK-NEXT: OpStore %a %int_52
// CHECK-NEXT: OpBranch %switch_53
          case 52:
            a = 52;
// CHECK-NEXT: %switch_53 = OpLabel
// CHECK-NEXT: OpStore %a %int_53
// CHECK-NEXT: OpBranch %switch_merge_6
          case 53:
            a = 53;
            break;
// CHECK-NEXT: %switch_54 = OpLabel
// CHECK-NEXT: OpStore %a %int_54
// CHECK-NEXT: OpBranch %switch_merge_6
          case 54 : {
            a = 54;
            break;
          }
        }
// CHECK-NEXT: %switch_merge_6 = OpLabel
// CHECK-NEXT: OpBranch %switch_merge_7
    }
  }
// CHECK-NEXT: %switch_merge_7 = OpLabel



  ///////////////////////////////////////////////
  // Constant integer variables as case values //
  ///////////////////////////////////////////////

  const int r = 35;
  const int s = 45;
  const int t = 2*r + s;  // evaluates to 115.

// CHECK:      [[a8:%[0-9]+]] = OpLoad %int %a
// CHECK-NEXT: OpSelectionMerge %switch_merge_8 None
// CHECK-NEXT: OpSwitch [[a8]] %switch_merge_8 35 %switch_35 115 %switch_115
  switch(a) {
// CHECK-NEXT: %switch_35 = OpLabel
// CHECK-NEXT: [[r:%[0-9]+]] = OpLoad %int %r
// CHECK-NEXT: OpStore %result [[r]]
// CHECK-NEXT: OpBranch %switch_115
    case r:
      result = r;
// CHECK-NEXT: %switch_115 = OpLabel
// CHECK-NEXT: [[t:%[0-9]+]] = OpLoad %int %t
// CHECK-NEXT: OpStore %result [[t]]
// CHECK-NEXT: OpBranch %switch_merge_8
    case t:
      result = t;
      break;
// CHECK-NEXT: %switch_merge_8 = OpLabel
  }


  //////////////////////////////////////////////////////////////////
  // Using float as selector results in multiple casts in the AST //
  //////////////////////////////////////////////////////////////////
  float sel;
// CHECK:      [[floatSelector:%[0-9]+]] = OpLoad %float %sel
// CHECK-NEXT:           [[sel:%[0-9]+]] = OpConvertFToS %int [[floatSelector]]
// CHECK-NEXT:                          OpSelectionMerge %switch_merge_9 None
// CHECK-NEXT:                          OpSwitch [[sel]] %switch_merge_9 0 %switch_0_0
  switch ((int)sel) {
  case 0:
    result = 0;
    break;
  }
}
