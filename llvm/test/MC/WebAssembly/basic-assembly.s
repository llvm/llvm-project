# RUN: llvm-mc -triple=wasm32-unknown-unknown-elf < %s | FileCheck %s
# RUN: llvm-mc -triple=wasm32-unknown-unknown < %s | FileCheck %s

    .text
    .type    test0,@function
test0:
    # Test all types:
    .param      i32, i64
    .local      f32, f64  #, i8x16, i16x8, i32x4, f32x4
    # Explicit getlocal/setlocal:
    get_local   $push0=, 2
    set_local   2, $pop0=
    # Implicit locals & immediates:
    i32.const   $0=, -1
    f64.const   $3=, 0x1.999999999999ap1
    # Indirect addressing:
    get_local   $push1=, 0
    f64.store   0($pop1), $3
    # Loops, conditionals, binary ops, calls etc:
    block
    i32.const   $push2=, 1
    get_local   $push7=, 0
    i32.ge_s    $push0=, $pop2, $pop7
    br_if       0, $pop0        # 0: down to label0
.LBB0_1:
    loop             # label1:
    call        $drop=, something1@FUNCTION
    i64.const   $push10=, 1234
    i32.call    $push8=, something2@FUNCTION, $pop10
    i32.const   $push11=, 0
    call_indirect $pop11
    i32.const   $push5=, 1
    i32.add     $push4=, $pop8, $pop5
    tee_local   $push3=, 0, $pop4
    get_local   $push9=, 0
    i32.lt_s    $push1=, $pop3, $pop9
    br_if       0, $pop1        # 0: up to label1
.LBB0_2:
    end_loop
    end_block                       # label0:
    end_function


# CHECK:           .text
# CHECK-LABEL: test0:
# CHECK-NEXT:      .param      i32, i64
# CHECK-NEXT:      .local      f32, f64
# CHECK-NEXT:      get_local   $push0=, 2
# CHECK-NEXT:      set_local   2, $pop0
# CHECK-NEXT:      i32.const   $0=, -1
# CHECK-NEXT:      f64.const   $3=, 0x1.999999999999ap1
# CHECK-NEXT:      get_local   $push1=, 0
# CHECK-NEXT:      f64.store   0($pop1):p2align=0, $3
# CHECK-NEXT:      block
# CHECK-NEXT:      i32.const   $push2=, 1
# CHECK-NEXT:      get_local   $push7=, 0
# CHECK-NEXT:      i32.ge_s    $push0=, $pop2, $pop7
# CHECK-NEXT:      br_if 0,    $pop0        # 0: down to label0
# CHECK-NEXT:  .LBB0_1:
# CHECK-NEXT:      loop                    # label1:
# CHECK-NEXT:      call        something1@FUNCTION
# CHECK-NEXT:      i64.const   $push10=, 1234
# CHECK-NEXT:      i32.call    $push8=, something2@FUNCTION
# CHECK-NEXT:      i32.const   $push11=, 0
# CHECK-NEXT:      call_indirect
# CHECK-NEXT:      i32.const   $push5=, 1
# CHECK-NEXT:      i32.add     $push4=, $pop8, $pop5
# CHECK-NEXT:      tee_local   $push3=, 0, $pop4
# CHECK-NEXT:      get_local   $push9=, 0
# CHECK-NEXT:      i32.lt_s    $push1=, $pop3, $pop9
# CHECK-NEXT:      br_if 0,    $pop1        # 0: up to label1
# CHECK-NEXT:  .LBB0_2:
# CHECK-NEXT:      end_loop
# CHECK-NEXT:      end_block                       # label0:
# CHECK-NEXT:      end_function
