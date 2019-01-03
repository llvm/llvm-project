# RUN: llvm-mc -triple=wasm32-unknown-unknown -mattr=+simd128,+nontrapping-fptoint,+exception-handling < %s | FileCheck %s
# this one is just here to see if it converts to .o without errors, but doesn't check any output:
# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj -mattr=+simd128,+nontrapping-fptoint,+exception-handling < %s

    .text
    .section .text.main,"",@
    .type    test0,@function
test0:
    # Test all types:
    .functype   test0 (i32, i64) -> (i32)
    .eventtype  __cpp_exception i32
    .local      f32, f64, v128, v128
    # Explicit getlocal/setlocal:
    get_local   2
    set_local   2
    # Immediates:
    i32.const   -1
    f64.const   0x1.999999999999ap1
    v128.const  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    v128.const  0, 1, 2, 3, 4, 5, 6, 7
    # Indirect addressing:
    get_local   0
    f64.store   0
    # Loops, conditionals, binary ops, calls etc:
    block       i32
    i32.const   1
    get_local   0
    i32.ge_s
    br_if       0        # 0: down to label0
.LBB0_1:
    loop        i32      # label1:
    call        something1@FUNCTION
    i64.const   1234
    i32.call    something2@FUNCTION
    i32.const   0
    call_indirect 0
    i32.const   1
    i32.add
    tee_local   0
    get_local   0
    i32.lt_s
    br_if       0        # 0: up to label1
.LBB0_2:
    end_loop
    end_block            # label0:
    get_local   4
    get_local   5
    block       void
    block       i64
    block       f32
    block       f64
    br_table {0, 1, 2}   # 2 entries, default
    end_block            # first entry jumps here.
    i32.const   1
    br 2
    end_block            # second entry jumps here.
    i32.const   2
    br 1
    end_block            # default jumps here.
    i32.const   3
    end_block            # "switch" exit.
    if          # void
    if          i32
    end_if
    else
    end_if
    f32x4.add
    # Test correct parsing of instructions with / and : in them:
    # TODO: enable once instruction has been added.
    #i32x4.trunc_s/f32x4:sat
    i32.trunc_s/f32
    try         except_ref
.LBB0_3:
    i32.catch   0
.LBB0_4:
    catch_all
.LBB0_5:
    end_try
    #i32.trunc_s:sat/f32
    get_global  __stack_pointer@GLOBAL
    end_function
.Lfunc_end0:
    .size	test0, .Lfunc_end0-test0
    .globaltype	__stack_pointer, i32

# CHECK:           .text
# CHECK-LABEL: test0:
# CHECK-NEXT:      .functype test0 (i32, i64) -> (i32)
# CHECK-NEXT:      .eventtype  __cpp_exception i32
# CHECK-NEXT:      .local      f32, f64
# CHECK-NEXT:      get_local   2
# CHECK-NEXT:      set_local   2
# CHECK-NEXT:      i32.const   -1
# CHECK-NEXT:      f64.const   0x1.999999999999ap1
# CHECK-NEXT:      v128.const  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
# CHECK-NEXT:      v128.const  0, 1, 2, 3, 4, 5, 6, 7
# CHECK-NEXT:      get_local   0
# CHECK-NEXT:      f64.store   0:p2align=0
# CHECK-NEXT:      block       i32
# CHECK-NEXT:      i32.const   1
# CHECK-NEXT:      get_local   0
# CHECK-NEXT:      i32.ge_s
# CHECK-NEXT:      br_if 0            # 0: down to label0
# CHECK-NEXT:  .LBB0_1:
# CHECK-NEXT:      loop        i32         # label1:
# CHECK-NEXT:      call        something1@FUNCTION
# CHECK-NEXT:      i64.const   1234
# CHECK-NEXT:      i32.call    something2@FUNCTION
# CHECK-NEXT:      i32.const   0
# CHECK-NEXT:      call_indirect 0
# CHECK-NEXT:      i32.const   1
# CHECK-NEXT:      i32.add
# CHECK-NEXT:      tee_local   0
# CHECK-NEXT:      get_local   0
# CHECK-NEXT:      i32.lt_s
# CHECK-NEXT:      br_if 0            # 0: up to label1
# CHECK-NEXT:  .LBB0_2:
# CHECK-NEXT:      end_loop
# CHECK-NEXT:      end_block                       # label0:
# CHECK-NEXT:      get_local   4
# CHECK-NEXT:      get_local   5
# CHECK-NEXT:      block
# CHECK-NEXT:      block       i64
# CHECK-NEXT:      block       f32
# CHECK-NEXT:      block       f64
# CHECK-NEXT:      br_table {0, 1, 2}  # 1: down to label4
# CHECK-NEXT:                          # 2: down to label3
# CHECK-NEXT:      end_block           # label5:
# CHECK-NEXT:      i32.const 1
# CHECK-NEXT:      br 2                # 2: down to label2
# CHECK-NEXT:      end_block           # label4:
# CHECK-NEXT:      i32.const 2
# CHECK-NEXT:      br 1                # 1: down to label2
# CHECK-NEXT:      end_block           # label3:
# CHECK-NEXT:      i32.const 3
# CHECK-NEXT:      end_block           # label2:
# CHECK-NEXT:      if
# CHECK-NEXT:      if          i32
# CHECK-NEXT:      end_if
# CHECK-NEXT:      else
# CHECK-NEXT:      end_if
# CHECK-NEXT:      f32x4.add
# CHECK-NEXT:      i32.trunc_s/f32
# CHECK-NEXT:      try         except_ref
# CHECK-NEXT:  .LBB0_3:
# CHECK-NEXT:      i32.catch   0
# CHECK-NEXT:  .LBB0_4:
# CHECK-NEXT:      catch_all
# CHECK-NEXT:  .LBB0_5:
# CHECK-NEXT:      end_try
# CHECK-NEXT:      get_global  __stack_pointer@GLOBAL
# CHECK-NEXT:      end_function

# CHECK:           .globaltype	__stack_pointer, i32
