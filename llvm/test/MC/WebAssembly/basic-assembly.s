# RUN: llvm-mc -triple=wasm32-unknown-unknown -mattr=+reference-types,atomics,+simd128,+nontrapping-fptoint,+exception-handling < %s | FileCheck %s
# Check that it converts to .o without errors, but don't check any output:
# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj -mattr=+reference-types,+atomics,+simd128,+nontrapping-fptoint,+exception-handling -o %t.o < %s


empty_func:
    .functype empty_func () -> ()
    end_function

test0:
# local labels can appear between label and its .functype.
.Ltest0begin:
    # Test all types:
    .functype   test0 (i32, i64) -> (i32)
    .eventtype  __cpp_exception i32
    .local      f32, f64, v128, v128
    # Explicit getlocal/setlocal:
    local.get   2
    local.set   2
    # Immediates:
    i32.const   -1
    f64.const   0x1.999999999999ap1
    f32.const   -1.0
    f32.const   -infinity
    f32.const   nan
    v128.const  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    v128.const  0, 1, 2, 3, 4, 5, 6, 7
    # Indirect addressing:
    local.get   0
    f64.store   1234:p2align=4
    f64.store   1234     # Natural alignment (3)
    # Loops, conditionals, binary ops, calls etc:
    block       i32
    i32.const   1
    local.get   0
    i32.ge_s
    br_if       0        # 0: down to label0
.LBB0_1:
    loop        i32      # label1:
    call        something1
    i64.const   1234
    call        something2
    i32.const   0
    call_indirect (i32, f64) -> ()
    i32.const   1
    i32.add
    local.tee   0
    local.get   0
    i32.lt_s
    br_if       0        # 0: up to label1
.LBB0_2:
    end_loop
    end_block            # label0:
    local.get   4
    local.get   5
    block       void
    block       i64
    block       f32
    block       f64
    block       () -> (i32, i32)
    i32.const   1
    i32.const   2
    end_block
    drop
    drop
    br_table {0, 1, 2}   # 2 entries, default
    end_block            # first entry jumps here.
    i32.const   1
    br          2
    end_block            # second entry jumps here.
    i32.const   2
    br          1
    end_block            # default jumps here.
    i32.const   3
    end_block            # "switch" exit.
    if                   # void
    if          i32
    end_if
    else
    end_if
    f32x4.add
    # Test correct parsing of instructions with / and : in them:
    # TODO: enable once instruction has been added.
    #i32x4.trunc_sat_f32x4_s
    i32.trunc_f32_s
    try
    i32.atomic.load 0
    memory.atomic.notify 0
.LBB0_3:
    catch       __cpp_exception
    local.set   0
    end_try
    i32.const   .L.str
    i32.load8_u .L.str+2
    i32.load16_u .L.str:p2align=0
    throw 0
.LBB0_4:
    #i32.trunc_sat_f32_s
    global.get  __stack_pointer
    end_function

    .section    .rodata..L.str,"",@
    .hidden     .L.str
    .type       .L.str,@object
.L.str:
    .int8       'H'
    .asciz      "ello, World!"
    .int16      1234
    .int64      5000000000
    .int32      2000000000
    .size       .L.str, 28

    .section    .init_array.42,"",@
    .p2align    2
    .int32      test0

    .ident      "clang version 9.0.0 (trunk 364502) (llvm/trunk 364571)"
    .globaltype __stack_pointer, i32

.tabletype empty_eref_table, externref
empty_eref_table:

.tabletype empty_fref_table, funcref
empty_fref_table:


# CHECK:           .text
# CHECK-LABEL: empty_func:
# CHECK-NEXT:      .functype	empty_func () -> ()
# CHECK-NEXT:      end_function
# CHECK-LABEL: test0:
# CHECK-NEXT:  .Ltest0begin:
# CHECK-NEXT:      .functype   test0 (i32, i64) -> (i32)
# CHECK-NEXT:      .eventtype  __cpp_exception i32
# CHECK-NEXT:      .local      f32, f64
# CHECK-NEXT:      local.get   2
# CHECK-NEXT:      local.set   2
# CHECK-NEXT:      i32.const   -1
# CHECK-NEXT:      f64.const   0x1.999999999999ap1
# CHECK-NEXT:      f32.const   -0x1p0
# CHECK-NEXT:      f32.const   -infinity
# CHECK-NEXT:      f32.const   nan
# CHECK-NEXT:      v128.const  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
# CHECK-NEXT:      v128.const  0, 1, 2, 3, 4, 5, 6, 7
# CHECK-NEXT:      local.get   0
# CHECK-NEXT:      f64.store   1234:p2align=4
# CHECK-NEXT:      f64.store   1234
# CHECK-NEXT:      block       i32
# CHECK-NEXT:      i32.const   1
# CHECK-NEXT:      local.get   0
# CHECK-NEXT:      i32.ge_s
# CHECK-NEXT:      br_if       0       # 0: down to label0
# CHECK-NEXT:  .LBB0_1:
# CHECK-NEXT:      loop        i32     # label1:
# CHECK-NEXT:      call        something1
# CHECK-NEXT:      i64.const   1234
# CHECK-NEXT:      call        something2
# CHECK-NEXT:      i32.const   0
# CHECK-NEXT:      call_indirect __indirect_function_table, (i32, f64) -> ()
# CHECK-NEXT:      i32.const   1
# CHECK-NEXT:      i32.add
# CHECK-NEXT:      local.tee   0
# CHECK-NEXT:      local.get   0
# CHECK-NEXT:      i32.lt_s
# CHECK-NEXT:      br_if       0       # 0: up to label1
# CHECK-NEXT:  .LBB0_2:
# CHECK-NEXT:      end_loop
# CHECK-NEXT:      end_block           # label0:
# CHECK-NEXT:      local.get   4
# CHECK-NEXT:      local.get   5
# CHECK-NEXT:      block
# CHECK-NEXT:      block       i64
# CHECK-NEXT:      block       f32
# CHECK-NEXT:      block       f64
# CHECK-NEXT:      block       () -> (i32, i32)
# CHECK-NEXT:      i32.const   1
# CHECK-NEXT:      i32.const   2
# CHECK-NEXT:      end_block
# CHECK-NEXT:      drop
# CHECK-NEXT:      drop
# CHECK-NEXT:      br_table {0, 1, 2}  # 1: down to label4
# CHECK-NEXT:                          # 2: down to label3
# CHECK-NEXT:      end_block           # label5:
# CHECK-NEXT:      i32.const   1
# CHECK-NEXT:      br          2       # 2: down to label2
# CHECK-NEXT:      end_block           # label4:
# CHECK-NEXT:      i32.const   2
# CHECK-NEXT:      br          1       # 1: down to label2
# CHECK-NEXT:      end_block           # label3:
# CHECK-NEXT:      i32.const   3
# CHECK-NEXT:      end_block           # label2:
# CHECK-NEXT:      if
# CHECK-NEXT:      if          i32
# CHECK-NEXT:      end_if
# CHECK-NEXT:      else
# CHECK-NEXT:      end_if
# CHECK-NEXT:      f32x4.add
# CHECK-NEXT:      i32.trunc_f32_s
# CHECK-NEXT:      try
# CHECK-NEXT:      i32.atomic.load 0
# CHECK-NEXT:      memory.atomic.notify 0
# CHECK-NEXT:  .LBB0_3:
# CHECK-NEXT:      catch       __cpp_exception
# CHECK-NEXT:      local.set   0
# CHECK-NEXT:      end_try
# CHECK-NEXT:      i32.const   .L.str
# CHECK-NEXT:      i32.load8_u .L.str+2
# CHECK-NEXT:      i32.load16_u .L.str:p2align=0
# CHECK-NEXT:      throw       0
# CHECK-NEXT:  .LBB0_4:
# CHECK-NEXT:      global.get  __stack_pointer
# CHECK-NEXT:      end_function

# CHECK:           .section    .rodata..L.str,"",@
# CHECK-NEXT:      .hidden     .L.str
# CHECK-NEXT:  .L.str:
# CHECK-NEXT:      .int8       72
# CHECK-NEXT:      .asciz      "ello, World!"
# CHECK-NEXT:      .int16      1234
# CHECK-NEXT:      .int64      5000000000
# CHECK-NEXT:      .int32      2000000000
# CHECK-NEXT:      .size       .L.str, 28

# CHECK:           .section    .init_array.42,"",@
# CHECK-NEXT:      .p2align    2
# CHECK-NEXT:      .int32      test0

# CHECK:           .globaltype __stack_pointer, i32

# CHECK:           .tabletype empty_eref_table, externref
# CHECK-NEXT: empty_eref_table:

# CHECK:           .tabletype empty_fref_table, funcref
# CHECK-NEXT: empty_fref_table:
