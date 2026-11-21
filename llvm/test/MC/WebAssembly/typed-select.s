# RUN: llvm-mc -show-encoding -triple=wasm32-unknown-unknown -mattr=+reference-types < %s | FileCheck %s
# RUN: llvm-mc -show-encoding -triple=wasm64-unknown-unknown -mattr=+reference-types < %s | FileCheck %s

select_i32:
    .functype select_i32 () -> (i32)
    i32.const 1
    i32.const 2
    i32.const 0
    # CHECK: select i32 # encoding: [0x1c,0x01,0x7f]
    select i32
    end_function

select_i64:
    .functype select_i64 () -> (i64)
    i64.const 1
    i64.const 2
    i32.const 0
    # CHECK: select i64 # encoding: [0x1c,0x01,0x7e]
    select i64
    end_function

select_f32:
    .functype select_f32 () -> (f32)
    f32.const 0x1p+0
    f32.const 0x1p+1
    i32.const 0
    # CHECK: select f32 # encoding: [0x1c,0x01,0x7d]
    select f32
    end_function

select_f64:
    .functype select_f64 () -> (f64)
    f64.const 0x1p+0
    f64.const 0x1p+1
    i32.const 0
    # CHECK: select f64 # encoding: [0x1c,0x01,0x7c]
    select f64
    end_function

select_funcref:
    .functype select_funcref () -> (funcref)
    ref.null_func
    ref.null_func
    i32.const 0
    # CHECK: select funcref # encoding: [0x1c,0x01,0x70]
    select funcref
    end_function

select_externref:
    .functype select_externref () -> (externref)
    ref.null_extern
    ref.null_extern
    i32.const 0
    # CHECK: select externref # encoding: [0x1c,0x01,0x6f]
    select externref
    end_function

select_exnref:
    .functype select_exnref () -> (exnref)
    ref.null_exn
    ref.null_exn
    i32.const 0
    # CHECK: select exnref # encoding: [0x1c,0x01,0x69]
    select exnref
    end_function

# A select t* can declare a vec of any length. Multi-value selects are not
# produced by current tools but the encoding is well-formed and must round
# trip through the assembler.
select_multi:
    .functype select_multi () -> (i32, i64)
    i32.const 1
    i64.const 1
    i32.const 2
    i64.const 2
    i32.const 0
    # CHECK: select i32 i64 # encoding: [0x1c,0x02,0x7f,0x7e]
    select i32 i64
    end_function
