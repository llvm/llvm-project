# RUN: llvm-mc -triple=wasm64-unknown-unknown -mattr=+atomics,+unimplemented-simd128,+nontrapping-fptoint,+exception-handling < %s | FileCheck %s
# Check that it converts to .o without errors, but don't check any output:
# RUN: llvm-mc -triple=wasm64-unknown-unknown -filetype=obj -mattr=+atomics,+unimplemented-simd128,+nontrapping-fptoint,+exception-handling -o %t.o < %s

# Most of our other tests are for wasm32, this one adds some wasm64 specific tests.

test:
    .functype   test (i64) -> ()
    .local      i64

    ### basic loads

    i64.const   0         # get i64 from constant.
    f32.load    0
    drop

    local.get   0         # get i64 from local.
    f32.load    0
    drop

#    i64.const   .L.str    # get i64 relocatable.
#    f32.load    0
#    drop

    global.get  myglob64  # get i64 from global
    f32.load    0
    drop

    ### basic stores

    f32.const   0.0
    i64.const   0         # get i64 from constant.
    f32.store   0

    f32.const   0.0
    local.get   0         # get i64 from local.
    f32.store   0

#    f32.const   0.0
#    i64.const   .L.str    # get i64 relocatable.
#    f32.store   0

    f32.const   0.0
    global.get  myglob64  # get i64 from global
    f32.store   0

    end_function

    .section    .rodata..L.str,"",@
    .hidden     .L.str
    .type       .L.str,@object
.L.str:
    .asciz      "Hello, World!"

    .globaltype myglob64, i64



# CHECK:              .functype       test (i64) -> ()
# CHECK-NEXT:         .local          i64


# CHECK:              i64.const       0
# CHECK-NEXT:         f32.load        0
# CHECK-NEXT:         drop

# CHECK:              local.get       0
# CHECK-NEXT:         f32.load        0
# CHECK-NEXT:         drop

# NCHECK:              i64.const       .L.str
# NCHECK-NEXT:         f32.load        0
# NCHECK-NEXT:         drop

# CHECK:              global.get      myglob64
# CHECK-NEXT:         f32.load        0
# CHECK-NEXT:         drop


# CHECK:              f32.const       0x0p0
# CHECK-NEXT:         i64.const       0
# CHECK-NEXT:         f32.store       0

# CHECK:              f32.const       0x0p0
# CHECK-NEXT:         local.get       0
# CHECK-NEXT:         f32.store       0

# NCHECK:              f32.const       0x0p0
# NCHECK-NEXT:         i64.const       .L.str
# NCHECK-NEXT:         f32.store       0

# CHECK:              f32.const       0x0p0
# CHECK-NEXT:         global.get      myglob64
# CHECK-NEXT:         f32.store       0


# CHECK:              end_function
# CHECK-NEXT: .Ltmp0:
# CHECK-NEXT:         .size   test, .Ltmp0-test

# CHECK:              .section        .rodata..L.str,"",@
# CHECK-NEXT:         .hidden .L.str
# CHECK-NEXT: .L.str:
# CHECK-NEXT:         .asciz  "Hello, World!"

# CHECK:              .globaltype     myglob64, i64
