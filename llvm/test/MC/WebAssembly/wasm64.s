# RUN: llvm-mc -triple=wasm64-unknown-unknown -mattr=+atomics,+simd128,+nontrapping-fptoint,+exception-handling < %s | FileCheck %s
# RUN: llvm-mc -triple=wasm64-unknown-unknown -filetype=obj -mattr=+atomics,+simd128,+nontrapping-fptoint,+exception-handling -o - < %s | obj2yaml | FileCheck %s -check-prefix=BIN

# Most of our other tests are for wasm32, this one adds some wasm64 specific tests.

.globaltype myglob64, i64
.globaltype __stack_pointer, i64

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

    i64.const   .L.str    # get i64 relocatable.
    f32.load    0
    drop

    global.get  myglob64  # get i64 from global
    f32.load    0
    drop

    i64.const   0
    f32.load    .L.str    # relocatable offset!
    drop

    ### basic stores

    i64.const   0         # get i64 from constant.
    f32.const   0.0
    f32.store   0

    local.get   0         # get i64 from local.
    f32.const   0.0
    f32.store   0

    i64.const   .L.str    # get i64 relocatable.
    f32.const   0.0
    f32.store   0

    global.get  myglob64  # get i64 from global
    f32.const   0.0
    f32.store   0

    i64.const   0
    f32.const   0.0
    f32.store   .L.str    # relocatable offset!

    ### 64-bit SP

    global.get  __stack_pointer
    drop

    end_function

    .section    .rodata..L.str,"",@
    .hidden     .L.str
    .type       .L.str,@object
.L.str:
    .asciz      "Hello, World!!!"
    .int64      .L.str    # relocatable inside data.
    .size       .L.str, 24


# CHECK:              .globaltype     myglob64, i64

# CHECK:              .functype       test (i64) -> ()
# CHECK-NEXT:         .local          i64


# CHECK:              i64.const       0
# CHECK-NEXT:         f32.load        0
# CHECK-NEXT:         drop

# CHECK:              local.get       0
# CHECK-NEXT:         f32.load        0
# CHECK-NEXT:         drop

# CHECK:              i64.const       .L.str
# CHECK-NEXT:         f32.load        0
# CHECK-NEXT:         drop

# CHECK:              global.get      myglob64
# CHECK-NEXT:         f32.load        0
# CHECK-NEXT:         drop

# CHECK:              i64.const       0
# CHECK-NEXT:         f32.load        .L.str
# CHECK-NEXT:         drop


# CHECK:              i64.const       0
# CHECK-NEXT:         f32.const       0x0p0
# CHECK-NEXT:         f32.store       0

# CHECK:              local.get       0
# CHECK-NEXT:         f32.const       0x0p0
# CHECK-NEXT:         f32.store       0

# CHECK:              i64.const       .L.str
# CHECK-NEXT:         f32.const       0x0p0
# CHECK-NEXT:         f32.store       0

# CHECK:              global.get      myglob64
# CHECK-NEXT:         f32.const       0x0p0
# CHECK-NEXT:         f32.store       0

# CHECK:              i64.const       0
# CHECK-NEXT:         f32.const       0x0p0
# CHECK-NEXT:         f32.store       .L.str


# CHECK:              end_function
# CHECK-NEXT: .Ltmp0:
# CHECK-NEXT:         .size   test, .Ltmp0-test

# CHECK:              .section        .rodata..L.str,"",@
# CHECK-NEXT:         .hidden .L.str
# CHECK-NEXT: .L.str:
# CHECK-NEXT:         .asciz  "Hello, World!!!"
# CHECK-NEXT:         .int64      .L.str
# CHECK-NEXT:         .size       .L.str, 24



# BIN:      --- !WASM
# BIN-NEXT: FileHeader:
# BIN-NEXT:   Version:         0x1
# BIN-NEXT: Sections:
# BIN-NEXT:   - Type:            TYPE
# BIN-NEXT:     Signatures:
# BIN-NEXT:       - Index:           0
# BIN-NEXT:         ParamTypes:
# BIN-NEXT:           - I64
# BIN-NEXT:         ReturnTypes:     []
# BIN-NEXT:   - Type:            IMPORT
# BIN-NEXT:     Imports:
# BIN-NEXT:       - Module:          env
# BIN-NEXT:         Field:           __linear_memory
# BIN-NEXT:         Kind:            MEMORY
# BIN-NEXT:         Memory:
# BIN-NEXT:           Flags:           [ IS_64 ]
# BIN-NEXT:           Minimum:         0x1
# BIN-NEXT:       - Module:          env
# BIN-NEXT:         Field:           myglob64
# BIN-NEXT:         Kind:            GLOBAL
# BIN-NEXT:         GlobalType:      I64
# BIN-NEXT:         GlobalMutable:   true
# BIN-NEXT:       - Module:          env
# BIN-NEXT:         Field:           __stack_pointer
# BIN-NEXT:         Kind:            GLOBAL
# BIN-NEXT:         GlobalType:      I64
# BIN-NEXT:         GlobalMutable:   true
# BIN-NEXT:   - Type:            FUNCTION
# BIN-NEXT:     FunctionTypes:   [ 0 ]
# BIN-NEXT:   - Type:            DATACOUNT
# BIN-NEXT:     Count:           1
# BIN-NEXT:   - Type:            CODE
# BIN-NEXT:     Relocations:
# BIN-NEXT:       - Type:            R_WASM_MEMORY_ADDR_SLEB64
# BIN-NEXT:         Index:           1
# BIN-NEXT:         Offset:          0x13
# BIN-NEXT:       - Type:            R_WASM_GLOBAL_INDEX_LEB
# BIN-NEXT:         Index:           2
# BIN-NEXT:         Offset:          0x22
# BIN-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB64
# BIN-NEXT:         Index:           1
# BIN-NEXT:         Offset:          0x2F
# BIN-NEXT:       - Type:            R_WASM_MEMORY_ADDR_SLEB64
# BIN-NEXT:         Index:           1
# BIN-NEXT:         Offset:          0x4F
# BIN-NEXT:       - Type:            R_WASM_GLOBAL_INDEX_LEB
# BIN-NEXT:         Index:           2
# BIN-NEXT:         Offset:          0x62
# BIN-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB64
# BIN-NEXT:         Index:           1
# BIN-NEXT:         Offset:          0x78
# BIN-NEXT:       - Type: R_WASM_GLOBAL_INDEX_LEB
# BIN-NEXT:         Index: 3
# BIN-NEXT:         Offset: 0x83
# BIN-NEXT:     Functions:
# BIN-NEXT:       - Index:           0
# BIN-NEXT:         Locals:
# BIN-NEXT:           - Type:            I64
# BIN-NEXT:             Count:           1
# BIN-NEXT:         Body:            42002A02001A20002A02001A42808080808080808080002A02001A2380808080002A02001A42002A02808080808080808080001A4200430000000038020020004300000000380200428080808080808080800043000000003802002380808080004300000000380200420043000000003802808080808080808080002381808080001A0B
# BIN-NEXT:   - Type:            DATA
# BIN-NEXT:     Relocations:
# BIN-NEXT:       - Type:            R_WASM_MEMORY_ADDR_I64
# BIN-NEXT:         Index:           1
# BIN-NEXT:         Offset:          0x16
# BIN-NEXT:     Segments:
# BIN-NEXT:       - SectionOffset:   6
# BIN-NEXT:         InitFlags:       0
# BIN-NEXT:         Offset:
# BIN-NEXT:           Opcode:          I64_CONST
# BIN-NEXT:           Value:           0
# BIN-NEXT:         Content:         48656C6C6F2C20576F726C64212121000000000000000000
# BIN-NEXT:   - Type:            CUSTOM
# BIN-NEXT:     Name:            linking
# BIN-NEXT:     Version:         2
# BIN-NEXT:     SymbolTable:
# BIN-NEXT:       - Index:           0
# BIN-NEXT:         Kind:            FUNCTION
# BIN-NEXT:         Name:            test
# BIN-NEXT:         Flags:           [ BINDING_LOCAL ]
# BIN-NEXT:         Function:        0
# BIN-NEXT:       - Index:           1
# BIN-NEXT:         Kind:            DATA
# BIN-NEXT:         Name:            .L.str
# BIN-NEXT:         Flags:           [ BINDING_LOCAL, VISIBILITY_HIDDEN ]
# BIN-NEXT:         Segment:         0
# BIN-NEXT:         Size:            24
# BIN-NEXT:       - Index:           2
# BIN-NEXT:         Kind:            GLOBAL
# BIN-NEXT:         Name:            myglob64
# BIN-NEXT:         Flags:           [ UNDEFINED ]
# BIN-NEXT:         Global:          0
# BIN-NEXT:       - Index:           3
# BIN-NEXT:         Kind:            GLOBAL
# BIN-NEXT:         Name:            __stack_pointer
# BIN-NEXT:         Flags:           [ UNDEFINED ]
# BIN-NEXT:         Global:          1
# BIN-NEXT:     SegmentInfo:
# BIN-NEXT:       - Index:           0
# BIN-NEXT:         Name:            .rodata..L.str
# BIN-NEXT:         Alignment:       0
# BIN-NEXT:         Flags:           [  ]
# BIN-NEXT: ...
