// RUN: llvm-mc -triple=amdgcn -mcpu=fiji %s | FileCheck %s --check-prefix=VI

//===----------------------------------------------------------------------===//
// Example of reg[expr] and reg[epxr1:expr2] syntax in macros.
//===----------------------------------------------------------------------===//

.macro REG_NUM_EXPR_EXAMPLE width iter iter_end
    .if \width == 4
      flat_load_dwordx4   v[8 + (\iter * 4):8 + (\iter * 4) + 3], v[2:3]
    .else
      flat_load_dword     v[8 + \iter], v[2:3]
    .endif

    .if (\iter_end - \iter)
      REG_NUM_EXPR_EXAMPLE \width, (\iter + 1), \iter_end
    .endif
 .endm

REG_NUM_EXPR_EXAMPLE 4, 0, 0
// VI:   flat_load_dwordx4 v[8:11], v[2:3]

REG_NUM_EXPR_EXAMPLE 1, 0, 0
// VI:   flat_load_dword v8, v[2:3]

REG_NUM_EXPR_EXAMPLE 4, 1, 4
// VI:   flat_load_dwordx4 v[12:15], v[2:3]
// VI:   flat_load_dwordx4 v[16:19], v[2:3]
// VI:   flat_load_dwordx4 v[20:23], v[2:3]
// VI:   flat_load_dwordx4 v[24:27], v[2:3]

REG_NUM_EXPR_EXAMPLE 1, 1, 4
// VI:   flat_load_dword v9, v[2:3]
// VI:   flat_load_dword v10, v[2:3]
// VI:   flat_load_dword v11, v[2:3]
// VI:   flat_load_dword v12, v[2:3]
