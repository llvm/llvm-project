//===- IndexDialect.td - Index dialect definition ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INDEX_DIALECT
#define INDEX_DIALECT

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// IndexDialect
//===----------------------------------------------------------------------===//

def IndexDialect : Dialect {
  let name = "index";

  let summary = "The Index dialect";
  let description = [{
    The Index dialect contains operations for manipulating values of the builtin
    `index` type. The index type models target-specific values of pointer width,
    like `intptr_t`. Index values are typically used as loop bounds, array
    subscripts, tensor dimensions, etc.

    The operations in this dialect operate exclusively on scalar index types.
    The dialect and its operations treat the index type as signless and contains
    signed and unsigned versions of certain operations where the distinction is
    meaningful. In particular, the operations and transformations are careful to
    be aware of the target-independent-ness of the index type, such as when
    folding.

    The folding semantics of the Index dialect operations ensure that folding
    produces the same results irrespective of the eventual target pointer width.
    All index constants are stored in `APInt`s of maximum index bitwidth: 64.
    Operations are folded using 64-bit integer arithmetic.

    For operations where the values of the upper 32 bits don't impact the values
    of the lower 32 bits, no additional handling is required because if the
    target is 32-bit, the truncated folded result will be the same as if the
    operation were computed with 32-bit arithmetic, and if the target is 64-bit,
    the fold result is valid by default.

    Consider addition: an overflow in 32-bit is the same as truncating the
    result computed in 64-bit. For example, `add(0x800000008, 0x800000008)` is
    `0x1000000010` in 64-bit, which truncates to `0x10`, the same result as
    truncating the operands first: `add(0x08, 0x08)`. Specifically, an operation
    `f` can always be folded if it satisfies the following for all 64-bit values
    of `a` and `b`:

    ```
    trunc(f(a, b)) = f(trunc(a), trunc(b))
    ```

    When materializing target-specific code, constants just need to be truncated
    as appropriate.

    Operations where the values of the upper 32 bits do impact the values of the
    lower 32 bits are not folded if the results would be different in 32-bit.
    These are operations that right shift -- division, remainder, etc. These
    operations are only folded for subsets of `a` and `b` for which the above
    property is satisfied. This is checked per fold attempt.

    Consider division: the 32-bit computation will differ from 64-bit if the
    latter results in a high bit shifted into the lower 32 bits. For example,
    `div(0x100000002, 2)` is `0x80000001` in 64-bit but `0x01` in 32-bit; it
    cannot be folded. However, `div(0x200000002, 2)` can be folded. The 64-bit
    result is `0x100000001`, which truncated to 32 bits is `0x01`. The 32-bit
    result of the operation with truncated operands `div(0x02, 2)` which is
    `0x01`, the same as truncating the 64-bit result.
  }];

  let cppNamespace = "::mlir::index";

  let extraClassDeclaration = [{
    /// Register all dialect attributes.
    void registerAttributes();
    /// Register all dialect operations.
    void registerOperations();
  }];

  let hasConstantMaterializer = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // INDEX_DIALECT
