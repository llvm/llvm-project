# 'shape' Dialect

Description of operations & types within the Shape dialect as well as their
[usage](#different-stages-of-lowering-shape-dialect).

[include "Dialects/ShapeDialectOps.md"]

## Different stages of lowering Shape dialect

In this section we shall give a brief overview of the different uses of the 
shape dialect and the lowering between these uses. Currently we have 3 worlds /
stages of lowering of shape functions:

1.  _Error monadic/error carrying/user specification_:
    This "input" form carries both the shape and whether in error state as
    value. Hence at this level all operations are pure operations producing and
    consuming values where the values could represent an error.

2.  _Constrained_:
    This form uses a variant of explicit evidence passing to allow leveraging
    existing compiler infrastructure to preserve safety information during
    optimization.

3.  _Side-effecting/asserting_:
    This final lowered form is imperative form with side-effecting ops (e.g.,
    assert) for final codegen.

We are going to do a quick step through of the lowering using the example of
a matmul.

Starting from the shape function of matmul in the error monadic form
below[^wip_form1]:

```mlir
shape.function_library @shplib {

func.func @matmul(%lhs: !shape.value_shape, %rhs: !shape.value_shape) -> !shape.shape {
  %c1 = shape.const_size 1
  %c2 = shape.const_size 2
  // We could also allow rank etc operations directly on value_shape too, that
  // would make it nicer as "input" language, but keeping it explicit inside the
  // IR instead and then we could have helper methods in front-end language.
  %lhs_shape = shape.shape_of %lhs : !shape.value_shape -> !shape.shape
  %rhs_shape = shape.shape_of %rhs : !shape.value_shape -> !shape.shape
  %lhs_rank = shape.rank %lhs_shape : !shape.shape -> !shape.size
  %rhs_rank = shape.rank %rhs_shape : !shape.shape -> !shape.size
  // This is not minimal as one could ensure the ranks are the same below, also a
  // variadic meet would make it more concise too.
  %r = "shape.meet"(%lhs_rank, %rhs_rank) : (!shape.size, !shape.size) -> !shape.size
  %rank = shape.meet %c2, %r, error="requires rank 2 operands" :
    !shape.size, !shape.size -> !shape.size
  %l0, %l1 = "shape.split_at"(%lhs_shape, %c1) :
    (!shape.shape, !shape.size) -> (!shape.shape, !shape.shape)
  %r0, %r1 = "shape.split_at"(%rhs_shape, %c1) :
    (!shape.shape, !shape.size) -> (!shape.shape, !shape.shape)
  %c = shape.meet %l1, %r0, error="inner dimensions required to match" :
    !shape.shape, !shape.shape -> !shape.shape
  %res = shape.concat %l0, %r1
  // Should have `shape.return %res requires %c, %rank` to enable
  return %res : !shape.shape
}

} mapping {
  foo.matmul = @matmul
}
```

*   We are using the default builtin func and return here. Preferably we'd use
    ‘shape\_func’ as a special function op that allows passing multiple results
    back that affect correct execution (e.g., serves as an error join)
    *   This would also means one can't reify it inside a regular function
        without handling the shape.return - that is a feature here as these are
        more of a template.
    *   Currently we also have not marked `meet` as having no side-effects to
        avoid DCE until we have `shape.return`, at which point computing the
        meet could be treated as purely computational returning error.
*   Meet represents a constraint that should hold, so should not be used to see
    *if* something is equal. E.g., this means `meet` can't be used to represent

    ```
       either(meet(x, y), meet(y,z))
    ```

*   This could have been written more concisely as something like

    ```
      concat(lhs[0], rhs[1]) if rank(lhs) == 2 &&
        rank(rhs) == 2 && lhs[1] == rhs[0]
    ```

    but not focusing on front-end proper here.

We are going to lower to "most" nested form directly (see
[test](https://github.com/tensorflow/tensorflow/blob/64062b5c51e04e370df26551d247496787d3f5c2/tensorflow/compiler/mlir/xla/tests/legalize-tf.mlir#L3088)
for an example reification along with legalization). In the above this was in a
separate shape function library, while here we would normally reify it as part
of lowering, but for simplicity will show as a standalone shape function.

```mlir
func.func @matmul_shape1(%lhs: tensor<*xf32>, %rhs: tensor<*xindex>) -> tensor<?xindex> {
  %c1 = shape.const_size 1
  %c2 = shape.const_size 2
  // We allow `shape.shape_of` to return either a `!shape.shape` or
  // `tensor<?xindex>` type, in the case where the input is a tensor the most
  // refined type is a tensor of `index` but not required.
  %lhs_shape = shape.shape_of %lhs : tensor<*xf32> -> !shape.shape
  %rhs_shape = shape.shape_of %rhs : tensor<*xf32> -> !shape.shape
  %lhs_rank = shape.rank %lhs_shape : !shape.shape -> !shape.size
  %rhs_rank = shape.rank %rhs_shape : !shape.shape -> !shape.size
  %w1 = shape.cstr_eq %lhs_rank, %rhs_rank : !shape.witness
  %res = shape.assuming %w1 -> tensor<?xindex> {
    %r1 = shape.any %lhs_rank, %rhs_rank : (!shape.size, !shape.size) -> !shape.size
    // Error message needs an addition, currently only on cstr_require.
    %w2 = shape.cstr_eq %c2, %r1, error="requires rank 2 operands"
    %res_1 = shape.assuming %w2 -> tensor<?xindex> {
      // Here the lowered
      //   %rank = shape.any %c2, %r1 (!shape.size, !shape.size) -> !shape.size
      // is dead and so elided further. But if `%rank` was actually consumed,
      // then it could have been folded in `shape.any`.
      %l0, %r0 = "shape.split_at"(%lhs_shape, %c1) :
        (!shape.shape, !shape.size) -> !shape.shape
      %l1, %r1 = "shape.split_at"(%lhs_shape, %c1) :
        (!shape.shape, !shape.size) -> !shape.shape
      %c = shape.meet %l1, %r0, error="inner dimensions required to match" :
        !shape.size, !shape.size -> !shape.size
      %res = concat(%l0, %r1)
      shape.assuming_yield %res
    }
    shape.assuming_yield %res_1
  }
  return %res : tensor<?xindex>
}
```

We can now hoist computations of constraint were possible (which in the case
below is not too many as we need to verify the rank before we can split)

```mlir
func.func @matmul_shape2(%lhs: tensor<*xf32>, %lhs: tensor<*xf32>) -> tensor<?xindex> {
  %c1 = shape.const_size 1
  %c2 = shape.const_size 2
  %lhs_shape = shape.shape_of %lhs : tensor<*xf32> -> tensor<?xindex>
  %rhs_shape = shape.shape_of %rhs : tensor<*xf32> -> tensor<?xindex>
  %lhs_rank = shape.rank %lhs_shape : tensor<?xindex> -> tensor<index>
  %rhs_rank = shape.rank %rhs_shape : tensor<?xindex> -> tensor<index>
  %w1 = shape.cstr_eq %c2, %lhs_rank, error="requires rank 2 operands"
  %w2 = shape.cstr_eq %c2, %rhs_rank, error="requires rank 2 operands"
  %w = shape.assuming_all %w1, %w2
  %res = shape.assuming %w -> tensor<?xindex> {
    %l0, %r0 = "shape.split_at"(%lhs_shape, %c1) :
      (tensor<?xindex>, !shape.size) -> tensor<?xindex>
    %l1, %r1 = "shape.split_at"(%lhs_shape, %c1) :
      (tensor<?xindex>, !shape.size) -> tensor<?xindex>
    %w3 = shape.cstr_eq %l1, %r0, error="inner dimensions required to match"
    %res_2 = shape.assuming %w3 {
      %res = concat(%l0, %r1)
      shape.assuming_yield %res
    }
    shape.assuming_yield %res_1
  }
  return %res
}
```

The above form can now be lowered to the fully imperative form (see
[test](https://github.com/tensorflow/mlir-hlo/blob/af14e1ded33c3164d4418c5d234b5b346b6d017c/tests/rank-specialization.mlir#L22)
for example).

```mlir
func.func @matmul_shape3(%lhs: tensor<*xf32>, %lhs: tensor<*xf32>) -> tensor<?xindex> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %lhs_shape = shape.shape_of %lhs : tensor<*xf32> -> tensor<?xindex>
  %rhs_shape = shape.shape_of %rhs : tensor<*xf32> -> tensor<?xindex>
  %lhs_rank = shape.rank %lhs_shape : tensor<?xindex> -> tensor<index>
  %rhs_rank = shape.rank %rhs_shape : tensor<?xindex> -> tensor<index>
  %w1 = shape.shape_eq %lhs_rank, %rhs_rank
  %w2 = shape.shape_eq %c2, %lhs_rank
  %w3 = and %w1, %w2
  assert %w3, "requires rank 2 operands"
  %l0, %l1 = shape.split_at(%lhs_shape, %c1) : tensor<?xindex>
  %r0, %r1 = shape.split_at(%rhs_shape, %c1) : tensor<?xindex>
  %w4 = shape.eq %l1, %r0
  assert %w4, "inner dimensions required to match"
  %res = concat(%l0, %r1)
  return %res
}
```

*   In this case form 3 is as easy and closer to form 1 (but only as no
    reordering was required). So it is a good question if the frontend authoring
    language could be more similar to the imperative form (under discussion).
*   The above form presented here is an intermittent form during a lowering
    pass. If used as input we would need to restrict the optimizations on it as
    the `shape` dialect operations are no longer connected by producer-consumer
    to enforce guard checking.

The above could be further lowered by using `tensor.dim`, `tensor.from_elements`
etc (or one could even lower these by way of, say, MHLO or TOSA dialect).

[^wip_form1]: This form is least use inside the current workflows and needs more work. In particular in the example we use `shape_func` where in the code we instead use standard func as first form 1 isn't used explicitly.
