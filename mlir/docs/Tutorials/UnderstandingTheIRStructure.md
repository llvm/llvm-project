# Understanding the IR Structure

The MLIR Language Reference describes the
[High Level Structure](../LangRef.md/#high-level-structure), this document
illustrates this structure through examples, and introduces at the same time the
C++ APIs involved in manipulating it.

We will implement a [pass](../PassManagement.md/#operation-pass) that traverses any
MLIR input and prints the entity inside the IR. A pass (or in general almost any
piece of IR) is always rooted with an operation. Most of the time the top-level
operation is a `ModuleOp`, the MLIR `PassManager` is actually limited to
operation on a top-level `ModuleOp`. As such a pass starts with an operation,
and so will our traversal:

```
  void runOnOperation() override {
    Operation *op = getOperation();
    resetIndent();
    printOperation(op);
  }
```

## Traversing the IR Nesting

The IR is recursively nested, an `Operation` can have one or multiple nested
`Region`s, each of which is actually a list of `Blocks`, each of which itself
wraps a list of `Operation`s. Our traversal will follow this structure with
three methods: `printOperation()`, `printRegion()`, and `printBlock()`.

The first method inspects the properties of an operation, before iterating on
the nested regions and print them individually:

```c++
  void printOperation(Operation *op) {
    // Print the operation itself and some of its properties
    printIndent() << "visiting op: '" << op->getName() << "' with "
                  << op->getNumOperands() << " operands and "
                  << op->getNumResults() << " results\n";
    // Print the operation attributes
    if (!op->getAttrs().empty()) {
      printIndent() << op->getAttrs().size() << " attributes:\n";
      for (NamedAttribute attr : op->getAttrs())
        printIndent() << " - '" << attr.getName() << "' : '"
                      << attr.getValue() << "'\n";
    }

    // Recurse into each of the regions attached to the operation.
    printIndent() << " " << op->getNumRegions() << " nested regions:\n";
    auto indent = pushIndent();
    for (Region &region : op->getRegions())
      printRegion(region);
  }
```

A `Region` does not hold anything other than a list of `Block`s:

```c++
  void printRegion(Region &region) {
    // A region does not hold anything by itself other than a list of blocks.
    printIndent() << "Region with " << region.getBlocks().size()
                  << " blocks:\n";
    auto indent = pushIndent();
    for (Block &block : region.getBlocks())
      printBlock(block);
  }
```

Finally, a `Block` has a list of arguments, and holds a list of `Operation`s:

```c++
  void printBlock(Block &block) {
    // Print the block intrinsics properties (basically: argument list)
    printIndent()
        << "Block with " << block.getNumArguments() << " arguments, "
        << block.getNumSuccessors()
        << " successors, and "
        // Note, this `.size()` is traversing a linked-list and is O(n).
        << block.getOperations().size() << " operations\n";

    // A block main role is to hold a list of Operations: let's recurse into
    // printing each operation.
    auto indent = pushIndent();
    for (Operation &op : block.getOperations())
      printOperation(&op);
  }
```

The code for the pass is available
[here in the repo](https://github.com/llvm/llvm-project/blob/main/mlir/test/lib/IR/TestPrintNesting.cpp)
and can be exercised with `mlir-opt -test-print-nesting`.

### Example

The Pass introduced in the previous section can be applied on the following IR
with `mlir-opt -test-print-nesting -allow-unregistered-dialect
llvm-project/mlir/test/IR/print-ir-nesting.mlir`:

```mlir
"builtin.module"() ( {
  %results:4 = "dialect.op1"() {"attribute name" = 42 : i32} : () -> (i1, i16, i32, i64)
  "dialect.op2"() ( {
    "dialect.innerop1"(%results#0, %results#1) : (i1, i16) -> ()
  },  {
    "dialect.innerop2"() : () -> ()
    "dialect.innerop3"(%results#0, %results#2, %results#3)[^bb1, ^bb2] : (i1, i32, i64) -> ()
  ^bb1(%1: i32):  // pred: ^bb0
    "dialect.innerop4"() : () -> ()
    "dialect.innerop5"() : () -> ()
  ^bb2(%2: i64):  // pred: ^bb0
    "dialect.innerop6"() : () -> ()
    "dialect.innerop7"() : () -> ()
  }) {"other attribute" = 42 : i64} : () -> ()
}) : () -> ()
```

And will yield the following output:

```
visiting op: 'builtin.module' with 0 operands and 0 results
 1 nested regions:
  Region with 1 blocks:
    Block with 0 arguments, 0 successors, and 2 operations
      visiting op: 'dialect.op1' with 0 operands and 4 results
      1 attributes:
       - 'attribute name' : '42 : i32'
       0 nested regions:
      visiting op: 'dialect.op2' with 0 operands and 0 results
      1 attributes:
       - 'other attribute' : '42 : i64'
       2 nested regions:
        Region with 1 blocks:
          Block with 0 arguments, 0 successors, and 1 operations
            visiting op: 'dialect.innerop1' with 2 operands and 0 results
             0 nested regions:
        Region with 3 blocks:
          Block with 0 arguments, 2 successors, and 2 operations
            visiting op: 'dialect.innerop2' with 0 operands and 0 results
             0 nested regions:
            visiting op: 'dialect.innerop3' with 3 operands and 0 results
             0 nested regions:
          Block with 1 arguments, 0 successors, and 2 operations
            visiting op: 'dialect.innerop4' with 0 operands and 0 results
             0 nested regions:
            visiting op: 'dialect.innerop5' with 0 operands and 0 results
             0 nested regions:
          Block with 1 arguments, 0 successors, and 2 operations
            visiting op: 'dialect.innerop6' with 0 operands and 0 results
             0 nested regions:
            visiting op: 'dialect.innerop7' with 0 operands and 0 results
             0 nested regions:
```

## Other IR Traversal Methods

In many cases, unwrapping the recursive structure of the IR is cumbersome and
you may be interested in using other helpers.

### Filtered iterator: `getOps<OpTy>()`

For example the `Block` class exposes a convenient templated method
`getOps<OpTy>()` that provided a filtered iterator. Here is an example:

```c++
  auto varOps = entryBlock.getOps<spirv::GlobalVariableOp>();
  for (spirv::GlobalVariableOp gvOp : varOps) {
     // process each GlobalVariable Operation in the block.
     ...
  }
```

Similarly, the `Region` class exposes the same `getOps` method that will iterate
on all the blocks in the region.

### Walkers

The `getOps<OpTy>()` is useful to iterate on some Operations immediately listed
inside a single block (or a single region), however it is frequently interesting
to traverse the IR in a nested fashion. To this end MLIR exposes the `walk()`
helper on `Operation`, `Block`, and `Region`. This helper takes a single
argument: a callback method that will be invoked for every operation recursively
nested under the provided entity.

```c++
  // Recursively traverse all the regions and blocks nested inside the function
  // and apply the callback on every single operation in post-order.
  getFunction().walk([&](mlir::Operation *op) {
    // process Operation `op`.
  });
```

The provided callback can be specialized to filter on a particular type of
Operation, for example the following will apply the callback only on `LinalgOp`
operations nested inside the function:

```c++
  getFunction().walk([](LinalgOp linalgOp) {
    // process LinalgOp `linalgOp`.
  });
```

Finally, the callback can optionally stop the walk by returning a
`WalkResult::interrupt()` value. For example the following walk will find all
`AllocOp` nested inside the function and interrupt the traversal if one of them
does not satisfy a criteria:

```c++
  WalkResult result = getFunction().walk([&](AllocOp allocOp) {
    if (!isValid(allocOp))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    // One alloc wasn't matching.
    ...
```

## Traversing the def-use chains

Another relationship in the IR is the one that links a `Value` with its users.
As defined in the
[language reference](../LangRef.md/#high-level-structure),
each Value is either a `BlockArgument` or the result of exactly one `Operation`
(an `Operation` can have multiple results, each of them is a separate `Value`).
The users of a `Value` are `Operation`s, through their arguments: each
`Operation` argument references a single `Value`.

Here is a code sample that inspects the operands of an `Operation` and prints
some information about them:

```c++
  // Print information about the producer of each of the operands.
  for (Value operand : op->getOperands()) {
    if (Operation *producer = operand.getDefiningOp()) {
      llvm::outs() << "  - Operand produced by operation '"
                   << producer->getName() << "'\n";
    } else {
      // If there is no defining op, the Value is necessarily a Block
      // argument.
      auto blockArg = operand.cast<BlockArgument>();
      llvm::outs() << "  - Operand produced by Block argument, number "
                   << blockArg.getArgNumber() << "\n";
    }
  }
```

Similarly, the following code sample iterates through the result `Value`s
produced by an `Operation` and for each result will iterate the users of these
results and print informations about them:

```c++
  // Print information about the user of each of the result.
  llvm::outs() << "Has " << op->getNumResults() << " results:\n";
  for (auto indexedResult : llvm::enumerate(op->getResults())) {
    Value result = indexedResult.value();
    llvm::outs() << "  - Result " << indexedResult.index();
    if (result.use_empty()) {
      llvm::outs() << " has no uses\n";
      continue;
    }
    if (result.hasOneUse()) {
      llvm::outs() << " has a single use: ";
    } else {
      llvm::outs() << " has "
                   << std::distance(result.getUses().begin(),
                                    result.getUses().end())
                   << " uses:\n";
    }
    for (Operation *userOp : result.getUsers()) {
      llvm::outs() << "    - " << userOp->getName() << "\n";
    }
  }
```

The illustrating code for this pass is available
[here in the repo](https://github.com/llvm/llvm-project/blob/main/mlir/test/lib/IR/TestPrintDefUse.cpp)
and can be exercised with `mlir-opt -test-print-defuse`.

The chaining of `Value`s and their uses can be viewed as following:

![Index Map Example](/includes/img/DefUseChains.svg)

The uses of a `Value` (`OpOperand` or `BlockOperand`) are also chained in a
doubly linked-list, which is particularly useful when replacing all uses of a
`Value` with a new one ("RAUW"):

![Index Map Example](/includes/img/Use-list.svg)
