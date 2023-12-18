# Chapter 2: Adding a Simple New Transformation Operation

## Setting Up to Add New Transformations

Before defining a new transform operation, we need to choose where its implementation should be located. While MLIR encourages upstream contributions, it is not always possible or even desirable to modify the main Transform dialect, for example, if the transformation is specific to some out-of-tree dialect that is not itself available upstream.

The Transform dialect uses the dialect extension mechanism to allow additional operations to be injected without modifying the dialect itself. Dialect extensions are registered with the context and loaded when the dialect itself is loaded. Extension definition is straightforward:

```cpp
// In MyExtension.cpp.
#include "mlir/Dialect/Transform/IR/TransformDialect.h"

// Define a new transform dialect extension. This uses the CRTP idiom to identify
// extensions.
class MyExtension : public ::mlir::transform::TransformDialectExtension<MyExtension> {
public:
  // The extension must derive the base constructor.
  using Base::Base;

  // This function initializes the extension, similarly to `initialize` in dialect 
  // definitions. List individual operations and dependent dialects here.
  void init();
};

void MyExtension::init() {
  // Similarly to dialects, an extension can declare a dependent dialect. This dialect 
  // will be loaded along with the extension and, therefore, along with the Transform 
  // dialect. Only declare as dependent the dialects that contain the attributes or 
  // types used by transform operations. Do NOT declare as dependent the dialects 
  // produced during the transformation.
  // declareDependentDialect<MyDialect>();

  // When transformations are applied, they may produce new operations from previously
  // unloaded dialects. Typically, a pass would need to declare itself dependent on
  // the dialects containing such new operations. To avoid confusion with the dialects
  // the extension itself depends on, the Transform dialects differentiates between:
  //   - dependent dialects, which are used by the transform operations, and
  //   - generated dialects, which contain the entities (attributes, operations, 
  //     types) that may be produced by applying the transformation even when not
  //     present in the original payload IR.
  // In the following chapter, we will be add operations that generate function calls
  // and structured control flow operations, so let's declare the corresponding
  // dialects as generated.
  declareGeneratedDialect<::mlir::scf::SCFDialect>();
  declareGeneratedDialect<::mlir::func::FuncDialect>();

  // Finally, we register the additional transform operations with the dialect.
  registerTransformOps<
    // TODO: list the operation classes.
  >();
}
```

The operations themselves can be defined using ODS, exactly in the same way as regular operations in a dialect.

```tablegen
// In MyExtension.td
#ifndef MY_EXTENSION
#define MY_EXTENSION

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MyOp : Op<Transform_Dialect, "transform.my.op", [
    // TODO: interfaces and traits here.
   ]> {
  let summary = "my transform op";
  // TODO: define the operation properties.
}

#endif // MY_EXTENSION
```

Similarly to dialects, we must use Tablegen to generate the header and implementation of these operations. We can instruct CMake to do it as follows.


```sh
# In CMakeLists.txt next to MyExtension.td.

# Tell Tablegen to use MyExtension.td as input.
set(LLVM_TARGET_DEFINITIONS MyExtension.td)

# Ask Tablegen to generate op declarations and definitions from ODS.
mlir_tablegen(MyExtension.h.inc -gen-op-decls)
mlir_tablegen(MyExtension.cpp.inc -gen-op-defs)

# Add a CMakeTarget we can depend on to ensure the generation happens before the compilation.
add_public_tablegen_target(MyExtensionIncGen)

# Don't forget to generate the documentation, this will produce a MyExtension.md under 
# Dialects.
add_mlir_doc(MyExtension MyExtension Dialects/ -gen-op-doc)
```

```sh
# In CMakeLists.txt next to MyExtension.cpp
add_mlir_library(
  # Library called MyExtension.
  MyExtension

  # Built from the following source files.
  MyExtension.cpp

  # Make sure ODS declaration and definitions are generated before compiling this.
  DEPENDS
  MyExtensionIncGen

  # Link in the transform dialect, and all generated dialects.
  LINK_LIBS PUBLIC
  MLIRTransformDialect
  MLIRFuncDialect
  MLIRSCFDialect
)
```

This will generate two files, `MyExtension.h.inc` and `MyExtension.cpp.inc`, that are supposed to be included into the declaration and definition of the transform operations, respectively.

```c++
// In MyExtension.h.
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"

#define GET_OP_CLASSES
#include "MyExtension.h.inc"
```

```c++
// In MyExtension.cpp.

#define GET_OP_CLASSES
#include "MyExtension.cpp.inc"

// …
void MyExtension::init() {
  // …

  // Finally, we register the additional transform operations with the dialect. List all 
  // operations generated from ODS. This call will perform additional checks that the 
  // operations implement the transform and memory effect interfaces required by the 
  // dialect interpreter and assert if they do not.
  registerTransformOps<
#define GET_OP_LIST
#include "MyExtension.cpp.inc"
  >();
}
```

## Defining a Transform Operation

With this setup, we are now ready to define the new transform operation to rewrite the function call. This is identical to defining a regular operation in a dialect. Note that the Transform dialect requires operations to implement the `TransformOpInterface` as well as `MemoryEffectsOpInterface` to indicate whether the operands are consumed or only read. Our operation can be defined along the following lines.

```tablegen
// In MyExtension.td.

// Define the new operation. By convention, prefix its name with the name of the dialect 
// extension, "my.". The full operation name will be further prefixed with "transform.".
def ChangeCallTargetOp : Op<Transform_Dialect, "my.change_call_target",
    // Indicate that the operation implements the required TransformOpInterface and
    // MemoryEffectsOpInterface.
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  // Provide a brief and a full description. It is recommended that the latter describes 
  // the effects on the operands and how the operation processes various failure modes.
  let summary = "Changes the callee of a call operation to the specified one";
  let description = [{
    For each `func.call` payload operation associated with the handle, changes its 
    callee to be the symbol whose name is provided as an attribute to this operation.

    Generates a silenceable failure if the operand is associated with payload operations 
    that are not `func.call`.
    Only reads the operand.
  }];

  // The arguments include the handle to the payload operations and the attribute that 
  // specifies the new callee. The handle must implement TransformHandleTypeInterface.   
  // We use a string attribute as the symbol may not exist in the transform IR so the 
  // verification may fail. 
  let arguments = (ins
    TransformHandleTypeInterface:$call,
    StrAttr:$new_target);

  // The results are empty as the transformation does not produce any new payload.
  let results = (outs);

  // Provide nice syntax.
  let assemblyFormat = "$call `,` $new_target attr-dict `:` type($call)";
}
```

To finalize the definition of the transform operation, we need to implement the
interface methods. The `TransformOpInterface` currently requires only one method
– `apply` – that performs the actual transformation. It is a good practice to
limit the body of the method to manipulation of the Transform dialect constructs
and have the actual transformation implemented as a standalone function so it
can be used from other places in the code. Similar to rewrite patterns, all IR
must be modified with the provided rewriter.

```c++
// In MyExtension.cpp

// Implementation of our transform dialect operation.
// This operation returns a tri-state result that can be one of:
// - success when the transformation succeeded;
// - definite failure when the transformation failed in such a way that
//   following transformations are impossible or undesirable, typically it could
//   have left payload IR in an invalid state; it is expected that a diagnostic
//   is emitted immediately before returning the definite error;
// - silenceable failure when the transformation failed but following
//   transformations are still applicable, typically this means a precondition
//   for the transformation is not satisfied and the payload IR has not been
//   modified. The silenceable failure additionally carries a Diagnostic that
//   can be emitted to the user.
::mlir::DiagnosedSilenceableFailure mlir::transform::ChangeCallTargetOp::apply(
    // The rewriter that should be used when modifying IR.
    ::mlir::transform::TransformRewriter &rewriter,
    // The list of payload IR entities that will be associated with the
    // transform IR values defined by this transform operation. In this case, it
    // can remain empty as there are no results.
    ::mlir::transform::TransformResults &results,
    // The transform application state. This object can be used to query the
    // current associations between transform IR values and payload IR entities.
    // It can also carry additional user-defined state.
    ::mlir::transform::TransformState &state) {

  // First, we need to obtain the list of payload operations that are associated with
  // the operand handle.
  auto payload = state.getPayloadOps(getCall());

  // Then, we iterate over the list of operands and call the actual IR-mutating
  // function. We also check the preconditions here.
  for (Operation *payloadOp : payload) {
    auto call = dyn_cast<::mlir::func::CallOp>(payloadOp);
    if (!call) {
      DiagnosedSilenceableFailure diag = emitSilenceableError()
          << "only applies to func.call payloads";
      diag.attachNote(payloadOp->getLoc()) << "offending payload";
      return diag;
    }

    updateCallee(call, getNewTarget());
  }

  // If everything went well, return success.
  return DiagnosedSilenceableFailure::success();
}
```

The implementation of the `MemoryEffectsOpInterface` must specify the effects this operation has on its operands (consumed or readonly) and on the payload IR (mutates or readonly). Transform dialect verifiers will check for side effects being present and assert in debug builds if they are not.

```c++
// In MyExtension.cpp

void ChangeCallTargetOp::getEffects(
    ::llvm::SmallVectorImpl<::mlir::MemoryEffects::EffectInstance> &effects) {
  // Indicate that the `call` handle is only read by this operation because the
  // associated operation is not erased but rather modified in-place, so the
  // reference to it remains valid.
  onlyReadsHandle(getCall(), effects);

  // Indicate that the payload is modified by this operation.
  modifiesPayload(effects);
}
```

## Registration and Usage

This is enough to define transform operations. The only remaining bit is providing the extension registration hook that can be called from the project’s `main`.


```c++
// In TransformDialect.cpp (don't forget a declaration in TransformDialect.h);

void registerMyExtension(::mlir::DialectRegistry &registry) {
  registry.addExtensions<MyExtension>();
}
```

After registering the extension, it becomes possible to use our new operation in the transform dialect interpreter. The upstream testing pass can be used as is.

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op,
     %arg1: !transform.op<"linalg.matmul">,
     %arg2: !transform.op<"linalg.elemwise_binary">):
  // Since the %arg2 handle is associated with both elementwise operations,
  // we need to split it into two handles so we can target only the second
  // elementwise operation.
  %add, %max = transform.split_handle %arg2 : (!transform.op<"linalg.elemwise_binary">)
      -> (!transform.any_op, !transform.any_op)

  // The actual tiling transformation takes tile sizes as attributes. It produces a
  // handle to the loop generated during tiling.
  %loop, %tiled = transform.structured.tile_using_forall %max tile_sizes [8, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // We can now fuse the other operations into the loop. Here, we fuse
  // operations one-by-one. This requires the operation that is being fused
  // to define the value used within the loop, so the order of such fusions
  // is important. We could also use "transform.merge_handles" to obtain
  // a single handle to all operations and give it to `fuse_into_containing_op`
  // that would take care of the ordering in this case.
  %add_fused = transform.structured.fuse_into_containing_op %add into %loop
      : (!transform.any_op, !transform.any_op) -> !transform.any_op
  %matmul_fused = transform.structured.fuse_into_containing_op %arg1 into %loop
      : (!transform.op<"linalg.matmul">, !transform.any_op) -> !transform.any_op

  // Tile again to get the desired size. Note that this time this tiles the
  // "add" operation and fuses matmul into the loop, but doesn't affect the
  // "max" operation. This illustrates the precise targeting with the transform
  // dialect. Otherwise, it is difficult to differentiate "add" and "max", both
  // of which having the same kind.
  %loop_2, %tiled_2 = transform.structured.tile_using_forall %add_fused tile_sizes [4, 4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %matmul_fused_2 = transform.structured.fuse_into_containing_op %matmul_fused into %loop_2
      : (!transform.any_op, !transform.any_op) -> !transform.any_op

  // Since outlining is currently only implemented for region-holding operations
  // such as loops, use tiling to size 1 to materialize the outer loop that is
  // going to be outlined.
  %outline_target, %_ = transform.structured.tile_using_forall %tiled_2 tile_sizes [1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.structured.fuse_into_containing_op %matmul_fused_2 into %outline_target
      : (!transform.any_op, !transform.any_op) -> !transform.any_op
  %func, %call = transform.loop.outline %outline_target {func_name = "outlined"}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Rewrite the call target.
  transform.my.change_call_target %call, "microkernel" : !transform.any_op

  transform.yield
}
```

## Appendix: Autogenerated Documentation

[include "Tutorials/transform/MyExtensionCh2.md"]
