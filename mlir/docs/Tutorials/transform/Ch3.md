# Chapter 3: More than Simple Transform Operations

## Type Constraints and ApplyEach Trait

A transform operation that applies to each payload operation individually and requires it to be of a specific kind is a repeated pattern. One can use Transform dialect types to specify the preconditions of the type. Specifically, we can change the expected operand type from the wide `TransformHandleTypeInterface` to the more narrow `Transform_ConcreteOp<"func.call">`. Furthermore, we use the `TransformEachOpTrait` trait to provide the skeleton implementation of the `apply` method that performs verification, iteration over payloads and result concatenation. The improved ODS op definition is as follows.

```tablegen
// In MyExtension.td.

// Define the new operation. By convention, prefix its name with the name of the dialect extension, "my.". The full operation name will be further prefixed with "transform.".
def ChangeCallTargetOp : Op<Transform_Dialect, "my.change_call_target",
    // Indicate that the operation implements the required TransformOpInterface and
    // MemoryEffectsOpInterface. Use the TransformEach trait to provide the
    // implementation for TransformOpInterface.
    [TransformOpInterface, TransformEachOpTrait,
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
    Transform_ConcreteOpType<"func.call">:$call,
    StrAttr:$new_target);

  // The results are empty as the transformation does not produce any new payload.
  let results = (outs);

  // Provide nice syntax.
  let assemblyFormat = "$call `,` $new_target attr-dict `:` type($call)";

  // Declare the function implementing the interface for a single payload operation.
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::func::CallOp call,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}
```

Now, instead of defining the `apply` method with a loop, we can simply define a function that applies to an individual payload operation and the trait will take care of the rest.

```c++
::mlir::DiagnosedSilenceableFailure ChangeCallTargetOp::applyToOne(
    ::mlir::transform::TransformRewriter &rewriter,
    ::mlir::func::CallOp call,
    ::mlir::transform::ApplyToEachResultList &results,
    ::mlir::transform::TransformState &state) {
  // Call the actual transformation function.
  updateCallee(call, getNewTarget());
  // Indicate success.
  return DiagnosedSilenceableFailure::success();
}
```

## Defining a Transform Type

In addition to operations, the Transform dialect allows extensions to define and inject additional attributes and types. As we have seen above, transform types are used to specify constraints on the payload operations. Our call rewriting operation currently applies only to `func.call`. We may want to generalize it to apply to any payload operation that implements `CallOpInterface`, but the Transform dialect currently doesn’t provide a type that checks if a payload operation implements this interface. Let’s define it in our extension.

Type definition is again identical to defining a dialect type with ODS.

```tablegen
// Transform dialect allows additional types to be defined and injected.
def CallOpInterfaceHandle
  : TypeDef<Transform_Dialect, "CallOpInterfaceHandle",
      // The type must implement `TransformHandleTypeInterface`.
      [DeclareTypeInterfaceMethods<TransformHandleTypeInterface>]> {

  // The usual components of a type such as description, mnemonic and assembly format 
  // should be provided.
  let summary = "handle to payload operations implementing CallOpInterface";
  let mnemonic = "my.call_op_interface";
  let assemblyFormat = "";
}
```

We will omit the generation of declaration and definitions using Tablegen for brevity as it is identical to the regular case. 

To finalize the definition of a transform type, one must implement the interface methods.

```c++
// In MyExtension.cpp.

// The interface declares this method to verify constraints this type has on
// payload operations. It returns the now familiar tri-state result.
mlir::DiagnosedSilenceableFailure
mlir::transform::CallOpInterfaceHandleType::checkPayload(
    // Location at which diagnostics should be emitted.
    mlir::Location loc,
    // List of payload operations that are about to be associated with the
    // handle that has this type.
    llvm::ArrayRef<mlir::Operation *> payload) const {

  // All payload operations are expected to implement CallOpInterface, check this.
  for (Operation *op : payload) {
    if (llvm::isa<mlir::CallOpInterface>(op))
      continue;

    // By convention, these verifiers always emit a silenceable failure since they are 
    // checking a precondition.
    DiagnosedSilenceableFailure diag = emitSilenceableError(loc) 
        << "expected the payload operation to implement CallOpInterface";
    diag.attachNote(op->getLoc()) << "offending operation";
    return diag;
  }

  // If everything is okay, return success.
  return DiagnosedSilenceableFailure::success();
}

```

Additional attributes and types need to be registered in the extension, next to operations.

```c++
// In MyExtension.cpp.

void MyExtension::init() {
  // …
  
  registerTypes<
#define GET_TYPEDEF_LIST
#include "MyExtensionTypes.cpp.inc"
  >();
}
```

This type is now directly available in the transform dialect and can be used in operations.


```mlir
  // Cast to our new type.
  %casted = transform.cast %call : !transform.any_op to !transform.my.call_op_interface
  // Using our new operation.
  transform.my.change_call_target %casted, "microkernel" : !transform.my.call_op_interface
```

## Operand Consumption

As an exercise, let us modify the rewriting operation to consume the operand. This would be necessary, for example, if the transformation were to rewrite the `func.call` operation into a custom operation `my.mm4`. Since the operand handle is now consumed, the operation can return a new handle to the newly produced payload operation. Otherwise, the ODS definition of the transform operation remains unchanged.

```tablegen
// In MyExtension.td.

// Define another transform operation.
def CallToOp : Op<Transform_Dialect, "my.call_to_op",
     // Indicate that the operation implements the required TransformOpInterface and
     // MemoryEffectsOpInterface. Use the TransformEach trait to provide the
     // implementation for TransformOpInterface.
    [TransformOpInterface, TransformEachOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  // Summary and description omitted for brevity.

  // The argument is the handle to the payload operations.
  let arguments = (ins CallOpInterfaceHandle:$call);

  // The result is the handle to the payload operations produced during the
  // transformation.
  let results = (outs TransformHandleTypeInterface:$transformed);

  // Provide nice syntax.
  let assemblyFormat = "$call attr-dict `:` functional-type(inputs, outputs)";

  // Declare the function implementing the interface for a single payload operation.
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::CallOpInterface call,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}
```

Now let’s look at the implementation of interface methods.

```c++
// In MyExtension.cpp.

::mlir::DiagnosedSilenceableFailure CallToOp::applyToOne(
    ::mlir::transform::TransformRewriter &rewriter,
    ::mlir::CallOpInterface call,
    ::mlir::transform::ApplyToEachResultList &results,
    ::mlir::transform::TransformState &state) {
  // Call the actual rewrite.
  Operation *rewritten = rewriteToOp(call);

  // Report an error if the rewriter produced a null pointer. Note that it may have
  // irreversibly modified the payload IR, so we produce a definite failure.
  if (!rewritten) {
    return emitDefiniteError() << "failed to rewrite call to operation";
  }

  // On success, push the resulting operation into the result list. The list is expected
  // to contain exactly one entity per result and per application. The handles will be
  // associated with lists of the respective values produced by each application.
  results.push_back(rewritten);

  // If everything is fine, return success.
  return DiagnosedSilenceableFailure::success();
}

void CallToOp::getEffects(
    ::llvm::SmallVectorImpl<::mlir::MemoryEffects::EffectInstance> &effects) {
  // Indicate using side effects that the operand handle is consumed, and the
  // result handle is produced.
  consumesHandle(getCall(), effects);
  producesHandle(getTransformed(), effects);

  // Indicate that the payload IR is modified.
  modifiesPayload(effects);
}
```

The overall flow of these implementations is similar to the previous one. The application also needs to specify the resulting entities that are going to be associated with the handles it produces. Operations are required to specify the entities to associate with _all_ results on success, even if the list is empty. An assertion will be triggered if it is not the case. In case of failure, the interpreter will automatically associate all results that are not yet defined with empty lists.

Note that since `applyToOne` always expects one payload entity to be associated with each result handle in each application, it cannot be used to return handles associated with empty lists for non-empty operand handles. Instead, one would use `apply` directly.

```c++
::mlir::DiagnosedSilenceableFailure SomeOtherOp::apply(
    ::mlir::transform::TransformRewriter &rewriter,
    ::mlir::transform::TransformResults &results,
    ::mlir::transform::TransformState &state) {
  // ...

  // Associate the result `transformed` with an empty list of payload operations.
  results.set(cast<OpResult>(getTransformed()), {});
  return DiagnosedSilenceableFailure::success();
}
```

## Memory Effects Traits

Some common memory effect patterns are also available as traits to minimize the boilerplate.

*   `FunctionalStyleTransformOpTrait` indicates that all handle-typed operands are consumed, all results are produced, and the payload IR is modified.
*   `NavigationTransformOpTrait` indicates that all handle-typed operands are only read, all results are produced, and the payload IR is only read.

Using these traits removes the need to declare or define the methods of the `MemoryEffectsOpInterface`.

```tablegen
// In MyExtension.td.

// Define another transform operation.
def CallToOp : Op<Transform_Dialect, "my.call_to_op",
     // Indicate that the operation implements the required TransformOpInterface.
     // Use the TransformEach trait to provide implementation of this interface.
    [TransformOpInterface, TransformEachOpTrait,
     // Indicate that the operation implements the required MemoryEffectsOpInterface.
     // Use the FunctionalStyle trait to provide the implementation for this interface.
     MemoryEffectsOpInterface, FunctionalStyleTransformOpTrait]> {
  // Summary and description omitted for brevity.

  // The argument is the handle to the payload operations.
  let arguments = (ins CallOpInterfaceHandle:$call);

  // The result is the handle to the payload operations produced during the
  // transformation.
  let results = (outs TransformHandleTypeInterface:$transformed);

  // Provide nice syntax.
  let assemblyFormat = "$call attr-dict `:` functional-type(operands, results)";

  // Declare the function implementing the interface for a single payload operation.
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::CallOpInterface call,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}
```

## Appendix: Autogenerated Documentation

[include "Tutorials/transform/MyExtensionCh3.md"]

