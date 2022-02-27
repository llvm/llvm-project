# Traits

[TOC]

MLIR allows for a truly open ecosystem, as any dialect may define attributes,
operations, and types that suit a specific level of abstraction. `Traits` are a
mechanism which abstracts implementation details and properties that are common
across many different attributes/operations/types/etc.. `Traits` may be used to
specify special properties and constraints of the object, including whether an
operation has side effects or that its output has the same type as the input.
Some examples of operation traits are `Commutative`, `SingleResult`,
`Terminator`, etc. See the more comprehensive list of
[operation traits](#operation-traits-list) below for more examples of what is
possible.

## Defining a Trait

Traits may be defined in C++ by inheriting from the `TraitBase<ConcreteType,
TraitType>` class for the specific IR type. For attributes, this is
`AttributeTrait::TraitBase`. For operations, this is `OpTrait::TraitBase`. For
types, this is `TypeTrait::TraitBase`. This base class takes as template
parameters:

*   ConcreteType
    -   The concrete class type that this trait was attached to.
*   TraitType
    -   The type of the trait class that is being defined, for use with the
        [`Curiously Recurring Template Pattern`](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern).

A derived trait class is expected to take a single template that corresponds to
the `ConcreteType`. An example trait definition is shown below:

```c++
template <typename ConcreteType>
class MyTrait : public TraitBase<ConcreteType, MyTrait> {
};
```

Operation traits may also provide a `verifyTrait` or `verifyRegionTrait` hook
that is called when verifying the concrete operation. The difference between
these two is that whether the verifier needs to access the regions, if so, the
operations in the regions will be verified before the verification of this
trait. The [verification order](OpDefinitions.md/#verification-ordering)
determines when a verifier will be invoked.

```c++
template <typename ConcreteType>
class MyTrait : public OpTrait::TraitBase<ConcreteType, MyTrait> {
public:
  /// Override the 'verifyTrait' hook to add additional verification on the
  /// concrete operation.
  static LogicalResult verifyTrait(Operation *op) {
    // ...
  }
};
```

Note: It is generally good practice to define the implementation of the
`verifyTrait` or `verifyRegionTrait` hook out-of-line as a free function when
possible to avoid instantiating the implementation for every concrete operation
type.

Operation traits may also provide a `foldTrait` hook that is called when folding
the concrete operation. The trait folders will only be invoked if the concrete
operation fold is either not implemented, fails, or performs an in-place fold.

The following signature of fold will be called if it is implemented and the op
has a single result.

```c++
template <typename ConcreteType>
class MyTrait : public OpTrait::TraitBase<ConcreteType, MyTrait> {
public:
  /// Override the 'foldTrait' hook to support trait based folding on the
  /// concrete operation.
  static OpFoldResult foldTrait(Operation *op, ArrayRef<Attribute> operands) { {
    // ...
  }
};
```

Otherwise, if the operation has a single result and the above signature is not
implemented, or the operation has multiple results, then the following signature
will be used (if implemented):

```c++
template <typename ConcreteType>
class MyTrait : public OpTrait::TraitBase<ConcreteType, MyTrait> {
public:
  /// Override the 'foldTrait' hook to support trait based folding on the
  /// concrete operation.
  static LogicalResult foldTrait(Operation *op, ArrayRef<Attribute> operands,
                                 SmallVectorImpl<OpFoldResult> &results) { {
    // ...
  }
};
```

Note: It is generally good practice to define the implementation of the
`foldTrait` hook out-of-line as a free function when possible to avoid
instantiating the implementation for every concrete operation type.

### Parametric Traits

The above demonstrates the definition of a simple self-contained trait. It is
also often useful to provide some static parameters to the trait to control its
behavior. Given that the definition of the trait class is rigid, i.e. we must
have a single template argument for the concrete object, the templates for the
parameters will need to be split out. An example is shown below:

```c++
template <int Parameter>
class MyParametricTrait {
public:
  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType, Impl> {
    // Inside of 'Impl' we have full access to the template parameters
    // specified above.
  };
};
```

## Attaching a Trait

Traits may be used when defining a derived object type, by simply appending the
name of the trait class to the end of the base object class operation type:

```c++
/// Here we define 'MyAttr' along with the 'MyTrait' and `MyParametric trait
/// classes we defined previously.
class MyAttr : public Attribute::AttrBase<MyAttr, ..., MyTrait, MyParametricTrait<10>::Impl> {};
/// Here we define 'MyOp' along with the 'MyTrait' and `MyParametric trait
/// classes we defined previously.
class MyOp : public Op<MyOp, MyTrait, MyParametricTrait<10>::Impl> {};
/// Here we define 'MyType' along with the 'MyTrait' and `MyParametric trait
/// classes we defined previously.
class MyType : public Type::TypeBase<MyType, ..., MyTrait, MyParametricTrait<10>::Impl> {};
```

### Attaching Operation Traits in ODS

To use an operation trait in the [ODS](OpDefinitions.md) framework, we need to
provide a definition of the trait class. This can be done using the
`NativeOpTrait` and `ParamNativeOpTrait` classes. `ParamNativeOpTrait` provides
a mechanism in which to specify arguments to a parametric trait class with an
internal `Impl`.

```tablegen
// The argument is the c++ trait class name.
def MyTrait : NativeOpTrait<"MyTrait">;

// The first argument is the parent c++ class name. The second argument is a
// string containing the parameter list.
class MyParametricTrait<int prop>
  : NativeOpTrait<"MyParametricTrait", !cast<string>(!head(parameters))>;
```

These can then be used in the `traits` list of an op definition:

```tablegen
def OpWithInferTypeInterfaceOp : Op<...[MyTrait, MyParametricTrait<10>]> { ... }
```

See the documentation on [operation definitions](OpDefinitions.md) for more
details.

## Using a Trait

Traits may be used to provide additional methods, static fields, or other
information directly on the concrete object. `Traits` internally become `Base`
classes of the concrete operation, so all of these are directly accessible. To
expose this information opaquely to transformations and analyses,
[`interfaces`](Interfaces.md) may be used.

To query if a specific object contains a specific trait, the `hasTrait<>` method
may be used. This takes as a template parameter the trait class, which is the
same as the one passed when attaching the trait to an operation.

```c++
Operation *op = ..;
if (op->hasTrait<MyTrait>() || op->hasTrait<MyParametricTrait<10>::Impl>())
  ...;
```

## Operation Traits List

MLIR provides a suite of traits that provide various functionalities that are
common across many different operations. Below is a list of some key traits that
may be used directly by any dialect. The format of the header for each trait
section goes as follows:

*   `Header`
    -   (`C++ class` -- `ODS class`(if applicable))

### AffineScope

*   `OpTrait::AffineScope` -- `AffineScope`

This trait is carried by region holding operations that define a new scope for
the purposes of polyhedral optimization and the affine dialect in particular.
Any SSA values of 'index' type that either dominate such operations, or are
defined at the top-level of such operations, or appear as region arguments for
such operations automatically become valid symbols for the polyhedral scope
defined by that operation. As a result, such SSA values could be used as the
operands or index operands of various affine dialect operations like affine.for,
affine.load, and affine.store. The polyhedral scope defined by an operation with
this trait includes all operations in its region excluding operations that are
nested inside of other operations that themselves have this trait.

### AutomaticAllocationScope

*   `OpTrait::AutomaticAllocationScope` -- `AutomaticAllocationScope`

This trait is carried by region holding operations that define a new scope for
automatic allocation. Such allocations are automatically freed when control is
transferred back from the regions of such operations. As an example, allocations
performed by
[`memref.alloca`](Dialects/MemRef.md/#memrefalloca-mlirmemrefallocaop) are
automatically freed when control leaves the region of its closest surrounding op
that has the trait AutomaticAllocationScope.

### Broadcastable

*   `OpTrait::ResultsBroadcastableShape` -- `ResultsBroadcastableShape`

This trait adds the property that the operation is known to have
[broadcast-compatible](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
operands and its result types' shape is the broadcast compatible with the shape
of the broadcasted operands. Specifically, starting from the most varying
dimension, each dimension pair of the two operands' shapes should either be the
same or one of them is one. Also, the result shape should have the corresponding
dimension equal to the larger one, if known. Shapes are checked partially if
ranks or dimensions are not known. For example, an op with `tensor<?x2xf32>` and
`tensor<2xf32>` as operand types and `tensor<3x2xf32>` as the result type is
broadcast-compatible.

This trait requires that the operands are either vector or tensor types.

### Commutative

*   `OpTrait::IsCommutative` -- `Commutative`

This trait adds the property that the operation is commutative, i.e. `X op Y ==
Y op X`

### ElementwiseMappable

*   `OpTrait::ElementwiseMappable` -- `ElementwiseMappable`

This trait tags scalar ops that also can be applied to vectors/tensors, with
their semantics on vectors/tensors being elementwise application. This trait
establishes a set of properties that allow reasoning about / converting between
scalar/vector/tensor code. These same properties allow blanket implementations
of various analyses/transformations for all `ElementwiseMappable` ops.

Note: Not all ops that are "elementwise" in some abstract sense satisfy this
trait. In particular, broadcasting behavior is not allowed. See the comments on
`OpTrait::ElementwiseMappable` for the precise requirements.

### HasParent

*   `OpTrait::HasParent<typename ParentOpType>` -- `HasParent<string op>` or
    `ParentOneOf<list<string> opList>`

This trait provides APIs and verifiers for operations that can only be nested
within regions that are attached to operations of `ParentOpType`.

### IsolatedFromAbove

*   `OpTrait::IsIsolatedFromAbove` -- `IsolatedFromAbove`

This trait signals that the regions of an operations are known to be isolated
from above. This trait asserts that the regions of an operation will not
capture, or reference, SSA values defined above the region scope. This means
that the following is invalid if `foo.region_op` is defined as
`IsolatedFromAbove`:

```mlir
%result = arith.constant 10 : i32
foo.region_op {
  foo.yield %result : i32
}
```

This trait is an important structural property of the IR, and enables operations
to have [passes](PassManagement.md) scheduled under them.

### MemRefsNormalizable

*   `OpTrait::MemRefsNormalizable` -- `MemRefsNormalizable`

This trait is used to flag operations that consume or produce values of `MemRef`
type where those references can be 'normalized'. In cases where an associated
`MemRef` has a non-identity memory-layout specification, such normalizable
operations can be modified so that the `MemRef` has an identity layout
specification. This can be implemented by associating the operation with its own
index expression that can express the equivalent of the memory-layout
specification of the MemRef type. See [the -normalize-memrefs pass].
(https://mlir.llvm.org/docs/Passes/#-normalize-memrefs-normalize-memrefs)

### Single Block Region

*   `OpTrait::SingleBlock` -- `SingleBlock`

This trait provides APIs and verifiers for operations with regions that have a
single block.

### Single Block with Implicit Terminator

*   `OpTrait::SingleBlockImplicitTerminator<typename TerminatorOpType>` --
    `SingleBlockImplicitTerminator<string op>`

This trait implies the `SingleBlock` above, but adds the additional requirement
that the single block must terminate with `TerminatorOpType`.

### SymbolTable

*   `OpTrait::SymbolTable` -- `SymbolTable`

This trait is used for operations that define a
[`SymbolTable`](SymbolsAndSymbolTables.md#symbol-table).

### Terminator

*   `OpTrait::IsTerminator` -- `Terminator`

This trait provides verification and functionality for operations that are known
to be [terminators](LangRef.md#terminator-operations).

*   `OpTrait::NoTerminator` -- `NoTerminator`

This trait removes the requirement on regions held by an operation to have
[terminator operations](LangRef.md#terminator-operations) at the end of a block.
This requires that these regions have a single block. An example of operation
using this trait is the top-level `ModuleOp`.
