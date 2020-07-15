# Interfaces

MLIR is a generic and extensible framework, representing different
dialects with their own operations, attributes, types, and so on.
MLIR Dialects can express operations with a wide variety of semantics
and different levels of abstraction. The downside to this is that MLIR
transformations and analyses need to account for the semantics of
every operation, or handle operations conservatively. Without care,
this can result in code with special-cases for each supported
operation type. To combat this, MLIR provides the concept of
`interfaces`.

## Motivation

Interfaces provide a generic way of interacting with the IR. The goal is to be
able to express transformations/analyses in terms of these interfaces without
encoding specific knowledge about the exact operation or dialect involved. This
makes the compiler more extensible by allowing the addition of new dialects and
operations in a decoupled way with respect to the implementation of
transformations/analyses.

### Dialect Interfaces

Dialect interfaces are generally useful for transformation passes or analyses
that want to operate generically on a set of attributes/operations/types, which
might even be defined in different dialects. These interfaces generally involve
wide coverage over the entire dialect and are only used for a handful of
transformations/analyses. In these cases, registering the interface directly on
each operation is overly complex and cumbersome. The interface is not core to
the operation, just to the specific transformation. An example of where this
type of interface would be used is inlining. Inlining generally queries
high-level information about the operations within a dialect, like legality and
cost modeling, that often is not specific to one operation.

A dialect interface can be defined by inheriting from the CRTP base class
`DialectInterfaceBase::Base`. This class provides the necessary utilities for
registering an interface with the dialect so that it can be looked up later.
Once the interface has been defined, dialects can override it using
dialect-specific information. The interfaces defined by a dialect are registered
in a similar mechanism to Attributes, Operations, Types, etc.

```c++
/// Define an Inlining interface to allow for dialects to opt-in.
class DialectInlinerInterface :
    public DialectInterface::Base<DialectInlinerInterface> {
public:
  /// Returns true if the given region 'src' can be inlined into the region
  /// 'dest' that is attached to an operation registered to the current dialect.
  /// 'valueMapping' contains any remapped values from within the 'src' region.
  /// This can be used to examine what values will replace entry arguments into
  /// the 'src' region, for example.
  virtual bool isLegalToInline(Region *dest, Region *src,
                               BlockAndValueMapping &valueMapping) const {
    return false;
  }
};

/// Override the inliner interface to add support for inlining affine
/// operations.
struct AffineInlinerInterface : public DialectInlinerInterface {
  /// Affine structures have specific inlining constraints.
  bool isLegalToInline(Region *dest, Region *src,
                       BlockAndValueMapping &valueMapping) const final {
    ...
  }
};

/// Register the interface with the dialect.
AffineDialect::AffineDialect(MLIRContext *context) ... {
  addInterfaces<AffineInlinerInterface>();
}
```

Once registered, these interfaces can be queried from the dialect by
the transformation/analysis that wants to use them, without
determining the particular dialect subclass:

```c++
Dialect *dialect = ...;
if (auto *interface = dialect->getInterface<DialectInlinerInterface>())
    ... // The dialect provides this interface.
```

#### DialectInterfaceCollections

An additional utility is provided via DialectInterfaceCollection. This CRTP
class allows for collecting all of the dialects that have registered a given
interface within the context.

```c++
class InlinerInterface : public
    DialectInterfaceCollection<DialectInlinerInterface> {
  /// The hooks for this class mirror the hooks for the DialectInlinerInterface,
  /// with default implementations that call the hook on the interface for a
  /// given dialect.
  virtual bool isLegalToInline(Region *dest, Region *src,
                               BlockAndValueMapping &valueMapping) const {
    auto *handler = getInterfaceFor(dest->getContainingOp());
    return handler ? handler->isLegalToInline(dest, src, valueMapping) : false;
  }
};

MLIRContext *ctx = ...;
InlinerInterface interface(ctx);
if(!interface.isLegalToInline(...))
   ...
```

### Attribute/Operation/Type Interfaces

Attribute/Operation/Type interfaces, as the names suggest, are those registered
at the level of a specific attribute/operation/type. These interfaces provide
access to derived objects by providing a virtual interface that must be
implemented. As an example, the `Linalg` dialect may implement an interface that
provides general queries about some of the dialects library operations. These
queries may provide things like: the number of parallel loops; the number of
inputs and outputs; etc.

These interfaces are defined by overriding the CRTP base class `AttrInterface`,
`OpInterface`, or `TypeInterface` respectively. These classes take, as a
template parameter, a `Traits` class that defines a `Concept` and a `Model`
class. These classes provide an implementation of concept-based polymorphism,
where the Concept defines a set of virtual methods that are overridden by the
Model that is templated on the concrete object type. It is important to note
that these classes should be pure in that they contain no non-static data
members. Objects that wish to override this interface should add the provided
trait `*Interface<..>::Trait` to the trait list upon registration.

```c++
struct ExampleOpInterfaceTraits {
  /// Define a base concept class that defines the virtual interface that needs
  /// to be overridden.
  struct Concept {
    virtual ~Concept();
    virtual unsigned getNumInputs(Operation *op) const = 0;
  };

  /// Define a model class that specializes a concept on a given operation type.
  template <typename OpT>
  struct Model : public Concept {
    /// Override the method to dispatch on the concrete operation.
    unsigned getNumInputs(Operation *op) const final {
      return llvm::cast<OpT>(op).getNumInputs();
    }
  };
};

class ExampleOpInterface : public OpInterface<ExampleOpInterface,
                                              ExampleOpInterfaceTraits> {
public:
  /// Use base class constructor to support LLVM-style casts.
  using OpInterface<ExampleOpInterface, ExampleOpInterfaceTraits>::OpInterface;

  /// The interface dispatches to 'getImpl()', an instance of the concept.
  unsigned getNumInputs() const {
    return getImpl()->getNumInputs(getOperation());
  }
};

```

Once the interface has been defined, it is registered to an operation by adding
the provided trait `ExampleOpInterface::Trait`. Using this interface is just
like using any other derived operation type, i.e. casting:

```c++
/// When defining the operation, the interface is registered via the nested
/// 'Trait' class provided by the 'OpInterface<>' base class.
class MyOp : public Op<MyOp, ExampleOpInterface::Trait> {
public:
  /// The definition of the interface method on the derived operation.
  unsigned getNumInputs() { return ...; }
};

/// Later, we can query if a specific operation(like 'MyOp') overrides the given
/// interface.
Operation *op = ...;
if (ExampleOpInterface example = dyn_cast<ExampleOpInterface>(op))
  llvm::errs() << "num inputs = " << example.getNumInputs() << "\n";
```

#### Utilizing the ODS Framework

Operation interfaces require a bit of boiler plate to connect all of the pieces
together. The ODS(Operation Definition Specification) framework provides
simplified mechanisms for [defining interfaces](OpDefinitions.md#interfaces).

As an example, using the ODS framework would allow for defining the example
interface above as:

```tablegen
def ExampleOpInterface : OpInterface<"ExampleOpInterface"> {
  let description = [{
    This is an example interface definition.
  }];

  let methods = [
    InterfaceMethod<
      "Get the number of inputs for the current operation.",
      "unsigned", "getNumInputs"
    >,
  ];
}
```

#### Operation Interface List

MLIR includes standard interfaces providing functionality that is
likely to be common across many different operations. Below is a list
of some key interfaces that may be used directly by any dialect. The
format of the header for each interface section goes as follows:

*   `Interface class name`
    -   (`C++ class` -- `ODS class`(if applicable))

##### CallInterfaces

*   `CallOpInterface` - Used to represent operations like 'call'
    -   `CallInterfaceCallable getCallableForCallee()`
*   `CallableOpInterface` - Used to represent the target callee of call.
    -   `Region * getCallableRegion()`
    -   `ArrayRef<Type> getCallableResults()`

##### RegionKindInterfaces

*   `RegionKindInterface` - Used to describe the abstract semantics of regions.
     - `RegionKind getRegionKind(unsigned index)` - Return the kind of the region with the given index inside this operation.
         - RegionKind::Graph - represents a graph region without control flow semantics
         - RegionKind::SSACFG - represents an [SSA-style control flow](LangRef.md#modeling-control-flow) region with basic blocks and reachability
     - `hasSSADominance(unsigned index)` - Return true if the region with the given index inside this operation requires dominance.
