# Interfaces

MLIR is a generic and extensible framework, representing different dialects with
their own attributes, operations, types, and so on. MLIR Dialects can express
operations with a wide variety of semantics and different levels of abstraction.
The downside to this is that MLIR transformations and analyses need to be able
to account for the semantics of every operation, or be overly conservative.
Without care, this can result in code with special-cases for each supported
operation type. To combat this, MLIR provides a concept of `interfaces`.

## Motivation

Interfaces provide a generic way of interacting with the IR. The goal is to be
able to express transformations/analyses in terms of these interfaces without
encoding specific knowledge about the exact operation or dialect involved. This
makes the compiler more easily extensible by allowing the addition of new
dialects and operations in a decoupled way with respect to the implementation of
transformations/analyses.

### Dialect Interfaces

Dialect interfaces are generally useful for transformation passes or analyses
that want to operate generically on a set of attributes/operations/types, which
may be defined in different dialects. These interfaces generally involve wide
coverage over an entire dialect and are only used for a handful of analyses or
transformations. In these cases, registering the interface directly on each
operation is overly complex and cumbersome. The interface is not core to the
operation, just to the specific transformation. An example of where this type of
interface would be used is inlining. Inlining generally queries high-level
information about the operations within a dialect, like cost modeling and
legality, that often is not specific to one operation.

A dialect interface can be defined by inheriting from the
[CRTP](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern) base
class `DialectInterfaceBase::Base<>`. This class provides the necessary
utilities for registering an interface with a dialect so that it can be
referenced later. Once the interface has been defined, dialects can override it
using dialect-specific information. The interfaces defined by a dialect are
registered via `addInterfaces<>`, a similar mechanism to Attributes, Operations,
Types, etc

```c++
/// Define a base inlining interface class to allow for dialects to opt-in to
/// the inliner.
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

/// Override the inliner interface to add support for the AffineDialect to
/// enable inlining affine operations.
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

Once registered, these interfaces can be queried from the dialect by an analysis
or transformation without the need to determine the specific dialect subclass:

```c++
Dialect *dialect = ...;
if (DialectInlinerInterface *interface = dyn_cast<DialectInlinerInterface>(dialect)) {
  // The dialect has provided an implementation of this interface.
  ...
}
```

#### DialectInterfaceCollection

An additional utility is provided via `DialectInterfaceCollection`. This class
allows for collecting all of the dialects that have registered a given interface
within an instance of the `MLIRContext`. This can be useful to hide and optimize
the lookup of a registered dialect interface.

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
implemented. As an example, many analyses and transformations want to reason
about the side effects of an operation to improve performance and correctness.
The side effects of an operation are generally tied to the semantics of a
specific operation, for example an `affine.load` operation has a `read` effect
(as the name may suggest).

These interfaces are defined by overriding the
[CRTP](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern) class
for the specific IR entity; `AttrInterface`, `OpInterface`, or `TypeInterface`
respectively. These classes take, as a template parameter, a `Traits` class that
defines a `Concept` and a `Model` class. These classes provide an implementation
of concept-based polymorphism, where the `Concept` defines a set of virtual
methods that are overridden by the `Model` that is templated on the concrete
entity type. It is important to note that these classes should be pure, and
should not contain non-static data members or other mutable data. To attach an
interface to an object, the base interface classes provide a
[`Trait`](Traits.md) class that can be appended to the trait list of that
object.

```c++
struct ExampleOpInterfaceTraits {
  /// Define a base concept class that specifies the virtual interface to be
  /// implemented.
  struct Concept {
    virtual ~Concept();

    /// This is an example of a non-static hook to an operation.
    virtual unsigned exampleInterfaceHook(Operation *op) const = 0;

    /// This is an example of a static hook to an operation. A static hook does
    /// not require a concrete instance of the operation. The implementation is
    /// a virtual hook, the same as the non-static case, because the
    /// implementation of the hook itself still requires indirection.
    virtual unsigned exampleStaticInterfaceHook() const = 0;
  };

  /// Define a model class that specializes a concept on a given operation type.
  template <typename ConcreteOp>
  struct Model : public Concept {
    /// Override the method to dispatch on the concrete operation.
    unsigned exampleInterfaceHook(Operation *op) const final {
      return llvm::cast<ConcreteOp>(op).exampleInterfaceHook();
    }

    /// Override the static method to dispatch to the concrete operation type.
    unsigned exampleStaticInterfaceHook() const final {
      return ConcreteOp::exampleStaticInterfaceHook();
    }
  };
};

/// Define the main interface class that analyses and transformations will
/// interface with.
class ExampleOpInterface : public OpInterface<ExampleOpInterface,
                                              ExampleOpInterfaceTraits> {
public:
  /// Inherit the base class constructor to support LLVM-style casting.
  using OpInterface<ExampleOpInterface, ExampleOpInterfaceTraits>::OpInterface;

  /// The interface dispatches to 'getImpl()', a method provided by the base
  /// `OpInterface` class that returns an instance of the concept.
  unsigned exampleInterfaceHook() const {
    return getImpl()->exampleInterfaceHook(getOperation());
  }
  unsigned exampleStaticInterfaceHook() const {
    return getImpl()->exampleStaticInterfaceHook(getOperation()->getName());
  }
};

```

Once the interface has been defined, it is registered to an operation by adding
the provided trait `ExampleOpInterface::Trait` as described earlier. Using this
interface is just like using any other derived operation type, i.e. casting:

```c++
/// When defining the operation, the interface is registered via the nested
/// 'Trait' class provided by the 'OpInterface<>' base class.
class MyOp : public Op<MyOp, ExampleOpInterface::Trait> {
public:
  /// The definition of the interface method on the derived operation.
  unsigned exampleInterfaceHook() { return ...; }
  static unsigned exampleStaticInterfaceHook() { return ...; }
};

/// Later, we can query if a specific operation(like 'MyOp') overrides the given
/// interface.
Operation *op = ...;
if (ExampleOpInterface example = dyn_cast<ExampleOpInterface>(op))
  llvm::errs() << "hook returned = " << example.exampleInterfaceHook() << "\n";
```

#### External Models for Attribute, Operation and Type Interfaces

It may be desirable to provide an interface implementation for an IR object
without modifying the definition of said object. Notably, this allows to
implement interfaces for attributes, operations and types outside of the dialect
that defines them, for example, to provide interfaces for built-in types.

This is achieved by extending the concept-based polymorphism model with two more
classes derived from `Concept` as follows.

```c++
struct ExampleTypeInterfaceTraits {
  struct Concept {
    virtual unsigned exampleInterfaceHook(Type type) const = 0;
    virtual unsigned exampleStaticInterfaceHook() const = 0;
  };

  template <typename ConcreteType>
  struct Model : public Concept { /*...*/ };

  /// Unlike `Model`, `FallbackModel` passes the type object through to the
  /// hook, making it accessible in the method body even if the method is not
  /// defined in the class itself and thus has no `this` access. ODS
  /// automatically generates this class for all interfaces.
  template <typename ConcreteType>
  struct FallbackModel : public Concept {
    unsigned exampleInterfaceHook(Type type) const override {
      getImpl()->exampleInterfaceHook(type);
    }
    unsigned exampleStaticInterfaceHook() const override {
      ConcreteType::exampleStaticInterfaceHook();
    }
  };

  /// `ExternalModel` provides a place for default implementations of interface
  /// methods by explicitly separating the model class, which implements the
  /// interface, from the type class, for which the interface is being
  /// implemented. Default implementations can be then defined generically
  /// making use of `cast<ConcreteType>`. If `ConcreteType` does not provide
  /// the APIs required by the default implementation, custom implementations
  /// may use `FallbackModel` directly to override the default implementation.
  /// Being located in a class template, it never gets instantiated and does not
  /// lead to compilation errors. ODS automatically generates this class and
  /// places default method implementations in it.
  template <typename ConcreteModel, typename ConcreteType>
  struct ExternalModel : public FallbackModel<ConcreteModel> {
    unsigned exampleInterfaceHook(Type type) const override {
      // Default implementation can be provided here.
      return type.cast<ConcreteType>().callSomeTypeSpecificMethod();
    }
  };
};
```

External models can be provided for attribute, operation and type interfaces by
deriving either `FallbackModel` or `ExternalModel` and by registering the model
class with the relevant class in a given context. Other contexts will not see
the interface unless registered.

```c++
/// External interface implementation for a concrete class. This does not
/// require modifying the definition of the type class itself.
struct ExternalModelExample
    : public ExampleTypeInterface::ExternalModel<ExternalModelExample,
                                                 IntegerType> {
  static unsigned exampleStaticInterfaceHook() {
    // Implementation is provided here.
    return IntegerType::someStaticMethod();
  }

  // No need to define `exampleInterfaceHook` that has a default implementation
  // in `ExternalModel`. But it can be overridden if desired.
}

int main() {
  MLIRContext context;
  /* ... */;

  // Attach the interface model to the type in the given context before
  // using it. The dialect containing the type is expected to have been loaded
  // at this point.
  IntegerType::attachInterface<ExternalModelExample>(context);
}
```

Note: It is strongly encouraged to only use this mechanism if you "own" the
interface being externally applied. This prevents a situation where neither the
owner of the dialect containing the object nor the owner of the interface are
aware of an interface implementation, which can lead to duplicate or
diverging implementations.

#### Dialect Fallback for OpInterface

Some dialects have an open ecosystem and don't register all of the possible
operations. In such cases it is still possible to provide support for
implementing an `OpInterface` for these operation. When an operation isn't
registered or does not provide an implementation for an interface, the query
will fallback to the dialect itself.

A second model is used for such cases and automatically generated when using ODS
(see below) with the name `FallbackModel`. This model can be implemented for a
particular dialect:

```c++
// This is the implementation of a dialect fallback for `ExampleOpInterface`.
struct FallbackExampleOpInterface
    : public ExampleOpInterface::FallbackModel<
          FallbackExampleOpInterface> {
  static bool classof(Operation *op) { return true; }

  unsigned exampleInterfaceHook(Operation *op) const;
  unsigned exampleStaticInterfaceHook() const;
};
```

A dialect can then instantiate this implementation and returns it on specific
operations by overriding the `getRegisteredInterfaceForOp` method :

```c++
void *TestDialect::getRegisteredInterfaceForOp(TypeID typeID,
                                               StringAttr opName) {
  if (typeID == TypeID::get<ExampleOpInterface>()) {
    if (isSupported(opName))
      return fallbackExampleOpInterface;
    return nullptr;
  }
  return nullptr;
}
```

#### Utilizing the ODS Framework

Note: Before reading this section, the reader should have some familiarity with
the concepts described in the
[`Operation Definition Specification`](OpDefinitions.md) documentation.

As detailed above, [Interfaces](#attributeoperationtype-interfaces) allow for
attributes, operations, and types to expose method calls without requiring that
the caller know the specific derived type. The downside to this infrastructure,
is that it requires a bit of boiler plate to connect all of the pieces together.
MLIR provides a mechanism with which to defines interfaces declaratively in ODS,
and have the C++ definitions auto-generated.

As an example, using the ODS framework would allow for defining the example
interface above as:

```tablegen
def ExampleOpInterface : OpInterface<"ExampleOpInterface"> {
  let description = [{
    This is an example interface definition.
  }];

  let methods = [
    InterfaceMethod<
      "This is an example of a non-static hook to an operation.",
      "unsigned", "exampleInterfaceHook"
    >,
    StaticInterfaceMethod<
      "This is an example of a static hook to an operation.",
      "unsigned", "exampleStaticInterfaceHook"
    >,
  ];
}
```

Providing a definition of the `AttrInterface`, `OpInterface`, or `TypeInterface`
class will auto-generate the C++ classes for the interface. Interfaces are
comprised of the following components:

*   C++ Class Name (Provided via template parameter)
    -   The name of the C++ interface class.
*   Description (`description`)
    -   A string description of the interface, its invariants, example usages,
        etc.
*   C++ Namespace (`cppNamespace`)
    -   The C++ namespace that the interface class should be generated in.
*   Methods (`methods`)
    -   The list of interface hook methods that are defined by the IR object.
    -   The structure of these methods is defined below.
*   Extra Class Declarations (Optional: `extraClassDeclaration`)
    -   Additional C++ code that is generated in the declaration of the
        interface class. This allows for defining methods and more on the user
        facing interface class, that do not need to hook into the IR entity.
        These declarations are _not_ implicitly visible in default
        implementations of interface methods, but static declarations may be
        accessed with full name qualification.
*   Extra Shared Class Declarations (Optional: `extraSharedClassDeclaration`)
    -   Additional C++ code that is injected into the declarations of both the
        interface and trait class. This allows for defining methods and more
        that are exposed on both the interface and trait class, e.g. to inject
        utilties on both the interface and the derived entity implementing the
        interface (e.g. attribute, operation, etc.).
    -   In non-static methods, `$_attr`/`$_op`/`$_type`
        (depending on the type of interface) may be used to refer to an
        instance of the IR entity. In the interface declaration, the type of
        the instance is the interface class. In the trait declaration, the
        type of the instance is the concrete entity class
        (e.g. `IntegerAttr`, `FuncOp`, etc.).

`OpInterface` classes may additionally contain the following:

*   Verifier (`verify`)
    -   A C++ code block containing additional verification applied to the
        operation that the interface is attached to.
    -   The structure of this code block corresponds 1-1 with the structure of a
        [`Trait::verifyTrait`](Traits.md) method.

There are two types of methods that can be used with an interface,
`InterfaceMethod` and `StaticInterfaceMethod`. They are both comprised of the
same core components, with the distinction that `StaticInterfaceMethod` models a
static method on the derived IR object.

Interface methods are comprised of the following components:

*   Description
    -   A string description of this method, its invariants, example usages,
        etc.
*   ReturnType
    -   A string corresponding to the C++ return type of the method.
*   MethodName
    -   A string corresponding to the C++ name of the method.
*   Arguments (Optional)
    -   A dag of strings that correspond to a C++ type and variable name
        respectively.
*   MethodBody (Optional)
    -   An optional explicit implementation of the interface method.
    -   This implementation is placed within the method defined on the `Model`
        traits class, and is not defined by the `Trait` class that is attached
        to the IR entity. More concretely, this body is only visible by the
        interface class and does not affect the derived IR entity.
    -   `ConcreteAttr`/`ConcreteOp`/`ConcreteType` is an implicitly defined
        `typename` that can be used to refer to the type of the derived IR
        entity currently being operated on.
    -   In non-static methods, `$_op` and `$_self` may be used to refer to an
        instance of the derived IR entity.
*   DefaultImplementation (Optional)
    -   An optional explicit default implementation of the interface method.
    -   This implementation is placed within the `Trait` class that is attached
        to the IR entity, and does not directly affect any of the interface
        classes. As such, this method has the same characteristics as any other
        [`Trait`](Traits.md) method.
    -   `ConcreteAttr`/`ConcreteOp`/`ConcreteType` is an implicitly defined
        `typename` that can be used to refer to the type of the derived IR
        entity currently being operated on.
    -   This may refer to static fields of the interface class using the
        qualified name, e.g., `TestOpInterface::staticMethod()`.

ODS also allows for generating declarations for the `InterfaceMethod`s of an
operation if the operation specifies the interface with
`DeclareOpInterfaceMethods` (see an example below).

Examples:

~~~tablegen
def MyInterface : OpInterface<"MyInterface"> {
  let description = [{
    This is the description of the interface. It provides concrete information
    on the semantics of the interface, and how it may be used by the compiler.
  }];

  let methods = [
    InterfaceMethod<[{
      This method represents a simple non-static interface method with no
      inputs, and a void return type. This method is required to be implemented
      by all operations implementing this interface. This method roughly
      correlates to the following on an operation implementing this interface:

      ```c++
      class ConcreteOp ... {
      public:
        void nonStaticMethod();
      };
      ```
    }], "void", "nonStaticMethod"
    >,

    InterfaceMethod<[{
      This method represents a non-static interface method with a non-void
      return value, as well as an `unsigned` input named `i`. This method is
      required to be implemented by all operations implementing this interface.
      This method roughly correlates to the following on an operation
      implementing this interface:

      ```c++
      class ConcreteOp ... {
      public:
        Value nonStaticMethod(unsigned i);
      };
      ```
    }], "Value", "nonStaticMethodWithParams", (ins "unsigned":$i)
    >,

    StaticInterfaceMethod<[{
      This method represents a static interface method with no inputs, and a
      void return type. This method is required to be implemented by all
      operations implementing this interface. This method roughly correlates
      to the following on an operation implementing this interface:

      ```c++
      class ConcreteOp ... {
      public:
        static void staticMethod();
      };
      ```
    }], "void", "staticMethod"
    >,

    StaticInterfaceMethod<[{
      This method corresponds to a static interface method that has an explicit
      implementation of the method body. Given that the method body has been
      explicitly implemented, this method should not be defined by the operation
      implementing this method. This method merely takes advantage of properties
      already available on the operation, in this case its `build` methods. This
      method roughly correlates to the following on the interface `Model` class:

      ```c++
      struct InterfaceTraits {
        /// ... The `Concept` class is elided here ...

        template <typename ConcreteOp>
        struct Model : public Concept {
          Operation *create(OpBuilder &builder, Location loc) const override {
            return builder.create<ConcreteOp>(loc);
          }
        }
      };
      ```

      Note above how no modification is required for operations implementing an
      interface with this method.
    }],
      "Operation *", "create", (ins "OpBuilder &":$builder, "Location":$loc),
      /*methodBody=*/[{
        return builder.create<ConcreteOp>(loc);
    }]>,

    InterfaceMethod<[{
      This method represents a non-static method that has an explicit
      implementation of the method body. Given that the method body has been
      explicitly implemented, this method should not be defined by the operation
      implementing this method. This method merely takes advantage of properties
      already available on the operation, in this case its `build` methods. This
      method roughly correlates to the following on the interface `Model` class:

      ```c++
      struct InterfaceTraits {
        /// ... The `Concept` class is elided here ...

        template <typename ConcreteOp>
        struct Model : public Concept {
          Operation *create(Operation *opaqueOp, OpBuilder &builder,
                            Location loc) const override {
            ConcreteOp op = cast<ConcreteOp>(opaqueOp);
            return op.getNumInputs() + op.getNumOutputs();
          }
        }
      };
      ```

      Note above how no modification is required for operations implementing an
      interface with this method.
    }],
      "unsigned", "getNumInputsAndOutputs", (ins), /*methodBody=*/[{
        return $_op.getNumInputs() + $_op.getNumOutputs();
    }]>,

    InterfaceMethod<[{
      This method represents a non-static method that has a default
      implementation of the method body. This means that the implementation
      defined here will be placed in the trait class that is attached to every
      operation that implements this interface. This has no effect on the
      generated `Concept` and `Model` class. This method roughly correlates to
      the following on the interface `Trait` class:

      ```c++
      template <typename ConcreteOp>
      class MyTrait : public OpTrait::TraitBase<ConcreteType, MyTrait> {
      public:
        bool isSafeToTransform() {
          ConcreteOp op = cast<ConcreteOp>(this->getOperation());
          return op.getNumInputs() + op.getNumOutputs();
        }
      };
      ```

      As detailed in [Traits](Traits.md), given that each operation implementing
      this interface will also add the interface trait, the methods on this
      interface are inherited by the derived operation. This allows for
      injecting a default implementation of this method into each operation that
      implements this interface, without changing the interface class itself. If
      an operation wants to override this default implementation, it merely
      needs to implement the method and the derived implementation will be
      picked up transparently by the interface class.

      ```c++
      class ConcreteOp ... {
      public:
        bool isSafeToTransform() {
          // Here we can override the default implementation of the hook
          // provided by the trait.
        }
      };
      ```
    }],
      "bool", "isSafeToTransform", (ins), /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
    }]>,
  ];
}

// Operation interfaces can optionally be wrapped inside
// DeclareOpInterfaceMethods. This would result in autogenerating declarations
// for members `foo`, `bar` and `fooStatic`. Methods with bodies are not
// declared inside the op declaration but instead handled by the op interface
// trait directly.
def OpWithInferTypeInterfaceOp : Op<...
    [DeclareOpInterfaceMethods<MyInterface>]> { ... }

// Methods that have a default implementation do not have declarations
// generated. If an operation wishes to override the default behavior, it can
// explicitly specify the method that it wishes to override. This will force
// the generation of a declaration for those methods.
def OpWithOverrideInferTypeInterfaceOp : Op<...
    [DeclareOpInterfaceMethods<MyInterface, ["getNumWithDefault"]>]> { ... }
~~~

Note: Existing operation interfaces defined in C++ can be accessed in the ODS
framework via the `OpInterfaceTrait` class.

#### Operation Interface List

MLIR includes standard interfaces providing functionality that is likely to be
common across many different operations. Below is a list of some key interfaces
that may be used directly by any dialect. The format of the header for each
interface section goes as follows:

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
    -   `RegionKind getRegionKind(unsigned index)` - Return the kind of the
        region with the given index inside this operation.
        -   RegionKind::Graph - represents a graph region without control flow
            semantics
        -   RegionKind::SSACFG - represents an
            [SSA-style control flow](LangRef.md/#control-flow-and-ssacfg-regions) region
            with basic blocks and reachability
    -   `hasSSADominance(unsigned index)` - Return true if the region with the
        given index inside this operation requires dominance.

##### SymbolInterfaces

*   `SymbolOpInterface` - Used to represent
    [`Symbol`](SymbolsAndSymbolTables.md/#symbol) operations which reside
    immediately within a region that defines a
    [`SymbolTable`](SymbolsAndSymbolTables.md/#symbol-table).

*   `SymbolUserOpInterface` - Used to represent operations that reference
    [`Symbol`](SymbolsAndSymbolTables.md/#symbol) operations. This provides the
    ability to perform safe and efficient verification of symbol uses, as well
    as additional functionality.
