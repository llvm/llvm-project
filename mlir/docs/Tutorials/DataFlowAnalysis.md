# Writing DataFlow Analyses in MLIR

Writing dataflow analyses in MLIR, or well any compiler, can often seem quite
daunting and/or complex. A dataflow analysis generally involves propagating
information about the IR across various different types of control flow
constructs, of which MLIR has many (Block-based branches, Region-based branches,
CallGraph, etc), and it isn't always clear how best to go about performing the
propagation. Dataflow analyses often require implementing fixed-point iteration
when data dependencies form cycles, as can happen with control-flow. Tracking
dependencies and making sure updates are properly propagated can get quite
difficult when writing complex analyses. That is why MLIR provides a framework
for writing general dataflow analyses as well as several utilities to streamline
the implementation of common analyses. The code and test from this tutorial can 
be found in `mlir/examples/dataflow`.

## DataFlow Analysis Framework

MLIR provides a general dataflow analysis framework for building fixed-point
iteration dataflow analyses with ease and utilities for common dataflow
analyses. Because the landscape of IRs in MLIR can be vast, the framework is
designed to be extensible and composable, so that utilities can be shared across
dialects with different semantics as much as possible. The framework also tries
to make debugging dataflow analyses easy by providing (hopefully) insightful
logs with `-debug-only="dataflow"`.

Suppose we want to compute at compile-time the constant-valued results of
operations. For example, consider:

```mlir
%0 = string.constant "foo"
%1 = string.constant "bar"
%2 = string.concat %0, %1
```
We can determine with the information in the IR at compile time the value of
`%2` to be "foobar". This is called constant propagation. In MLIR's dataflow
analysis framework, this is in general called the "analysis state of a program
point"; the "state" being, in this case, the constant value, and the "program
point" being the SSA value `%2`.

The constant value state of an SSA value is implemented as a subclass of
`AnalysisState`, and program points are represented by the `ProgramPoint` union,
which can be operations, SSA values, or blocks. They can also be just about
anything, see [Extending ProgramPoint](#extending-programpoint). In general, an
analysis state represents information about the IR computed by an analysis. 

Let us define an analysis state to represent a compile time known string value
of an SSA value:

```c++
class StringConstant : public AnalysisState {
  /// This is the known string constant value of an SSA value at compile time
  /// as determined by a dataflow analysis. To implement the concept of being
  /// "uninitialized", the potential string value is wrapped in an `Optional`
  /// and set to `None` by default to indicate that no value has been provided.
  std::optional<std::string> stringValue = std::nullopt;

public:
  using AnalysisState::AnalysisState;

  /// Return true if no value has been provided for the string constant value.
  bool isUninitialized() const { return !stringValue.has_value(); }

  /// Default initialized the state to an empty string. Return whether the value
  /// of the state has changed.
  ChangeResult defaultInitialize() {
    // If the state already has a value, do nothing.
    if (!isUninitialized())
      return ChangeResult::NoChange;
    // Initialize the state and indicate that its value changed.
    stringValue = "";
    return ChangeResult::Change;
  }

  /// Get the currently known string value.
  StringRef getStringValue() const {
    assert(!isUninitialized() && "getting the value of an uninitialized state");
    return stringValue.value();
  }

  /// "Join" the value of the state with another constant.
  ChangeResult join(const Twine &value) {
    // If the current state is uninitialized, just take the value.
    if (isUninitialized()) {
      stringValue = value.str();
      return ChangeResult::Change;
    }
    // If the current state is "overdefined", no new information can be taken.
    if (stringValue->empty())
      return ChangeResult::NoChange;
    // If the current state has a different value, it now has two conflicting
    // values and should go to overdefined.
    if (stringValue != value.str()) {
      stringValue = "";
      return ChangeResult::Change;
    }
    return ChangeResult::NoChange;
  }

  /// Print the constant value.
  void print(raw_ostream &os) const override {
    os << stringValue.value_or("") << "\n";
  }
};
```

Analysis states often depend on each other. In our example, the constant value
of `%2` depends on that of `%0` and `%1`. It stands to reason that the constant
value of `%2` needs to be recomputed when that of `%0` and `%1` change. The
`DataFlowSolver` implements the fixed-point iteration algorithm and manages the
dependency graph between analysis states.

The computation of analysis states, on the other hand, is performed by dataflow
analyses, subclasses of `DataFlowAnalysis`. A dataflow analysis has to implement
a "transfer function", that is, code that computes the values of some states
using the values of others, and set up the dependency graph correctly. Since the
dependency graph inside the solver is initially empty, it must also set up the
dependency graph.

```c++
class DataFlowAnalysis {
public:
  /// "Visit" the provided program point. This method is typically used to
  /// implement transfer functions on or across program points.
  virtual LogicalResult visit(ProgramPoint point) = 0;

  /// Initialize the dependency graph required by this analysis from the given
  /// top-level operation. This function is called once by the solver before
  /// running the fixed-point iteration algorithm.
  virtual LogicalResult initialize(Operation *top) = 0;

protected:
  /// Create a dependency between the given analysis state and lattice anchor
  /// on this analysis.
  void addDependency(AnalysisState *state, ProgramPoint *point);

  /// Propagate an update to a state if it changed.
  void propagateIfChanged(AnalysisState *state, ChangeResult changed);

  /// Get the analysis state associated with the lattice anchor. The returned
  /// state is expected to be "write-only", and any updates need to be
  /// propagated by `propagateIfChanged`.
  template <typename StateT, typename AnchorT>
  StateT *getOrCreate(AnchorT anchor) {
    return solver.getOrCreateState<StateT>(anchor);
  }
};
```

Dependency management is a little unusual in this framework. The dependents of
the value of a state are not other states but invocations of dataflow analyses
on certain program points. For example:

```c++
class StringConstantPropagation : public DataFlowAnalysis {
public:
  /// Implement the transfer function for string operations. When visiting a
  /// string operation, this analysis will try to determine compile time values
  /// of the operation's results and set them in `StringConstant` states. This
  /// function is invoked on an operation whenever the states of its operands
  /// are changed.
  LogicalResult visit(ProgramPoint point) override {
    // This function expects only to receive operations.
    auto *op = point->getPrevOp();

    // Get or create the constant string values of the operands.
    SmallVector<StringConstant *> operandValues;
    for (Value operand : op->getOperands()) {
      auto *value = getOrCreate<StringConstant>(operand);
      // Create a dependency from the state to this analysis. When the string
      // value of one of the operation's operands are updated, invoke the
      // transfer function again.
      addDependency(value, point);
      // If the state is uninitialized, bail out and come back later when it is
      // initialized.
      if (value->isUninitialized())
        return success();
      operandValues.push_back(value);
    }

    // Try to compute a constant value of the result.
    auto *result = getOrCreate<StringConstant>(op->getResult(0));
    if (auto constant = dyn_cast<string::ConstantOp>(op)) {
      // Just grab and set the constant value of the result of the operation.
      // Propagate an update to the state if it changed.
      propagateIfChanged(result, result->join(constant.getValue()));
    } else if (auto concat = dyn_cast<string::ConcatOp>(op)) {
      StringRef lhs = operandValues[0]->getStringValue();
      StringRef rhs = operandValues[1]->getStringValue();
      // If either operand is overdefined, the results are overdefined.
      if (lhs.empty() || rhs.empty()) {
        propagateIfChanged(result, result->defaultInitialize());

        // Otherwise, compute the constant value and join it with the result.
      } else {
        propagateIfChanged(result, result->join(lhs + rhs));
      }
    } else {
      // We don't know how to implement the transfer function for this
      // operation. Mark its results as overdefined.
      propagateIfChanged(result, result->defaultInitialize());
    }
    return success();
  }
};
```

In the above example, the `visit` function sets up the dependencies of the
analysis invocation on an operation as the constant values of the operands of
each operation. When the operand states have initialized values but overdefined
values, it sets the state of the result to overdefined. Otherwise, it computes
the state of the result and merges the new information in with `join`.

However, the dependency graph still needs to be initialized before the solver
knows what to call `visit` on. This is done in the `initialize` function:

```c++
LogicalResult StringConstantPropagation::initialize(Operation *top) {
  // Visit every nested string operation and set up its dependencies.
  top->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      auto *state = getOrCreate<StringConstant>(operand);
      addDependency(state, getProgramPointAfter(op));
    }
  });
  // Now that the dependency graph has been set up, "seed" the evolution of the
  // analysis by marking the constant values of all block arguments as
  // overdefined and the results of (non-constant) operations with no operands.
  auto defaultInitializeAll = [&](ValueRange values) {
    for (Value value : values) {
      auto *state = getOrCreate<StringConstant>(value);
      propagateIfChanged(state, state->defaultInitialize());
    }
  };
  top->walk([&](Operation *op) {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        defaultInitializeAll(block.getArguments());
    if (auto constant = dyn_cast<string::ConstantOp>(op)) {
      auto *result = getOrCreate<StringConstant>(constant.getResult());
      propagateIfChanged(result, result->join(constant.getValue()));
    } else if (op->getNumOperands() == 0) {
      defaultInitializeAll(op->getResults());
    }
  });
  // The dependency graph has been set up and the analysis has been seeded.
  // Finish initialization and let the solver run.
  return success();
}
```

Note that we can remove the call to `addDependency` inside our `visit` function
because the dependencies are set by the initialize function. Dependencies added
inside the `visit` function -- that is, while the solver is running -- are
called "dynamic dependencies". Dependending on the kind of analysis, it may be
more efficient to set some dependencies statically or dynamically.

Another way to improve the efficiency of our analysis is to recognize that this
is a *sparse*, *forward* analysis. It is sparse because the dependencies of an
operation's transfer function are only the states of its operands, meaning that
we can track dependencies through the IR instead of relying on the solver to do
the bookkeeping. It is forward (assuming our IR has SSA dominance) because
information can only be propagated from an SSA value's definition to its users.

That is a lot of code to write, however, so the framework comes with utilities
for implementing conditional sparse and dense dataflow analyses. See
[Sparse Forward DataFlowAnalysis](#sparse-forward-dataflow-analysis).

### Running the Solver

Setting up the dataflow solver is straightforward:

```c++
void MyPass::runOnOperation() {
  Operation *top = getOperation();
  DataFlowSolver solver;
  // Load the analysis.
  solver.load<StringConstantPropagation>();
  // Run the solver!
  if (failed(solver.initializeAndRun(top)))
    return signalPassFailure();
  // Query the results and do something...
  top->walk([&](string::ConcatOp concat) {
    auto *result = solver.lookupState<StringConstant>(concat.getResult());
    // ...
  });
}
```

The following is a simple example.

```mlir
func.func @single_concat() {
  %1 = string.constant "hello "
  %2 = string.constant "world."
  %3 = string.concat %1, %2
  return
}
```

The above IR will print the following after running pass.

```mlir
%0 = string.constant "hello " : hello 
%1 = string.constant "world." : world.
%2 = string.concat %0, %1 : hello world.
```

### Extending ProgramPoint

`ProgramPoint` can be extended to represent just about anything in a program:
control-flow edges or memory addresses. Custom "generic" program points are
implemented as subclasses of `GenericProgramPointBase`, a user of the storage
uniquer API, with a content-key.

Example 1: a control-flow edge between two blocks. Suppose we want to represent
the state of an edge in the control-flow graph, such as its liveness. We can
attach such a state to the custom program point:

```c++
/// This program point represents a control-flow edge between two blocks. The
/// block `from` is a predecessor of `to`.
class CFGEdge
    : public GenericLatticeAnchorBase<CFGEdge, std::pair<Block *, Block *>> {
public:
  Block *getFrom() const { return getValue().first; }
  Block *getTo() const { return getValue().second; }
};
```

Example 2: a raw memory address after the execution of an operation. This
program point allows us to attach states to a raw memory address before an
operation after an operation is executed.

```c++
class RawMemoryAddr : public GenericProgramPointBase<
    RawMemoryAddr, std::pair<uintptr_t, Operation *>> { /* ... */ };
```

Instances of program points can be accessed as follows:

```c++
Block *from = /* ... */, *to = /* ... */;
auto *cfgEdge = solver.getProgramPoint<CFGEdge>(from, to);

Operation *op = /* ... */;
auto *addr = solver.getProgramPoint<RawMemoryAddr>(0x3000, op);
```

## Sparse Forward DataFlow Analysis

One type of dataflow analysis is a sparse forward propagation analysis. This
type of analysis, as the name may suggest, propagates information forward (e.g.
from definitions to uses). The class `SparseDataFlowAnalysis` implements much of
the analysis logic, including handling control-flow, and abstracts away the
dependency management.

To provide a bit of concrete context, let's go over writing a simple forward
dataflow analysis in MLIR. Let's say for this analysis that we want to propagate
information about a special "metadata" dictionary attribute. The contents of
this attribute are simply a set of metadata that describe a specific value, e.g.
`metadata = { likes_pizza = true }`. We will collect the `metadata` for
operations in the IR and propagate them about.

### Lattices

Before going into how one might setup the analysis itself, it is important to
first introduce the concept of a `Lattice` and how we will use it for the
analysis. A lattice represents all of the possible values or results of the
analysis for a given value. A lattice element holds the set of information
computed by the analysis for a given value, and is what gets propagated across
the IR. For our analysis, this would correspond to the `metadata` dictionary
attribute.

Regardless of the value held within, every type of lattice contains two special
element states:

*   `uninitialized`

    -   The element has not been initialized.

*   `top`/`overdefined`/`unknown`

    -   The element encompasses every possible value.
    -   This is a very conservative state, and essentially means "I can't make
        any assumptions about the value, it could be anything"

These two states are important when merging, or `join`ing as we will refer to it
further in this document, information as part of the analysis. Lattice elements
are `join`ed whenever there are two different source points, such as an argument
to a block with multiple predecessors. One important note about the `join`
operation, is that it is required to be monotonic (see the `join` method in the
example below for more information). This ensures that `join`ing elements is
consistent. The two special states mentioned above have unique properties during
a `join`:

*   `uninitialized`

    -   If one of the elements is `uninitialized`, the other element is used.
    -   `uninitialized` in the context of a `join` essentially means "take the
        other thing".

*   `top`/`overdefined`/`unknown`

    -   If one of the elements being joined is `overdefined`, the result is
        `overdefined`.

For our analysis in MLIR, we will need to define a class representing the value
held by an element of the lattice used by our dataflow analysis:

```c++
/// The value of our lattice represents the inner structure of a DictionaryAttr,
/// for the `metadata`.
struct MetadataLatticeValue {
  MetadataLatticeValue() = default;
  /// Compute a lattice value from the provided dictionary.
  MetadataLatticeValue(DictionaryAttr attr) {
    for (NamedAttribute pair : attr) {
      metadata.insert(
          std::pair<StringAttr, Attribute>(pair.getName(), pair.getValue()));
    }
  }

  /// This method conservatively joins the information held by `lhs` and `rhs`
  /// into a new value. This method is required to be monotonic. `monotonicity`
  /// is implied by the satisfaction of the following axioms:
  ///   * idempotence:   join(x,x) == x
  ///   * commutativity: join(x,y) == join(y,x)
  ///   * associativity: join(x,join(y,z)) == join(join(x,y),z)
  ///
  /// When the above axioms are satisfied, we achieve `monotonicity`:
  ///   * monotonicity: join(x, join(x,y)) == join(x,y)
  static MetadataLatticeValue join(const MetadataLatticeValue &lhs,
                                   const MetadataLatticeValue &rhs) {
  // To join `lhs` and `rhs` we will define a simple policy, which is that we
  // directly insert the metadata of rhs into the metadata of lhs.If lhs and rhs
  // have overlapping attributes, keep the attribute value in lhs unchanged.
  MetadataLatticeValue result;
  for (auto &&lhsIt : lhs.metadata) {
    result.metadata.insert(
        std::pair<StringAttr, Attribute>(lhsIt.first, lhsIt.second));
  }

  for (auto &&rhsIt : rhs.metadata) {
    result.metadata.insert(
        std::pair<StringAttr, Attribute>(rhsIt.first, rhsIt.second));
  }
  return result;
}

  /// A simple comparator that checks to see if this value is equal to the one
  /// provided.
  bool operator==(const MetadataLatticeValue &rhs) const {
  if (metadata.size() != rhs.metadata.size())
    return false;

  // Check that `rhs` contains the same metadata.
  for (auto &&it : metadata) {
    auto rhsIt = rhs.metadata.find(it.first);
    if (rhsIt == rhs.metadata.end() || it.second != rhsIt->second)
      return false;
  }
  return true;
}

  /// Print data in metadata.
  void print(llvm::raw_ostream &os) const  {
  SmallVector<StringAttr> metadataKey(metadata.keys());
  std::sort(metadataKey.begin(), metadataKey.end(),
            [&](StringAttr a, StringAttr b) { return a < b; });
  os << "{";
  for (StringAttr key : metadataKey) {
    os << key << ": " << metadata.at(key) << ", ";
  }
  os << "\b\b}\n";
}

  /// Our value represents the combined metadata, which is originally a
  /// DictionaryAttr, so we use a map.
  DenseMap<StringAttr, Attribute> metadata;
};
```

One interesting thing to note above is that we don't have an explicit method for
the `uninitialized` state. This state is handled by the `LatticeElement` class,
which manages a lattice value for a given IR entity. A quick overview of this
class, and the API that will be interesting to us while writing our analysis, is
shown below:

```c++
/// This class represents a lattice element holding a specific value of type
/// `ValueT`.
template <typename ValueT>
class Lattice ... {
public:
  /// Return the value held by this element. This requires that a value is
  /// known, i.e. not `uninitialized`.
  ValueT &getValue();
  const ValueT &getValue() const;

  /// Join the information contained in the 'rhs' element into this
  /// element. Returns if the state of the current element changed.
  ChangeResult join(const LatticeElement<ValueT> &rhs);

  /// Join the information contained in the 'rhs' value into this
  /// lattice. Returns if the state of the current lattice changed.
  ChangeResult join(const ValueT &rhs);
  
  ...
};
```

With our lattice defined, we can now define the driver that will compute and
propagate our lattice across the IR. The following is our definition of metadata
lattice.

```c++
class MetadataLatticeValueLattice : public Lattice<MetadataLatticeValue> {
public:
  using Lattice::Lattice;
};
```

### SparseForwardDataFlowAnalysis Driver

The `SparseForwardDataFlowAnalysis` class represents the driver of the dataflow
analysis, and performs all of the related analysis computation. When defining
our analysis, we will inherit from this class and implement some of its hooks.
Before that, let's look at a quick overview of this class and some of the
important API for our analysis:

```c++
/// This class represents the main driver of the forward dataflow analysis. It
/// takes as a template parameter the value type of lattice being computed.
template <typename StateT>
class SparseForwardDataFlowAnalysis : ... {
public:
  explicit SparseForwardDataFlowAnalysis(DataFlowSolver &solver)
      : AbstractSparseForwardDataFlowAnalysis(solver) {}

  /// Visit an operation with the lattices of its operands. This function is
  /// expected to set the lattices of the operation's results.
  virtual LogicalResult visitOperation(Operation *op,
                                       ArrayRef<const StateT *> operands,
                                       ArrayRef<StateT *> results) = 0;
  ...

protected:
  /// Return the lattice element attached to the given value. If a lattice has
  /// not been added for the given value, a new 'uninitialized' value is
  /// inserted and returned.
  StateT *getLatticeElement(Value value);

  /// Get the lattice element for a value and create a dependency on the
  /// provided program point.
  const StateT *getLatticeElementFor(ProgramPoint *point, Value value);

  /// Set the given lattice element(s) at control flow entry point(s).
  virtual void setToEntryState(StateT *lattice) = 0;
  ...
};
```

NOTE: Some API has been redacted for our example. The `SparseForwardDataFlowAnalysis`
contains various other hooks that allow for injecting custom behavior when
applicable.

The main API that we are responsible for defining is the `visitOperation`
method. This method is responsible for computing new lattice elements for the
results and block arguments owned by the given operation. This is where we will
inject the lattice element computation logic, also known as the transfer
function for the operation, that is specific to our analysis. A simple
implementation for our example is shown below:

```c++
class MetadataAnalysis
    : public SparseForwardDataFlowAnalysis<MetadataLatticeValueLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const MetadataLatticeValueLattice *> operands,
                 ArrayRef<MetadataLatticeValueLattice *> results) override {
  DictionaryAttr metadata = op->getAttrOfType<DictionaryAttr>("metadata");
  // If we have no metadata for this operation and the operands is empty, we
  // will conservatively mark all of the results as having reached a pessimistic
  // fixpoint.
  if (!metadata && operands.empty()) {
    setAllToEntryStates(results);
    return success();
  }

  MetadataLatticeValue latticeValue;
  if (metadata)
    latticeValue = MetadataLatticeValue(metadata);

  // Otherwise, we will compute a lattice value for the metadata and join it
  // into the current lattice element for all of our results.`results` stores
  // the lattices corresponding to the results of op, We use a loop to traverse
  // them.
  for (MetadataLatticeValueLattice *: results) {

    // `isChanged` records whether the result has been changed.
    ChangeResult isChanged = ChangeResult::NoChange;

    // Op's metadata is joined result's lattice.
    isChanged |= result->join(latticeValue);

    // All lattice of operands of op are joined to the lattice of result.
    for (auto operand : operands)
      isChanged |= result->join(*operand);

    propagateIfChanged(result, isChanged);
  }
  return success();
  }
};
```

With that, we have all of the necessary components to compute our analysis.
After the analysis has been computed, we need to run our analysis using 
`DataFlowSolver`, and we can grab any computed information for values by 
using `lookupState`. See below for a quick example, after the pass runs the
analysis, we print the metadata of each op's results.

```c++
void MyPass::runOnOperation() {
  Operation *op = getOperation();
  DataFlowSolver solver;
  solver.load<DeadCodeAnalysis>();
  solver.load<MetadataAnalysis>();
  if (failed(solver.initializeAndRun(op)))
    return signalPassFailure();

  // If an op has more than one result, then the lattice is the same for each
  // result, and we just print one of the results.
  op->walk([&](Operation *op) {
    if (op->getNumResults()) {
      Value result = op->getResult(0);
      auto lattice = solver.lookupState<MetadataLatticeValueLattice>(result);
      llvm::outs() << OpWithFlags(op, OpPrintingFlags().skipRegions()) << " : ";
      lattice->print(llvm::outs());
    }
  });
  ...
}
```

The following is a simple example. More tests can be found in the `mlir/Example/dataflow`.

```mlir
func.func @single_join(%arg0 : index, %arg1 : index) -> index {
  %1 = arith.addi %arg0, %arg1 {metadata = { likes_pizza = true }} : index
  %2 = arith.addi %1, %arg1 : index
  return %2 : index
}
```

The above IR will print the following after running pass.

```mlir
%0 = arith.addi %arg0, %arg1 {metadata = {likes_pizza = true}} : index : {"likes_pizza": true} 
%1 = arith.addi %0, %arg1 : index : {"likes_pizza": true} 
```
