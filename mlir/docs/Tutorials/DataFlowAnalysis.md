# Writing DataFlow Analyses in MLIR

Writing dataflow analyses in MLIR, or well any compiler, can often seem quite
daunting and/or complex. A dataflow analysis generally involves propagating
information about the IR across various different types of control flow
constructs, of which MLIR has many (Block-based branches, Region-based branches,
CallGraph, etc), and it isn't always clear how best to go about performing the
propagation. To help writing these types of analyses in MLIR, this document
details several utilities that simplify the process and make it a bit more
approachable. The code from this tutorial can be found in `mlir/examples/dataflow`.

## Forward Dataflow Analysis

One type of dataflow analysis is a forward propagation analysis. This type of
analysis, as the name may suggest, propagates information forward (e.g. from
definitions to uses). To provide a bit of concrete context, let's go over
writing a simple forward dataflow analysis in MLIR. Let's say for this analysis
that we want to propagate information about a special "metadata" dictionary
attribute. The contents of this attribute are simply a set of metadata that
describe a specific value, e.g. `metadata = { likes_pizza = true }`. We will
collect the `metadata` for operations in the IR and propagate them about.

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
  for (auto result : results) {

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

```
%0 = arith.addi %arg0, %arg1 {metadata = {likes_pizza = true}} : index : {"likes_pizza": true} 
%1 = arith.addi %0, %arg1 : index : {"likes_pizza": true} 
```
