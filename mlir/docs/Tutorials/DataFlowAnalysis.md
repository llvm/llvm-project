# Writing DataFlow Analyses in MLIR

Writing dataflow analyses in MLIR, or well any compiler, can often seem quite
daunting and/or complex. A dataflow analysis generally involves propagating
information about the IR across various different types of control flow
constructs, of which MLIR has many (Block-based branches, Region-based branches,
CallGraph, etc), and it isn't always clear how best to go about performing the
propagation. To help writing these types of analyses in MLIR, this document
details several utilities that simplify the process and make it a bit more
approachable.

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
  MetadataLatticeValue(DictionaryAttr attr)
      : metadata(attr.begin(), attr.end()) {}

  /// Return a pessimistic value state, i.e. the `top`/`overdefined`/`unknown`
  /// state, for our value type. The resultant state should not assume any
  /// information about the state of the IR.
  static MetadataLatticeValue getPessimisticValueState(MLIRContext *context) {
    // The `top`/`overdefined`/`unknown` state is when we know nothing about any
    // metadata, i.e. an empty dictionary.
    return MetadataLatticeValue();
  }
  /// Return a pessimistic value state for our value type using only information
  /// about the state of the provided IR. This is similar to the above method,
  /// but may produce a slightly more refined result. This is okay, as the
  /// information is already encoded as fact in the IR.
  static MetadataLatticeValue getPessimisticValueState(Value value) {
    // Check to see if the parent operation has metadata.
    if (Operation *parentOp = value.getDefiningOp()) {
      if (auto metadata = parentOp->getAttrOfType<DictionaryAttr>("metadata"))
        return MetadataLatticeValue(metadata);

      // If no metadata is present, fallback to the
      // `top`/`overdefined`/`unknown` state.
    }
    return MetadataLatticeValue();
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
    // only keep information that is the same. This means that we only keep
    // facts that are true in both.
    MetadataLatticeValue result;
    for (const auto &lhsIt : lhs.metadata) {
      // As noted above, we only merge if the values are the same.
      auto it = rhs.metadata.find(lhsIt.first);
      if (it == rhs.metadata.end() || it.second != lhsIt.second)
        continue;
      result.insert(lhsIt);
    }
    return result;
  }

  /// A simple comparator that checks to see if this value is equal to the one
  /// provided.
  bool operator==(const MetadataLatticeValue &rhs) const {
    if (metadata.size() != rhs.metadata.size())
      return false;
    // Check that `rhs` contains the same metadata.
    for (const auto &it : metadata) {
      auto rhsIt = rhs.metadata.find(it.first);
      if (rhsIt == rhs.metadata.end() || it.second != rhsIt.second)
        return false;
    }
    return true;
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
class LatticeElement ... {
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

  /// Mark the lattice element as having reached a pessimistic fixpoint. This
  /// means that the lattice may potentially have conflicting value states, and
  /// only the conservatively known value state should be relied on.
  ChangeResult markPessimisticFixPoint();
};
```

With our lattice defined, we can now define the driver that will compute and
propagate our lattice across the IR.

### ForwardDataflowAnalysis Driver

The `ForwardDataFlowAnalysis` class represents the driver of the dataflow
analysis, and performs all of the related analysis computation. When defining
our analysis, we will inherit from this class and implement some of its hooks.
Before that, let's look at a quick overview of this class and some of the
important API for our analysis:

```c++
/// This class represents the main driver of the forward dataflow analysis. It
/// takes as a template parameter the value type of lattice being computed.
template <typename ValueT>
class ForwardDataFlowAnalysis : ... {
public:
  ForwardDataFlowAnalysis(MLIRContext *context);

  /// Compute the analysis on operations rooted under the given top-level
  /// operation. Note that the top-level operation is not visited.
  void run(Operation *topLevelOp);

  /// Return the lattice element attached to the given value. If a lattice has
  /// not been added for the given value, a new 'uninitialized' value is
  /// inserted and returned.
  LatticeElement<ValueT> &getLatticeElement(Value value);

  /// Return the lattice element attached to the given value, or nullptr if no
  /// lattice element for the value has yet been created.
  LatticeElement<ValueT> *lookupLatticeElement(Value value);

  /// Mark all of the lattice elements for the given range of Values as having
  /// reached a pessimistic fixpoint.
  ChangeResult markAllPessimisticFixPoint(ValueRange values);

protected:
  /// Visit the given operation, and join any necessary analysis state
  /// into the lattice elements for the results and block arguments owned by
  /// this operation using the provided set of operand lattice elements
  /// (all pointer values are guaranteed to be non-null). Returns if any result
  /// or block argument value lattice elements changed during the visit. The
  /// lattice element for a result or block argument value can be obtained, and
  /// join'ed into, by using `getLatticeElement`.
  virtual ChangeResult visitOperation(
      Operation *op, ArrayRef<LatticeElement<ValueT> *> operands) = 0;
};
```

NOTE: Some API has been redacted for our example. The `ForwardDataFlowAnalysis`
contains various other hooks that allow for injecting custom behavior when
applicable.

The main API that we are responsible for defining is the `visitOperation`
method. This method is responsible for computing new lattice elements for the
results and block arguments owned by the given operation. This is where we will
inject the lattice element computation logic, also known as the transfer
function for the operation, that is specific to our analysis. A simple
implementation for our example is shown below:

```c++
class MetadataAnalysis : public ForwardDataFlowAnalysis<MetadataLatticeValue> {
public:
  using ForwardDataFlowAnalysis<MetadataLatticeValue>::ForwardDataFlowAnalysis;

  ChangeResult visitOperation(
      Operation *op, ArrayRef<LatticeElement<ValueT> *> operands) override {
    DictionaryAttr metadata = op->getAttrOfType<DictionaryAttr>("metadata");

    // If we have no metadata for this operation, we will conservatively mark
    // all of the results as having reached a pessimistic fixpoint.
    if (!metadata)
      return markAllPessimisticFixPoint(op->getResults());

    // Otherwise, we will compute a lattice value for the metadata and join it
    // into the current lattice element for all of our results.
    MetadataLatticeValue latticeValue(metadata);
    ChangeResult result = ChangeResult::NoChange;
    for (Value value : op->getResults()) {
      // We grab the lattice element for `value` via `getLatticeElement` and
      // then join it with the lattice value for this operation's metadata. Note
      // that during the analysis phase, it is fine to freely create a new
      // lattice element for a value. This is why we don't use the
      // `lookupLatticeElement` method here.
      result |= getLatticeElement(value).join(latticeValue);
    }
    return result;
  }
};
```

With that, we have all of the necessary components to compute our analysis.
After the analysis has been computed, we can grab any computed information for
values by using `lookupLatticeElement`. We use this function over
`getLatticeElement` as the analysis is not guaranteed to visit all values, e.g.
if the value is in a unreachable block, and we don't want to create a new
uninitialized lattice element in this case. See below for a quick example:

```c++
void MyPass::runOnOperation() {
  MetadataAnalysis analysis(&getContext());
  analysis.run(getOperation());
  ...
}

void MyPass::useAnalysisOn(MetadataAnalysis &analysis, Value value) {
  LatticeElement<MetadataLatticeValue> *latticeElement = analysis.lookupLatticeElement(value);

  // If we don't have an element, the `value` wasn't visited during our analysis
  // meaning that it could be dead. We need to treat this conservatively.
  if (!lattice)
    return;

  // Our lattice element has a value, use it:
  MetadataLatticeValue &value = lattice->getValue();
  ...
}
```
