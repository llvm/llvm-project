=============================================
Machine Learning - Guided Optimization (MLGO)
=============================================

Introduction
============

MLGO refers to integrating ML techniques (primarily) to replace heuristics within
LLVM with machine learned models.

Currently the following heuristics feature such integration:

* Inlining for size
* Register allocation (LLVM greedy eviction heuristic) for performance

This document is an outline of the tooling and APIs facilitating MLGO.

Note that tools for orchestrating ML training are not part of LLVM, as they are
dependency-heavy - both on the ML infrastructure choice, as well as choices of
distrubuted computing. For the training scenario, LLVM only contains facilities
enabling it, such as corpus extraction, training data extraction, and evaluation
of models during training.


.. contents::

Corpus Tooling
==============

..
    TODO(boomanaiden154): Write this section.

Interacting with ML models
==========================

We interact with ML models in 2 primary scenarios: one is to train such a model.
The other, inference, is to use a model during compilation, to make optimization
decisions.

For a specific optimization problem - i.e. inlining, or regalloc eviction - we
first separate correctness - preserving decisions from optimization decisions.
For example, not inlining functions marked "no inline" is an example of the
former. Same is not evicting an unevictable live range. An exmple of the latter
is deciding to inline a function that will bloat the caller size, just because
we have reason to believe that later, the effect will be some constant
propagation that will actually reduce the size (or dynamic instruction count).

ML models can be understood as functions. Their inputs are tensors - buffers of
scalars. The output (in our case, singular) is a scalar. For example, for
inlining, the inputs are properties of the caller, callee, and the callsite
being analyzed for inlining. The output is a boolean.

Inputs and outputs are named, have a scalar type (e.g. int32_t) and a shape
(e.g. 3x4). These are the elements that we use to bind to a ML model.

In both training and inference, we want to expose to ML (training algorithms or
trained model, respectively) the features we want to make optimization
decisions on. In that regard, the interface from the compiler side to the ML
side is the same: pass features, and get a decision. It's essentially a function
call, where the parameters and result are bound by name and are described by
name, scalar type, and shape tuples.

The main types in LLVM are:
- ``MLModelRunner`` - an abstraction for the decision making mechanism
- ``TensorSpec`` which describes a tensor.

TensorSpec
----------

See ``llvm/Analysis/TensorSpec.h``. This is a simple data bag, identifying a
tensor by name (a string), scalar type, and shape (a vector of ints). The scalar
type can only be int (8, 16, 32, or 64), signed or unsigned; float; or double.

MLModelRunner
-------------

See ``llvm/Analysis/MLModelRunner.h``. The abstraction has a pure virtual,
``evaluateUntyped``, but the contract with implementers is a bit more involved:

Implementers
^^^^^^^^^^^^

At construction, the implementer is expected to receive a list of ``TensorSpec``
for input features and the ``TensorSpec`` of the output (e.g. 
``std::vector<TensorSpec>``). The list type is not contractual, but it must be
a 0-based indexing array-like container. Given a ``TensorSpec`` at index "I" in
the input list, that has a name "N", shape "D1 x D2x ... Dn", and scalar type
"T", the implementer must:

- set up a contiguous buffer sized ``sizeof(T) * D1 * D2 * ... * Dn``. This
  buffer's lifetime must be the same as the lifetime of the implementer object.
- call ``MLModelRunner::setUpBufferForTensor`` passing I, the ``TensorSpec``,
  and the buffer above.

Internally, the expectation is that the implementer uses the name (and maybe
shape) of a ``TensorSpec`` for binding (e.g. lookup in an underlying ML model).

``MLModelRunner::setUpBufferForTensor`` stores each buffer at the corresponding
index (i.e. its position in the list used at construction). The expectation is
that the user will use that position when calling ``MLModelRunner::getTensor``
to retrieve the underlying buffer (more on that in a bit).

The implementation of ``evaluateUntyped`` is expected to use the value in the
buffers described above, carry out whatever computation (e.g. evaluate a ML
model) and then place the outcome in an output buffer which will be returned to
the caller. Importantly, ``evaluateUntyped`` must not reset the input buffers.
This is because during training we may want to log the features and decisions,
and since the data is already buffered, there's no reason to force backing it
up elsewhere.

Users
^^^^^

The users must pass the input ``TensorSpec`` list at the construction of a
specific ``MLModelRunner`` object. After that, users can be agnostic of the
specific implementation, and would typically follow the following workflow:

- call ``getTensor`` or ``getTensorUntyped``, for each input tensor, identified
  by its index (i.e. the index of the corresponding ``TensorSpec`` in the list
  used at construction).
- populate the tensor buffer of each input tensor with values. Users can take
  advantage of the stability of the tensor buffers like set only once those that
  don't change, or cache the buffer address
- call ``evaluate`` and use its result.

Versioning
^^^^^^^^^^

We support a model "knowing" less inputs than the compiler. This is supported by
``MLModelRunner::setUpBufferForTensor``. If a ``TensorSpec`` requested by the
compiler is not supported by the underlying model, the ``MLModelRunner``
implementer must still call ``setUpBufferForTensor`` with a ``nullptr`` value
for the buffer. In turn, ``MLModelRunner`` will allocate an appropriately - sized
buffer and track its lifetime. The user can safely populate that buffer. Since
the rest of the inputs are still provided, this allows an evolution model where
we first add features to the compiler and continue using older models without
regressing. Then, the new compiler can be used to train new models. Deprecating
features in the compiler involves, then, training first a model without those
features.

``MLModelRunner`` implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We currently feature 3 implementations:

- ``ModelUnderTrainingRunner``. This requires the compiler be built with TFLite
  support. It allows loading a TFLite model dynamically and is primarily
  intended for training scenarios, but it can be used relatively easily in
  production build environments, as it does not change how the compiler operates
  (why this remark is necessary will become clear in a few paragraphs)

- ``ReleaseModeModelRunner``. This is intended for inference scenarios. This
  uses the rules defined in ``llvm/cmake/modules/TensorFlowCompile.cmake`` to
  convert, at the time the compiler is built, TensorFlow Saved Models into a
  header (.h) and native object (.o). The latter is a CPU-based implementation of
  the neural network, together with its weights (essentially, loops performing
  matrix multiplications)

NOTE: we are actively working on replacing this with an EmitC implementation
requiring no out of tree build-time dependencies.

- ``InteractiveModelRunner``. This is intended for training scenarios where the
  training algorithm drives compilation. This model runner has no special
  dependencies, and relies on I/O pipes to communicate with a separate process,
  presumably a python training algorithm. We do not envision using this in a
  production environment.

Note that training leaves it to the training infrastructure to handle
distributed computing. The assumed architecture has python processes
communicating remotely between themselves, but managing local communication with
clang.

..
    TODO(mtrofin): 
        - logging, and the use in interactive mode.
        - discuss an example (like the inliner)
