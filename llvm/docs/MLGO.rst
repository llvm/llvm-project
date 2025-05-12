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

IR2Vec Embeddings
=================

IR2Vec is a program embedding approach designed specifically for LLVM IR. It
is implemented as a function analysis pass in LLVM. The IR2Vec embeddings
capture syntactic, semantic, and structural properties of the IR through 
learned representations. These representations are obtained as a JSON 
vocabulary that maps the entities of the IR (opcodes, types, operands) to 
n-dimensional floating point vectors (embeddings). 

With IR2Vec, representation at different granularities of IR, such as
instructions, functions, and basic blocks, can be obtained. Representations 
of loops and regions can be derived from these representations, which can be
useful in different scenarios. The representations can be useful for various
downstream tasks, including ML-guided compiler optimizations.

Currently, to use IR2Vec embeddings, the JSON vocabulary first needs to be read
and used to obtain the vocabulary mapping. Then, use this mapping to
derive the representations. In LLVM, this process is implemented using two
independent passes: ``IR2VecVocabAnalysis`` and ``IR2VecAnalysis``. The former
reads the JSON vocabulary and populates ``IR2VecVocabResult``, which is then used
by ``IR2VecAnalysis``. 

``IR2VecVocabAnalysis`` is immutable and is intended to
be run once before ``IR2VecAnalysis`` is run. In the future, we plan
to improve this requirement by automatically generating default the vocabulary mappings
during build time, eliminating the need for a separate file read.

IR2VecAnalysis Usage
--------------------

To use IR2Vec in an LLVM-based tool or pass, interaction with the analysis 
results can be done through the following APIs:
    
1. **Accessing the Analysis Results:**

   To access the IR2Vec embeddings, obtain the ``IR2VecAnalysis``
   result from the Function Analysis Manager (FAM).

   .. code-block:: c++

      #include "llvm/Analysis/IR2VecAnalysis.h"

      // ... other includes and code ...

      llvm::FunctionAnalysisManager &FAM = ...; // The FAM instance
      llvm::Function &F = ...; // The function to analyze
      auto &IR2VecResult = FAM.getResult<llvm::IR2VecAnalysis>(F);

2. **Checking for Valid Results:**

   Ensure that the analysis result is valid before accessing the embeddings:

   .. code-block:: c++

      if (IR2VecResult.isValid()) {
        // Proceed to access embeddings
      }

3. **Retrieving Embeddings:**

   The ``IR2VecResult`` provides access to embeddings (currently) at three levels:

   - **Instruction Embeddings:**

     .. code-block:: c++

        const auto &instVecMap = IR2VecResult.getInstVecMap();
        // instVecMap is a SmallMapVector<const Instruction*, ir2vec::Embedding, 128>
        for (const auto &it : instVecMap) {
          const Instruction *I = it.first;
          const ir2vec::Embedding &embedding = it.second;
          // Use the instruction embedding
        }
   - **Basic Block Embeddings:**

     .. code-block:: c++

        const auto &bbVecMap = IR2VecResult.getBBVecMap();
        // bbVecMap is a SmallMapVector<const BasicBlock*, ir2vec::Embedding, 16>
        for (const auto &it : bbVecMap) {
          const BasicBlock *BB = it.first;
          const ir2vec::Embedding &embedding = it.second;
          // Use the basic block embedding
        }
   - **Function Embedding:**

     .. code-block:: c++

        const ir2vec::Embedding &funcEmbedding = IR2VecResult.getFunctionVector();
        // Use the function embedding

4. **Working with Embeddings:**

   Embeddings are represented as ``std::vector<double>``. These
   vectors as features for machine learning models, compute similarity scores
   between different code snippets, or perform other analyses as needed.

Example Usage
^^^^^^^^^^^^^

.. code-block:: c++

   #include "llvm/Analysis/IR2VecAnalysis.h"
   #include "llvm/IR/Function.h"
   #include "llvm/IR/Instructions.h"
   #include "llvm/Passes/PassBuilder.h"

   // ... other includes and code ...

   void processFunction(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
     auto &IR2VecResult = FAM.getResult<llvm::IR2VecAnalysis>(F);

     if (IR2VecResult.isValid()) {
       const auto &instVecMap = IR2VecResult.getInstVecMap();
       for (const auto &it : instVecMap) {
         const Instruction *I = it.first;
         const auto &embedding = it.second;
         llvm::errs() << "Instruction: " << *I << "\n";
         llvm::errs() << "Embedding: ";
         for (double val : embedding) {
           llvm::errs() << val << " ";
         }
         llvm::errs() << "\n";
       }
     } else {
       llvm::errs() << "IR2Vec analysis failed for function " << F.getName() << "\n";
     }
   }

   // ... rest of the pass ...

   // In the pass's run method:
   // processFunction(F, FAM);

Further Details
---------------

For more detailed information about the IR2Vec algorithm, its parameters, and
advanced usage, please refer to the original paper:
`IR2Vec: LLVM IR Based Scalable Program Embeddings <https://doi.org/10.1145/3418463>`_.
The LLVM source code for ``IR2VecAnalysis`` can also be explored to understand the 
implementation details.
