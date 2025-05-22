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
distributed computing. For the training scenario, LLVM only contains facilities
enabling it, such as corpus extraction, training data extraction, and evaluation
of models during training.


.. contents::

Corpus Tooling
==============

Within the LLVM monorepo, there is the ``mlgo-utils`` python packages that
lives at ``llvm/utils/mlgo-utils``. This package primarily contains tooling
for working with corpora, or collections of LLVM bitcode. We use these corpora
to train and evaluate ML models. Corpora consist of a description in JSON
format at ``corpus_description.json`` in the root of the corpus, and then
a bitcode file and command line flags file for each extracted module. The
corpus structure is designed to contain sufficient information to fully
compile the bitcode to bit-identical object files.

.. program:: extract_ir.py

Synopsis
--------

Extracts a corpus from some form of a structured compilation database. This
tool supports a variety of different scenarios and input types.

Options
-------

.. option:: --input

  The path to the input. This should be a path to a supported structured
  compilation database. Currently only ``compile_commands.json`` files, linker
  parameter files, a directory containing object files (for the local
  ThinLTO case only), or a JSON file containing a bazel aquery result are
  supported.

.. option:: --input_type

  The type of input that has been passed to the ``--input`` flag.

.. option:: --output_dir

  The output directory to place the corpus in.

.. option:: --num_workers

  The number of workers to use for extracting bitcode into the corpus. This
  defaults to the number of hardware threads available on the host system.

.. option:: --llvm_objcopy_path

  The path to the llvm-objcopy binary to use when extracting bitcode.

.. option:: --obj_base_dir

  The base directory for object files. Bitcode files that get extracted into
  the corpus will be placed into the output directory based on where their
  source object files are placed relative to this path.

.. option:: --cmd_filter

  Allows filtering of modules by command line. If set, only modules that much
  the filter will be extracted into the corpus. Regular expressions are
  supported in some instances.

.. option:: --thinlto_build

  If the build was performed with ThinLTO, this should be set to either
  ``distributed`` or ``local`` depending upon how the build was performed.

.. option:: --cmd_section_name

  This flag allows specifying the command line section name. This is needed
  on non-ELF platforms where the section name might differ.

.. option:: --bitcode_section_name

  This flag allows specifying the bitcode section name. This is needed on
  non-ELF platforms where the section name might differ.

Example: CMake
--------------

CMake can output a ``compilation_commands.json`` compilation database if the
``CMAKE_EXPORT_COMPILE_COMMANDS`` switch is turned on at compile time. It is
also necessary to enable bitcode embedding (done by passing 
``-Xclang -fembed-bitcode=all`` to all C/C++ compilation actions in the
non-ThinLTO case). For example, to extract a corpus from clang, you would
run the following commands (assuming that the system C/C++ compiler is clang):

.. code-block:: bash

  cmake -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_C_FLAGS="-Xclang -fembed-bitcode=all" \
    -DCMAKE_CXX_FLAGS="-Xclang -fembed-bitcode-all"
    ../llvm
  ninja

After running CMake and building the project, there should be a
 ``compilation_commands.json`` file within the build directory. You can then
 run the following command to create a corpus:

.. code-block:: bash

  python3 ./extract_ir.py \
    --input=./build/compile_commands.json \
    --input_type=json \
    --output_dir=./corpus

After running the above command, there should be a full
corpus of bitcode within the ``./corpus`` directory.

Example: Bazel Aquery
---------------------

This tool also supports extracting bitcode from bazel in multiple ways
depending upon the exact configuration. For ThinLTO, a linker parameters file
is preferred. For the non-ThinLTO case, the script will accept the output of
``bazel aquery`` which it will use to find all the object files that are linked
into a specific target and then extract bitcode from them. First, you need
to generate the aquery output:

.. code-block:: bash

  bazel aquery --output=jsonproto //path/to:target > /path/to/aquery.json

Afterwards, assuming that the build is already complete, you can run this
script to create a corpus:

.. code-block:: bash

  python3 ./extract_ir.py \
    --input=/path/to/aquery.json \
    --input_type=bazel_aqeury \
    --output_dir=./corpus \
    --obj_base_dir=./bazel-bin

This will again leave a corpus that contains all the bitcode files. This mode
does not capture all object files in the build however, only the ones that
are involved in the link for the binary passed to the ``bazel aquery``
invocation.

.. program:: make_corpus.py

Synopsis
--------

Creates a corpus from a collection of bitcode files.

Options
-------

.. option:: --input_dir

  The input directory to search for bitcode files in.

.. option:: --output_dir

  The output directory to place the constructed corpus in.

.. option:: --default_args

  A list of space separated flags that are put into the corpus description.
  These are used by some tooling when compiling the modules within the corpus.

.. program:: combine_training_corpus.py

Synopsis
--------

Combines two training corpora that share the same parent folder by generating
a new ``corpus_description.json`` that contains all the modules in both corpora.

Options
-------

.. option:: --root_dir

  The root directory that contains subfolders consisting of the corpora that
  should be combined.

Interacting with ML models
==========================

We interact with ML models in 2 primary scenarios: one is to train such a model.
The other, inference, is to use a model during compilation, to make optimization
decisions.

For a specific optimization problem - i.e. inlining, or regalloc eviction - we
first separate correctness - preserving decisions from optimization decisions.
For example, not inlining functions marked "no inline" is an example of the
former. Same is not evicting an unevictable live range. An example of the latter
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

We currently feature 4 implementations:

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

- ``NoInferenceModelRunner``. This serves as a store for feature values, and its
  ``evaluate`` should never be called. It's used for training scenarios, when we
  want to capture the behavior of the default (non-ML) heuristic.

Note that training leaves it to the training infrastructure to handle
distributed computing. The assumed architecture has python processes
communicating remotely between themselves, but managing local communication with
clang.

Logging Facility
----------------

When training models, we need to expose the features we will want to use during
inference, as well as outcomes, to guide reward-based learning techniques. This
can happen in 2 forms:

- when running the compiler on some input, as a capture of the features and
  actions taken by some policy or a model currently being used.
  For example, see ``DevelopmentModeInlineAdvisor`` or ``DevelopmentModeEvictAdvisor``
  in ``MLRegallocEvictAdvisor.cpp``. In more detail, in the former case, if
  ``-training-log`` is specified, the features and actions (inline/no inline)
  from each inlining decision are saved to the specified file. Since
  ``MLModelRunner`` implementations hold on to feature values (they don't get
  cleared by ``evaluate``), logging is easily supported by just looping over the
  model runner's features and passing the tensor buffers to the logger. Note how
  we use the ``NoInferenceModelRunner`` to capture the features observed when
  using the default policy.

- as a serialization mechanism for the ``InteractiveModelRunner``. Here, we need
  to pass the observed features over IPC (a file descriptor, likely a named
  pipe).

Both cases require serializing the same kind of data and we support both with
``Analysis/Utils/TrainingLogger``.

The goal of the logger design was avoiding any new dependency, and optimizing
for the tensor scenario - i.e. exchanging potentially large buffers of fixed
size, containing scalars. We explicitly assume the reader of the format has the
same endianness as the compiler host, and we further expect the reader and the
compiler run on the same host. This is because we expect the training scenarios
have a (typically python) process managing the compiler process, and we leave to
the training side to handle remoting.

The logger produces the following sequence:

- a header describing the structure of the log. This is a one-line textual JSON
  dictionary with the following elements:
  
  - ``features``: a list of JSON-serialized ``TensorSpec`` values. The position
    in the list matters, as it will be the order in which values will be
    subsequently recorded. If we are just logging (i.e. not using the
    ``InteractiveModelRunner``), the last feature should be that of the action
    (e.g. "inline/no inline", or "index of evicted live range")
  - (optional) ``score``: a ``TensorSpec`` describing a value we will include to
    help formulate a reward. This could be a size estimate or a latency estimate.
  - (optional) ``advice``: a ``TensorSpec`` describing the action. This is used
    for the ``InteractiveModelRunner``, in which case it shouldn't be in the 
    ``features`` list.
- a sequence of ``contexts``. Contexts are independent traces of the optimization
  problem. For module passes, there is only one context, for function passes,
  there is a context per function. The start of a context is marked with a
  one-line JSON dictionary of the form ``{"context": <context name, a string>}``
  
  Each context has a sequence of:

  - ``observations``. An observation is:
    
    - one-line JSON ``{"observation": <observation number. 0-indexed>}``
    - a binary dump of the tensor buffers, in the order in which they were
      specified in the header.
    - a new line character
    - if ``score`` was specified in the header:
    
      - a one-line JSON object ``{"outcome": <value>}``, where the ``value``
        conforms to the ``TensorSpec`` in defined for the ``score`` in the header.
      - the outcome value, as a binary dump
      - a new line character.

The format uses a mix of textual JSON (for headers) and binary dumps (for tensors)
because the headers are not expected to dominate the payload - the tensor values
are. We wanted to avoid overburdening the log reader - likely python - from
additional dependencies; and the one-line JSON makes it rudimentarily possible
to inspect a log without additional tooling.

A python utility for reading logs, used for tests, is available at
``Analysis/models/log_reader.py``. A utility showcasing the ``InteractiveModelRunner``,
which uses this reader as well, is at ``Analysis/models/interactive_host.py``.
The latter is also used in tests.

There is no C++ implementation of a log reader. We do not have a scenario
motivating one.

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

The core components are:
  - **Vocabulary**: A mapping from IR entities (opcodes, types, etc.) to their
    vector representations. This is managed by ``IR2VecVocabAnalysis``.
  - **Embedder**: A class (``ir2vec::Embedder``) that uses the vocabulary to
    compute embeddings for instructions, basic blocks, and functions.

Using IR2Vec
------------

For generating embeddings, first the vocabulary should be obtained. Then, the 
embeddings can be computed and accessed via an ``ir2vec::Embedder`` instance.

1. **Get the Vocabulary**:
   In a ModulePass, get the vocabulary analysis result:

   .. code-block:: c++

      auto &VocabRes = MAM.getResult<IR2VecVocabAnalysis>(M);
      if (!VocabRes.isValid()) {
        // Handle error: vocabulary is not available or invalid
        return;
      }
      const ir2vec::Vocab &Vocabulary = VocabRes.getVocabulary();
      unsigned Dimension = VocabRes.getDimension();

    Note that ``IR2VecVocabAnalysis`` pass is immutable.

2. **Create Embedder instance**:
   With the vocabulary, create an embedder for a specific function:

   .. code-block:: c++

      // Assuming F is an llvm::Function&
      // For example, using IR2VecKind::Symbolic:
      Expected<std::unique_ptr<ir2vec::Embedder>> EmbOrErr =
          ir2vec::Embedder::create(IR2VecKind::Symbolic, F, Vocabulary, Dimension);

      if (auto Err = EmbOrErr.takeError()) {
        // Handle error in embedder creation
        return;
      }
      std::unique_ptr<ir2vec::Embedder> Emb = std::move(*EmbOrErr);

3. **Compute and Access Embeddings**:
   Call ``computeEmbeddings()`` on the embedder instance to compute the 
   embeddings. Then the embeddings can be accessed using different getter 
   methods. Currently, ``Embedder`` can generate embeddings at three levels:
   Instructions, Basic Blocks, and Functions.

   .. code-block:: c++

      Emb->computeEmbeddings();
      const ir2vec::Embedding &FuncVector = Emb->getFunctionVector();
      const ir2vec::InstEmbeddingsMap &InstVecMap = Emb->getInstVecMap();
      const ir2vec::BBEmbeddingsMap &BBVecMap = Emb->getBBVecMap();

      // Example: Iterate over instruction embeddings
      for (const auto &Entry : InstVecMap) {
        const Instruction *Inst = Entry.getFirst();
        const ir2vec::Embedding &InstEmbedding = Entry.getSecond();
        // Use Inst and InstEmbedding
      }

4. **Working with Embeddings:**
   Embeddings are represented as ``std::vector<double>``. These
   vectors as features for machine learning models, compute similarity scores
   between different code snippets, or perform other analyses as needed.

Further Details
---------------

For more detailed information about the IR2Vec algorithm, its parameters, and
advanced usage, please refer to the original paper:
`IR2Vec: LLVM IR Based Scalable Program Embeddings <https://doi.org/10.1145/3418463>`_.
The LLVM source code for ``IR2Vec`` can also be explored to understand the 
implementation details.
