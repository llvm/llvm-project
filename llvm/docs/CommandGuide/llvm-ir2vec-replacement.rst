llvm-ir2vec - IR2Vec and MIR2Vec Embedding Generation
======================================================

.. program:: llvm-ir2vec

IR2Vec represents LLVM IR as fixed-length numerical vectors (embeddings) by
modelling the relationships between opcodes, types, and operands as a knowledge
graph. These embeddings serve as a compact, learnable representation of program
structure for machine learning-based compiler research and optimization.

This page documents two ways to generate embeddings:

* :ref:`Python bindings <ir2vec-python-bindings>` — the recommended interface
  for ML research workflows.
* :ref:`Command-line tool <ir2vec-cli>` — for scripting, batch processing, and
  vocabulary training data generation.

For information on using IR2Vec programmatically within LLVM passes via the C++
API, see the `IR2Vec Embeddings <https://llvm.org/docs/MLGO.html#ir2vec-embeddings>`_
section in the MLGO documentation.

.. _ir2vec-python-bindings:

Python Bindings
---------------

The IR2Vec Python bindings expose the full embedding API to Python, with no
LLVM build environment required at runtime.

Installation
~~~~~~~~~~~~

Pre-built wheels are available on TestPyPI for Linux (x86_64, aarch64),
macOS (arm64, x86_64), and Windows (x86_64), covering Python 3.10 through
3.13:

.. code-block:: console

   $ pip install ir2vec \
       --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/

.. note::

   The package is currently published on TestPyPI during active development.
   A stable release to PyPI is planned once the API stabilises. The
   ``--extra-index-url`` flag is required so that ``pip`` can resolve
   dependencies (such as ``numpy``) from the main Python Package Index.

Building from Source
^^^^^^^^^^^^^^^^^^^^

The bindings are built as part of the LLVM monorepo. From the repository root:

.. code-block:: console

   $ cmake -G Ninja -B build \
       -DCMAKE_BUILD_TYPE=Release \
       -DLLVM_TARGETS_TO_BUILD=host \
       -DLLVM_IR2VEC_ENABLE_PYTHON_BINDINGS=ON \
       llvm
   $ ninja -C build llvm-ir2vec

The resulting extension module is placed under the build tree. Add the
appropriate ``python_packages/`` subdirectory to ``PYTHONPATH`` to use it
in place.

Quick Start
~~~~~~~~~~~

.. code-block:: python

   import ir2vec

   vocab    = ir2vec.loadVocab(ir2vec.vocab.seedEmbedding75D)
   emb      = ir2vec.initEmbedding(
                  filename="path/to/file.ll",
                  mode=ir2vec.IR2VecKind.FlowAware,
                  vocab=vocab,
              )

   # Function-level embeddings
   func_names   = emb.getFuncNames()
   func_emb_map = emb.getFuncEmbMap()
   func_emb     = emb.getFuncEmb("foo")

   # Sub-function granularity for function "foo"
   bb_map       = emb.getBBEmbMap("foo")
   inst_map     = emb.getInstEmbMap("foo")

The input file must be LLVM IR in either textual (``.ll``) or bitcode
(``.bc``) format. The LLVM IR file can be generated from source using:

.. code-block:: console

   $ clang -O1 -emit-llvm -S -o file.ll file.c

Vocabulary
~~~~~~~~~~

IR2Vec embeddings are grounded in a trained vocabulary — a mapping from each
IR entity (opcode, type, operand kind) to a dense vector, learned by training
a knowledge graph embedding model over a large corpus of LLVM IR. The quality
and dimensionality of this vocabulary directly determines the expressiveness of
the resulting program embeddings.

The package ships one pre-trained seed embedding vocabulary:

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Attribute
     - Dimensions
     - Notes
   * - ``ir2vec.vocab.seedEmbedding75D``
     - 75
     - General-purpose vocabulary trained on a broad LLVM IR corpus.
       Suitable for most downstream ML tasks.

To use a vocabulary trained on a domain-specific corpus, pass the path to a
JSON vocabulary file directly:

.. code-block:: python

   vocab = ir2vec.loadVocab(vocabPath="/path/to/custom_vocab.json")

The vocabulary JSON format matches the output of the ``llvm-ir2vec entities``
and knowledge graph training pipeline described in
:ref:`Vocabulary Training <ir2vec-vocab-training>`.

Embedding Kinds
~~~~~~~~~~~~~~~

The ``mode`` argument to :func:`initEmbedding` selects between two embedding
strategies:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - ``ir2vec.IR2VecKind``
     - Description
   * - ``Symbolic``
     - Each instruction is represented purely by the seed embeddings of its
       opcode, type, and operands, combined using a weighted sum. No
       information flows across instructions. This mode is fast, produces
       deterministic results regardless of program structure, and is
       appropriate when the task depends on the syntactic character of
       instructions rather than their dataflow context.
   * - ``FlowAware``
     - Extends symbolic embeddings by propagating information along def-use
       chains within each basic block, so that an instruction's embedding
       reflects the instructions that produce its operands. This mode
       captures local dataflow relationships and generally produces richer
       representations for tasks that depend on how values are computed.

If you are unsure which to choose, ``FlowAware`` is the stronger default for
most ML training tasks. Use ``Symbolic`` when you need a representation that
is invariant to instruction ordering or when computational cost is a concern.

API Reference
~~~~~~~~~~~~~

**Module-level functions**

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Signature
     - Description
   * - ``ir2vec.loadVocab(vocabPath)``
     - Load a vocabulary from *vocabPath*, which may be either an
       ``ir2vec.vocab`` constant (for bundled vocabularies) or a
       filesystem path string (for custom vocabulary JSON files).
       Returns a vocabulary object to be passed to :func:`initEmbedding`.
   * - ``ir2vec.initEmbedding(filename, mode, vocab)``
     - Parse the LLVM IR file at *filename*, initialise the IR2Vec embedding
       engine with the given *mode* and *vocab*, and return an
       ``Embedding`` object. The file must be in ``.ll`` or ``.bc`` format.

**Embedding object methods**

All methods below operate on the ``Embedding`` object returned by
:func:`initEmbedding`. Embeddings are represented as ``list[float]``.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Signature
     - Description
   * - ``getFuncNames() -> list[str]``
     - Return the names of all functions defined in the module.
   * - ``getFuncEmbMap() -> dict[str, list[float]]``
     - Return a mapping from every function name to its embedding vector.
   * - ``getFuncEmb(func: str) -> list[float]``
     - Return the embedding vector for the function named *func*.
   * - ``getBBEmbMap(func: str) -> dict[str, list[float]]``
     - Return a mapping from basic block labels to their embedding vectors,
       for all basic blocks within function *func*.
   * - ``getInstEmbMap(func: str) -> dict[str, list[float]]``
     - Return a mapping from instruction identifiers to their embedding
       vectors, for all instructions within function *func*.

**ir2vec.IR2VecKind values**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Value
     - Meaning
   * - ``ir2vec.IR2VecKind.Symbolic``
     - Symbolic (seed-only) embeddings.
   * - ``ir2vec.IR2VecKind.FlowAware``
     - Flow-aware embeddings with def-use propagation.

.. _ir2vec-cli:

Command-Line Tool
-----------------

.. program:: llvm-ir2vec

SYNOPSIS
~~~~~~~~

:program:`llvm-ir2vec` [*subcommand*] [*options*]

DESCRIPTION
~~~~~~~~~~~

:program:`llvm-ir2vec` is a standalone command-line tool for generating IR2Vec
and MIR2Vec embeddings and vocabulary training data. It operates on both LLVM
IR bitcode files and Machine IR (``.mir``) files, and supports three
subcommands:

* **embeddings** — generate IR2Vec or MIR2Vec embeddings from a compiled IR
  file using a pre-trained vocabulary.
* **triplets** — generate numeric triplets in train2id format for vocabulary
  training.
* **entities** — generate entity-to-ID mapping files for vocabulary training.

The tool operates in one of two modes, selected with ``--mode``:

* **LLVM IR mode** (``--mode=llvm``, default) — processes LLVM IR bitcode
  files and generates IR2Vec embeddings.
* **Machine IR mode** (``--mode=mir``) — processes Machine IR (``.mir``) files
  and generates MIR2Vec embeddings.

All three subcommands are available in both modes.

.. _ir2vec-vocab-training:

OPERATION MODES
~~~~~~~~~~~~~~~

IR2Vec requires a trained vocabulary to generate embeddings. The vocabulary is
produced by training a knowledge graph embedding model over the *triplets* and
*entity mappings* that describe the structure of LLVM IR. The ``triplets`` and
``entities`` subcommands exist to generate this training data.

The resulting vocabulary JSON file is then consumed by the ``embeddings``
subcommand via ``--ir2vec-vocab-path`` (or ``--mir2vec-vocab-path`` for Machine
IR). For most users, the pre-trained vocabulary shipped with the Python bindings
or available in ``llvm/lib/Analysis/models`` is sufficient — the ``triplets``
and ``entities`` subcommands are relevant primarily when training a custom
vocabulary for a specialised corpus or target.

See ``llvm/utils/mlgo-utils/IR2Vec/generateTriplets.py`` for a reference
implementation of the full training pipeline.

Triplet Generation
^^^^^^^^^^^^^^^^^^

The ``triplets`` subcommand extracts numeric triplets from IR in train2id
format. Each triplet encodes a relationship between two IR entities — for
example, the relationship between an instruction and its type, or between two
sequential instructions. These triplets form the training data for a knowledge
graph embedding model.

The tool outputs numeric entity IDs directly, using the same vocabulary
mapping infrastructure as the ``entities`` subcommand, eliminating the
need for a separate string-to-ID preprocessing step.

Usage for LLVM IR:

.. code-block:: console

   $ llvm-ir2vec triplets --mode=llvm input.bc -o triplets_train2id.txt

Usage for Machine IR:

.. code-block:: console

   $ llvm-ir2vec triplets --mode=mir input.mir -o triplets_train2id.txt

Entity Mapping Generation
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``entities`` subcommand generates the complete set of IR entities
recognised by IR2Vec or MIR2Vec, each assigned a numeric ID, in the standard
format used for knowledge graph embedding training.

For LLVM IR, entities include instruction opcodes, types, and operand kinds.
For Machine IR, entities include machine opcodes, common operands, and register
classes — both physical and virtual. Machine IR entity mappings are
target-specific and therefore vary by architecture.

Usage for LLVM IR:

.. code-block:: console

   $ llvm-ir2vec entities --mode=llvm -o entity2id.txt

Usage for Machine IR:

.. code-block:: console

   $ llvm-ir2vec entities --mode=mir input.mir -o entity2id.txt

.. note::

   For LLVM IR mode, the entity mapping is target-independent and does not
   require an input file. For Machine IR mode, an input ``.mir`` file is
   required so that the tool can determine the target architecture, since
   entity mappings vary by target.

Embedding Generation
^^^^^^^^^^^^^^^^^^^^

The ``embeddings`` subcommand uses a pre-trained vocabulary to generate
numerical embeddings for LLVM IR or Machine IR at instruction, basic block,
or function granularity.

Usage for LLVM IR:

.. code-block:: console

   $ llvm-ir2vec embeddings \
       --mode=llvm \
       --ir2vec-vocab-path=vocab.json \
       --ir2vec-kind=symbolic \
       --level=func \
       input.bc -o embeddings.txt

Usage for Machine IR:

.. code-block:: console

   $ llvm-ir2vec embeddings \
       --mode=mir \
       --mir2vec-vocab-path=vocab.json \
       --level=func \
       input.mir -o embeddings.txt

OPTIONS
~~~~~~~

**Common options**

.. option:: --mode=<mode>

   Specify the operation mode. Valid values are:

   * ``llvm`` — process LLVM IR bitcode files (default).
   * ``mir`` — process Machine IR (``.mir``) files.

.. option:: -o <filename>

   Specify the output filename. Defaults to standard output (``-``).

.. option:: --help

   Print a summary of command-line options.

**embeddings subcommand**

.. option:: <input-file>

   The input LLVM IR/bitcode file (``.ll``/``.bc``) or Machine IR file
   (``.mir``) to process. Required for the ``embeddings`` subcommand.

.. option:: --level=<level>

   Specify the embedding granularity. Valid values are:

   * ``inst`` — instruction-level embeddings.
   * ``bb`` — basic block-level embeddings.
   * ``func`` — function-level embeddings (default).

.. option:: --function=<name>

   Process only the specified function rather than all functions in the
   module.

**IR2Vec-specific options** (``--mode=llvm``)

.. option:: --ir2vec-kind=<kind>

   Specify the IR2Vec embedding strategy. Valid values are:

   * ``symbolic`` — symbolic embeddings (default). Each instruction is
     represented by the seed embeddings of its opcode, type, and operands,
     combined using a weighted sum.
   * ``flow-aware`` — flow-aware embeddings. Extends symbolic embeddings by
     propagating information along def-use chains within each basic block.

.. option:: --ir2vec-vocab-path=<path>

   Path to the IR2Vec vocabulary JSON file. Required for LLVM IR embedding
   generation. Pre-trained vocabulary files are available in
   ``llvm/lib/Analysis/models``.

.. option:: --ir2vec-opc-weight=<weight>

   Weight applied to opcode embeddings when computing the weighted sum.
   Default: ``1.0``.

.. option:: --ir2vec-type-weight=<weight>

   Weight applied to type embeddings when computing the weighted sum.
   Default: ``0.5``.

.. option:: --ir2vec-arg-weight=<weight>

   Weight applied to operand embeddings when computing the weighted sum.
   Default: ``0.2``.

**MIR2Vec-specific options** (``--mode=mir``)

.. option:: --mir2vec-vocab-path=<path>

   Path to the MIR2Vec vocabulary JSON file. Required for Machine IR
   embedding generation.

.. option:: --mir2vec-kind=<kind>

   Specify the MIR2Vec embedding strategy. Valid values are:

   * ``symbolic`` — symbolic embeddings (default).

.. option:: --mir2vec-opc-weight=<weight>

   Weight applied to machine opcode embeddings. Default: ``1.0``.

.. option:: --mir2vec-common-operand-weight=<weight>

   Weight applied to common operand embeddings (e.g., immediate values,
   frame indices). Default: ``1.0``.

.. option:: --mir2vec-reg-operand-weight=<weight>

   Weight applied to register operand embeddings. Default: ``1.0``.

**triplets subcommand**

.. option:: <input-file>

   The input LLVM IR/bitcode file (``.ll``/``.bc``) or Machine IR file
   (``.mir``) to process. Required.

**entities subcommand**

.. option:: <input-file>

   The input Machine IR file (``.mir``) to process. Required when using
   ``--mode=mir``, as entity mappings are target-specific. Not required for
   ``--mode=llvm``, since LLVM IR entity mappings are target-independent.

OUTPUT FORMAT
~~~~~~~~~~~~~

Triplet Mode Output
^^^^^^^^^^^^^^^^^^^

The output is a sequence of numeric triplets in train2id format, preceded by a
metadata header:

.. code-block:: text

   MAX_RELATION=<max_relation_id>
   <head_entity_id> <tail_entity_id> <relation_id>
   <head_entity_id> <tail_entity_id> <relation_id>
   ...

Each line after the header represents one instruction relationship. The
``MAX_RELATION`` header records the highest relation ID present in the file.

**Relation types for LLVM IR (IR2Vec):**

* ``0`` — type relationship (instruction to its result type).
* ``1`` — next relationship (sequential instruction ordering).
* ``2+`` — argument relationships (Arg0, Arg1, Arg2, …).

**Relation types for Machine IR (MIR2Vec):**

* ``0`` — next relationship (sequential instruction ordering).
* ``1+`` — argument relationships (Arg0, Arg1, Arg2, …).

Entity Mode Output
^^^^^^^^^^^^^^^^^^

.. code-block:: text

   <total_entity_count>
   <entity_string>	<numeric_id>
   <entity_string>	<numeric_id>
   ...

The first line is the total number of entities. Each subsequent line is a
tab-separated pair of an entity string and its assigned numeric ID.

For LLVM IR, entities include instruction opcodes (e.g., ``Add``, ``Ret``),
types (e.g., ``INT``, ``PTR``), and operand kinds.

For Machine IR, entities include machine opcodes (e.g., ``COPY``, ``ADD``),
common operands (e.g., ``Immediate``, ``FrameIndex``), physical register
classes (e.g., ``PhyReg_GR32``), and virtual register classes
(e.g., ``VirtReg_GR32``). The full entity set is target-specific.

Embedding Mode Output
^^^^^^^^^^^^^^^^^^^^^

The output format depends on the ``--level`` specified:

* **Function level** — one embedding vector per function.
* **Basic block level** — one embedding vector per basic block, grouped by
  function.
* **Instruction level** — one embedding vector per instruction, grouped by
  basic block and function.

Each embedding is a space-separated sequence of floating-point values.

EXIT STATUS
~~~~~~~~~~~

:program:`llvm-ir2vec` returns 0 on success and a non-zero value on failure.

Common failure conditions:

* Invalid or missing input file.
* Missing or invalid vocabulary file (``embeddings`` subcommand).
* Specified function not found in the module (``--function`` option).
* Invalid command-line options.

SEE ALSO
--------

:doc:`../MLGO`
   C++ API and pass-level integration of IR2Vec within LLVM optimization
   pipelines.

`IR2Vec: LLVM IR Based Scalable Program Embeddings <https://doi.org/10.1145/3418463>`_
   The original IR2Vec paper describing the algorithm, vocabulary training
   procedure, and evaluation.

`RL4ReAl: Reinforcement Learning for Register Allocation <https://doi.org/10.1145/3578360.3580273>`_
   The paper describing MIR2Vec and its application to register allocation.
