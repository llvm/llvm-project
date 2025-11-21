llvm-ir2vec - IR2Vec and MIR2Vec Embedding Generation Tool
==========================================================

.. program:: llvm-ir2vec

SYNOPSIS
--------

:program:`llvm-ir2vec` [*subcommand*] [*options*]

DESCRIPTION
-----------

:program:`llvm-ir2vec` is a standalone command-line tool for IR2Vec and MIR2Vec.
It generates embeddings for both LLVM IR and Machine IR (MIR) and supports 
triplet generation for vocabulary training. 

The tool provides three main subcommands:

1. **triplets**: Generates numeric triplets in train2id format for vocabulary
   training from LLVM IR.

2. **entities**: Generates entity mapping files (entity2id.txt) for vocabulary 
   training.

3. **embeddings**: Generates IR2Vec or MIR2Vec embeddings using a trained vocabulary
   at different granularity levels (instruction, basic block, or function).

The tool supports two operation modes:

* **LLVM IR mode** (``--mode=llvm``): Process LLVM IR bitcode files and generate
  IR2Vec embeddings
* **Machine IR mode** (``--mode=mir``): Process Machine IR (.mir) files and generate
  MIR2Vec embeddings

The tool is designed to facilitate machine learning applications that work with
LLVM IR or Machine IR by converting them into numerical representations that can 
be used by ML models. The `triplets` subcommand generates numeric IDs directly 
instead of string triplets, streamlining the training data preparation workflow.

.. note::

   For information about using IR2Vec and MIR2Vec programmatically within LLVM 
   passes and the C++ API, see the `IR2Vec Embeddings <https://llvm.org/docs/MLGO.html#ir2vec-embeddings>`_ 
   section in the MLGO documentation.

OPERATION MODES
---------------

The tool operates in two modes: **LLVM IR mode** and **Machine IR mode**. The mode
is selected using the ``--mode`` option (default: ``llvm``).

Triplet Generation and Entity Mapping Modes are used for preparing
vocabulary and training data for knowledge graph embeddings. The Embedding Mode
is used for generating embeddings from LLVM IR using a pre-trained vocabulary.

The Seed Embedding Vocabulary of IR2Vec is trained on a large corpus of LLVM IR
by modeling the relationships between opcodes, types, and operands as a knowledge
graph. For this purpose, Triplet Generation and Entity Mapping Modes generate
triplets and entity mappings in the standard format used for knowledge graph
embedding training (see 
<https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch?tab=readme-ov-file#data-format> 
for details).

See `llvm/utils/mlgo-utils/IR2Vec/generateTriplets.py` for more details on how
these two modes are used to generate the triplets and entity mappings.

Triplet Generation
~~~~~~~~~~~~~~~~~~

With the `triplets` subcommand, :program:`llvm-ir2vec` analyzes LLVM IR or Machine IR
and extracts numeric triplets consisting of opcode IDs and operand IDs. These triplets
are generated in the standard format used for knowledge graph embedding training.
The tool outputs numeric IDs directly using the vocabulary mapping infrastructure,
eliminating the need for string-to-ID preprocessing.

Usage for LLVM IR:

.. code-block:: bash

   llvm-ir2vec triplets --mode=llvm input.bc -o triplets_train2id.txt

Usage for Machine IR:

.. code-block:: bash

   llvm-ir2vec triplets --mode=mir input.mir -o triplets_train2id.txt

Entity Mapping Generation
~~~~~~~~~~~~~~~~~~~~~~~~~

With the `entities` subcommand, :program:`llvm-ir2vec` generates the entity mappings
supported by IR2Vec or MIR2Vec in the standard format used for knowledge graph embedding
training. This subcommand outputs all supported entities with their corresponding numeric IDs.

For LLVM IR, entities include opcodes, types, and operands. For Machine IR, entities include
machine opcodes, common operands, and register classes (both physical and virtual).

Usage for LLVM IR:

.. code-block:: bash

   llvm-ir2vec entities --mode=llvm -o entity2id.txt

Usage for Machine IR:

.. code-block:: bash

   llvm-ir2vec entities --mode=mir input.mir -o entity2id.txt

.. note::

   For LLVM IR mode, the entity mapping is target-independent and does not require an input file.
   For Machine IR mode, an input .mir file is required to determine the target architecture,
   as entity mappings vary by target (different architectures have different instruction sets
   and register classes).

Embedding Generation
~~~~~~~~~~~~~~~~~~~~

With the `embeddings` subcommand, :program:`llvm-ir2vec` uses a pre-trained vocabulary to
generate numerical embeddings for LLVM IR or Machine IR at different levels of granularity.

Example Usage for LLVM IR:

.. code-block:: bash

   llvm-ir2vec embeddings --mode=llvm --ir2vec-vocab-path=vocab.json --ir2vec-kind=symbolic --level=func input.bc -o embeddings.txt

Example Usage for Machine IR:

.. code-block:: bash

   llvm-ir2vec embeddings --mode=mir --mir2vec-vocab-path=vocab.json --level=func input.mir -o embeddings.txt

OPTIONS
-------

Common options (applicable to both LLVM IR and Machine IR modes):

.. option:: --mode=<mode>

   Specify the operation mode. Valid values are:

   * ``llvm`` - Process LLVM IR bitcode files (default)
   * ``mir`` - Process Machine IR (.mir) files

.. option:: -o <filename>

   Specify the output filename. Use ``-`` to write to standard output (default).

.. option:: --help

   Print a summary of command line options.

Subcommand-specific options:

**embeddings** subcommand:

.. option:: <input-file>

   The input LLVM IR/bitcode file (.ll/.bc) or Machine IR file (.mir) to process. 
   This positional argument is required for the `embeddings` subcommand.

.. option:: --level=<level>

   Specify the embedding generation level. Valid values are:

   * ``inst`` - Generate instruction-level embeddings
   * ``bb`` - Generate basic block-level embeddings  
   * ``func`` - Generate function-level embeddings (default)

.. option:: --function=<name>

   Process only the specified function instead of all functions in the module.

**IR2Vec-specific options** (for ``--mode=llvm``):

.. option:: --ir2vec-kind=<kind>

   Specify the kind of IR2Vec embeddings to generate. Valid values are:

   * ``symbolic`` - Generate symbolic embeddings (default)
   * ``flow-aware`` - Generate flow-aware embeddings

   Flow-aware embeddings consider control flow relationships between instructions,
   while symbolic embeddings focus on the symbolic representation of instructions.

.. option:: --ir2vec-vocab-path=<path>

   Specify the path to the IR2Vec vocabulary file (required for LLVM IR embedding 
   generation). The vocabulary file should be in JSON format and contain the trained
   vocabulary for embedding generation. See `llvm/lib/Analysis/models`
   for pre-trained vocabulary files.

.. option:: --ir2vec-opc-weight=<weight>

   Specify the weight for opcode embeddings (default: 1.0). This controls
   the relative importance of instruction opcodes in the final embedding.

.. option:: --ir2vec-type-weight=<weight>

   Specify the weight for type embeddings (default: 0.5). This controls
   the relative importance of type information in the final embedding.

.. option:: --ir2vec-arg-weight=<weight>

   Specify the weight for argument embeddings (default: 0.2). This controls
   the relative importance of operand information in the final embedding.

**MIR2Vec-specific options** (for ``--mode=mir``):

.. option:: --mir2vec-vocab-path=<path>

   Specify the path to the MIR2Vec vocabulary file (required for Machine IR 
   embedding generation). The vocabulary file should be in JSON format and 
   contain the trained vocabulary for embedding generation.

.. option:: --mir2vec-kind=<kind>

   Specify the kind of MIR2Vec embeddings to generate. Valid values are:

   * ``symbolic`` - Generate symbolic embeddings (default)

.. option:: --mir2vec-opc-weight=<weight>

   Specify the weight for machine opcode embeddings (default: 1.0). This controls
   the relative importance of machine instruction opcodes in the final embedding.

.. option:: --mir2vec-common-operand-weight=<weight>

   Specify the weight for common operand embeddings (default: 1.0). This controls
   the relative importance of common operand types in the final embedding.

.. option:: --mir2vec-reg-operand-weight=<weight>

   Specify the weight for register operand embeddings (default: 1.0). This controls
   the relative importance of register operands in the final embedding.


**triplets** subcommand:

.. option:: <input-file>

   The input LLVM IR/bitcode file (.ll/.bc) or Machine IR file (.mir) to process. 
   This positional argument is required for the `triplets` subcommand.

**entities** subcommand:

.. option:: <input-file>

   The input Machine IR file (.mir) to process. This positional argument is required
   for the `entities` subcommand when using ``--mode=mir``, as the entity mappings
   are target-specific. For ``--mode=llvm``, no input file is required as IR2Vec
   entity mappings are target-independent.

OUTPUT FORMAT
-------------

Triplet Mode Output
~~~~~~~~~~~~~~~~~~~

In triplet mode, the output consists of numeric triplets in train2id format with
metadata headers. The format includes:

.. code-block:: text

   MAX_RELATION=<max_relation_count>
   <head_entity_id> <tail_entity_id> <relation_id>
   <head_entity_id> <tail_entity_id> <relation_id>
   ...

Each line after the metadata header represents one instruction relationship,
with numeric IDs for head entity, tail entity, and relation type. The metadata 
header (MAX_RELATION) indicates the maximum relation ID used.

**Relation Types:**

For LLVM IR (IR2Vec):
  * **0** = Type relationship (instruction to its type)
  * **1** = Next relationship (sequential instructions)
  * **2+** = Argument relationships (Arg0, Arg1, Arg2, ...)

For Machine IR (MIR2Vec):
  * **0** = Next relationship (sequential instructions)
  * **1+** = Argument relationships (Arg0, Arg1, Arg2, ...)

**Entity IDs:**

For LLVM IR: Entity IDs represent opcodes, types, and operands as defined by the IR2Vec vocabulary.

For Machine IR: Entity IDs represent machine opcodes, common operands (immediate, frame index, etc.),
physical register classes, and virtual register classes as defined by the MIR2Vec vocabulary. The entity layout is target-specific.

Entity Mode Output
~~~~~~~~~~~~~~~~~~

In entity mode, the output consists of entity mappings in the format:

.. code-block:: text

   <total_entities>
   <entity_string>	<numeric_id>
   <entity_string>	<numeric_id>
   ...

The first line contains the total number of entities, followed by one entity
mapping per line with tab-separated entity string and numeric ID.

For LLVM IR, entities include instruction opcodes (e.g., "Add", "Ret"), types 
(e.g., "INT", "PTR"), and operand kinds.

For Machine IR, entities include machine opcodes (e.g., "COPY", "ADD"), 
common operands (e.g., "Immediate", "FrameIndex"), physical register classes 
(e.g., "PhyReg_GR32"), and virtual register classes (e.g., "VirtReg_GR32").

Embedding Mode Output
~~~~~~~~~~~~~~~~~~~~~

In embedding mode, the output format depends on the specified level:

* **Function Level**: One embedding vector per function
* **Basic Block Level**: One embedding vector per basic block, grouped by function
* **Instruction Level**: One embedding vector per instruction, grouped by basic block and function

Each embedding is represented as a floating point vector.

EXIT STATUS
-----------

:program:`llvm-ir2vec` returns 0 on success, and a non-zero value on failure.

Common failure cases include:

* Invalid or missing input file
* Missing or invalid vocabulary file (in embedding mode)
* Specified function not found in the module
* Invalid command line options

SEE ALSO
--------

:doc:`../MLGO`

For more information about the IR2Vec algorithm and approach, see:
`IR2Vec: LLVM IR Based Scalable Program Embeddings <https://doi.org/10.1145/3418463>`_.

For more information about the MIR2Vec algorithm and approach, see:
`RL4ReAl: Reinforcement Learning for Register Allocation <https://doi.org/10.1145/3578360.3580273>`_.
