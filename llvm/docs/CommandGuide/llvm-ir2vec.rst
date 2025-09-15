llvm-ir2vec - IR2Vec Embedding Generation Tool
==============================================

.. program:: llvm-ir2vec

SYNOPSIS
--------

:program:`llvm-ir2vec` [*subcommand*] [*options*]

DESCRIPTION
-----------

:program:`llvm-ir2vec` is a standalone command-line tool for IR2Vec. It
generates IR2Vec embeddings for LLVM IR and supports triplet generation 
for vocabulary training. 

The tool provides three main subcommands:

1. **triplets**: Generates numeric triplets in train2id format for vocabulary
   training from LLVM IR.

2. **entities**: Generates entity mapping files (entity2id.txt) for vocabulary 
   training.

3. **embeddings**: Generates IR2Vec embeddings using a trained vocabulary
   at different granularity levels (instruction, basic block, or function).

The tool is designed to facilitate machine learning applications that work with
LLVM IR by converting the IR into numerical representations that can be used by
ML models. The `triplets` subcommand generates numeric IDs directly instead of string 
triplets, streamlining the training data preparation workflow.

.. note::

   For information about using IR2Vec programmatically within LLVM passes and 
   the C++ API, see the `IR2Vec Embeddings <https://llvm.org/docs/MLGO.html#ir2vec-embeddings>`_ 
   section in the MLGO documentation.

OPERATION MODES
---------------

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

With the `triplets` subcommand, :program:`llvm-ir2vec` analyzes LLVM IR and extracts
numeric triplets consisting of opcode IDs, type IDs, and operand IDs. These triplets
are generated in the standard format used for knowledge graph embedding training.
The tool outputs numeric IDs directly using the ir2vec::Vocabulary mapping
infrastructure, eliminating the need for string-to-ID preprocessing.

Usage:

.. code-block:: bash

   llvm-ir2vec triplets input.bc -o triplets_train2id.txt

Entity Mapping Generation
~~~~~~~~~~~~~~~~~~~~~~~~~

With the `entities` subcommand, :program:`llvm-ir2vec` generates the entity mappings
supported by IR2Vec in the standard format used for knowledge graph embedding
training. This subcommand outputs all supported entities (opcodes, types, and
operands) with their corresponding numeric IDs, and is not specific for an
LLVM IR file.

Usage:

.. code-block:: bash

   llvm-ir2vec entities -o entity2id.txt

Embedding Generation
~~~~~~~~~~~~~~~~~~~~

With the `embeddings` subcommand, :program:`llvm-ir2vec` uses a pre-trained vocabulary to
generate numerical embeddings for LLVM IR at different levels of granularity.

Example Usage:

.. code-block:: bash

   llvm-ir2vec embeddings --ir2vec-vocab-path=vocab.json --ir2vec-kind=symbolic --level=func input.bc -o embeddings.txt

OPTIONS
-------

Global options:

.. option:: -o <filename>

   Specify the output filename. Use ``-`` to write to standard output (default).

.. option:: --help

   Print a summary of command line options.

Subcommand-specific options:

**embeddings** subcommand:

.. option:: <input-file>

   The input LLVM IR or bitcode file to process. This positional argument is
   required for the `embeddings` subcommand.

.. option:: --level=<level>

   Specify the embedding generation level. Valid values are:

   * ``inst`` - Generate instruction-level embeddings
   * ``bb`` - Generate basic block-level embeddings  
   * ``func`` - Generate function-level embeddings (default)

.. option:: --function=<name>

   Process only the specified function instead of all functions in the module.

.. option:: --ir2vec-kind=<kind>

   Specify the kind of IR2Vec embeddings to generate. Valid values are:

   * ``symbolic`` - Generate symbolic embeddings (default)
   * ``flow-aware`` - Generate flow-aware embeddings

   Flow-aware embeddings consider control flow relationships between instructions,
   while symbolic embeddings focus on the symbolic representation of instructions.

.. option:: --ir2vec-vocab-path=<path>

   Specify the path to the vocabulary file (required for embedding generation).
   The vocabulary file should be in JSON format and contain the trained
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


**triplets** subcommand:

.. option:: <input-file>

   The input LLVM IR or bitcode file to process. This positional argument is
   required for the `triplets` subcommand.

**entities** subcommand:

   No subcommand-specific options.

OUTPUT FORMAT
-------------

Triplet Mode Output
~~~~~~~~~~~~~~~~~~~

In triplet mode, the output consists of numeric triplets in train2id format with
metadata headers. The format includes:

.. code-block:: text

   MAX_RELATIONS=<max_relations_count>
   <head_entity_id> <tail_entity_id> <relation_id>
   <head_entity_id> <tail_entity_id> <relation_id>
   ...

Each line after the metadata header represents one instruction relationship,
with numeric IDs for head entity, relation, and tail entity. The metadata 
header (MAX_RELATIONS) provides counts for post-processing and training setup.

Entity Mode Output
~~~~~~~~~~~~~~~~~~

In entity mode, the output consists of entity mapping in the format:

.. code-block:: text

   <total_entities>
   <entity_string>	<numeric_id>
   <entity_string>	<numeric_id>
   ...

The first line contains the total number of entities, followed by one entity
mapping per line with tab-separated entity string and numeric ID.

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
