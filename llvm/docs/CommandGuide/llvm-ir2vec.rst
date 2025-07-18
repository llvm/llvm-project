llvm-ir2vec - IR2Vec Embedding Generation Tool
==============================================

.. program:: llvm-ir2vec

SYNOPSIS
--------

:program:`llvm-ir2vec` [*options*] *input-file*

DESCRIPTION
-----------

:program:`llvm-ir2vec` is a standalone command-line tool for IR2Vec. It
generates IR2Vec embeddings for LLVM IR and supports triplet generation 
for vocabulary training. It provides two main operation modes:

1. **Triplet Mode**: Generates triplets (opcode, type, operands) for vocabulary
   training from LLVM IR.

2. **Embedding Mode**: Generates IR2Vec embeddings using a trained vocabulary
   at different granularity levels (instruction, basic block, or function).

The tool is designed to facilitate machine learning applications that work with
LLVM IR by converting the IR into numerical representations that can be used by
ML models.

.. note::

   For information about using IR2Vec programmatically within LLVM passes and 
   the C++ API, see the `IR2Vec Embeddings <https://llvm.org/docs/MLGO.html#ir2vec-embeddings>`_ 
   section in the MLGO documentation.

OPERATION MODES
---------------

Triplet Generation Mode
~~~~~~~~~~~~~~~~~~~~~~~

In triplet mode, :program:`llvm-ir2vec` analyzes LLVM IR and extracts triplets
consisting of opcodes, types, and operands. These triplets can be used to train
vocabularies for embedding generation.

Usage:

.. code-block:: bash

   llvm-ir2vec --mode=triplets input.bc -o triplets.txt

Embedding Generation Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~

In embedding mode, :program:`llvm-ir2vec` uses a pre-trained vocabulary to
generate numerical embeddings for LLVM IR at different levels of granularity.

Example Usage:

.. code-block:: bash

   llvm-ir2vec --mode=embeddings --ir2vec-vocab-path=vocab.json --level=func input.bc -o embeddings.txt

OPTIONS
-------

.. option:: --mode=<mode>

 Specify the operation mode. Valid values are:

 * ``triplets`` - Generate triplets for vocabulary training
 * ``embeddings`` - Generate embeddings using trained vocabulary (default)

.. option:: --level=<level>

 Specify the embedding generation level. Valid values are:

 * ``inst`` - Generate instruction-level embeddings
 * ``bb`` - Generate basic block-level embeddings  
 * ``func`` - Generate function-level embeddings (default)

.. option:: --function=<name>

 Process only the specified function instead of all functions in the module.

.. option:: --ir2vec-vocab-path=<path>

 Specify the path to the vocabulary file (required for embedding mode).
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

.. option:: -o <filename>

 Specify the output filename. Use ``-`` to write to standard output (default).

.. option:: --help

 Print a summary of command line options.

.. note::

   ``--level``, ``--function``, ``--ir2vec-vocab-path``, ``--ir2vec-opc-weight``, 
   ``--ir2vec-type-weight``, and ``--ir2vec-arg-weight`` are only used in embedding 
   mode. These options are ignored in triplet mode.

INPUT FILE FORMAT
-----------------

:program:`llvm-ir2vec` accepts LLVM bitcode files (``.bc``) and LLVM IR files 
(``.ll``) as input. The input file should contain valid LLVM IR.

OUTPUT FORMAT
-------------

Triplet Mode Output
~~~~~~~~~~~~~~~~~~~

In triplet mode, the output consists of lines containing space-separated triplets:

.. code-block:: text

   <opcode> <type> <operand1> <operand2> ...

Each line represents the information of one instruction, with the opcode, type,
and operands.

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
