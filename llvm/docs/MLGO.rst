====
MLGO
====

Introduction
============

MLGO is a framework for integrating ML techniques systematically in LLVM. It is
designed primarily to replace heuristics within LLVM with machine learned
models. Currently there is upstream infrastructure for the following
heuristics:

* Inlining for size
* Register allocation (LLVM greedy eviction heuristic) for performance

This document is an outline of the tooling that composes MLGO.

Corpus Tooling
==============

Within upstream LLVM, there is the ``mlgo-utils`` python packages that lives at
``llvm/utils/mlgo-utils``. This package primarily contains tooling for working
with corpora, or collections of LLVM bitcode. We use these corpora to 

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
``CMAKE_EXPORT_COMPILE_COMMANDS`` switch is turned on at compile time. Assuming
it was specified and there is a ``compilation_commands.json`` file within the
``./build`` directory, you can run the following command to create a corpus:

.. code-block:: bash

  python3 ./extract_ir.py \
    --input=./build/compile_commands.json \
    --input_type=json \
    --output_dir=./corpus

This assumes that the compilation was performed with bitcode embedding
enabled (done by passing ``-Xclang -fembed-bitcode=all`` to all C/C++
compilation actions). After running the above command, there should be a full
corpus of bitcode within the ``./corpus`` directory.

Example: Bazel Aquery
---------------------

This tool also supports extracting bitcode from bazel in multiple ways
depending upon the exact configuration. For ThinLTO, a linker parameters file
is preferred. For the non-ThinLTO case, the script will accept the output of
``bazel aquery`` which it will use to find all the object files that are linked
into a specific target and then extract bitcode from them. First, you need
to generate the aquery output

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

Model Runner Interfaces
=======================

..
    TODO(mtrofin): Write this section.
