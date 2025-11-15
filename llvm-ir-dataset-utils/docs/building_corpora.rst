Building Corpora
================

Building a Corpus from an Individual description
------------------------------------------------

To build a corpus from an individual description, run the following command from
the root directory of this repoistory:

.. code-block:: bash

  PYTHONPATH="./" python3 ./llvm_ir_dataset_utils/tools/corpus_from_description.py \
    --base_dir=<path to build> \
    --corpus_dir=<path to corpus> \
    --corpus_description=<path to corpus description json>


The script will take the application description, clone the source, build the
application with the appropriate flags, and then extract unoptimized IR from the
build, placing it in a subdirectory of the directory passed to `--corpus_dir` in
the ml-compiler-opt corpus format.
