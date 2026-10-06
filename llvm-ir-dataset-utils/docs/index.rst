LLVM-IR Dataset Utilities Documentation
=======================================

**LLVM-IR Dataset Utilities** is a set of utilities for the construction of large LLVM IR-based
datasets from multiple sources for the development of LLVM-focussed machine learning approaches.
It is specifically designed to build corpora of bitcode out of language package indices. Built
versions of the dataset are available from the `LLVM-ML HuggingFace Organization <https://huggingface.co/llvm-ml>`_.


Features
^^^^^^^^^

.. grid::

   .. grid-item-card::
      :columns: 12 12 12 6

      .. card:: Scalability
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Readily scalable build infrastructure, rapidly scaling with `Ray <https://ray.io/>`_
            to support the rapid compilation of 1000s of code bases across entire CPU clusters.

   .. grid-item-card::
      :columns: 12 12 12 6

      .. card:: Vast Builder Support
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Extensive support for building from a variety of sources including C, C++, Rust,
            Swift, Julia, and more.

   .. grid-item-card::
      :columns: 12 12 12 6

      .. card:: Statistical Introspection
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Enabling cross-language statistical analysis of across LLVM infrastructure-based programming languages,
            on their primitive usage patterns, pass mutations, and beyond.

   .. grid-item-card::
      :columns: 12 12 12 6

      .. card:: IR-Mutation Interception
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Able to intercept the compilation process at every instance the IR gets mutated for in-depth
            analysis of the compilation process, and construction of IR compilation stages-based datasets.

.. toctree::
   :hidden:
   :maxdepth: 2

   installation
   quickstart
   building_corpora
   API Reference <_autosummary/llvm_ir_dataset_utils>
