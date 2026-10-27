==========
AIR Target
==========

The AIR target compiles LLVM IR into Apple GPU Intermediate Representation
(AIR) and packages it inside a ``.metallib`` container that Apple's Metal
runtime can load and JIT.

The target was written by imperatormk
(`imperatormk/llvm-project <https://github.com/imperatormk/llvm-project/tree/metal-target-poc>`_,
branch ``metal-target-poc``) and brought into this tree with its commit
history intact.

.. warning::
   The AIR backend is experimental and under active development. It is not
   yet feature-complete or ready for use outside of experimental contexts.

.. toctree::
   :maxdepth: 1

   AIRTarget
   AIRLibFormat
   AIRStatus
