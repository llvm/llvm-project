=============
Metal / AIR
=============

The Metal target compiles LLVM IR into Apple GPU Intermediate Representation
(AIR) and packages it inside a ``.metallib`` container that Apple's Metal
runtime can load and JIT.

.. warning::
   The Metal backend is experimental and under active development. It is not
   yet feature-complete or ready for use outside of experimental contexts.

.. toctree::
   :maxdepth: 1

   MetalTarget
   MetallibFormat
