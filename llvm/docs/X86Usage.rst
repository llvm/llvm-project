=============================
User Guide for X86 Backend
=============================

.. contents::
   :local:

Introduction
============

The X86 backend provides ISA code generation for X86 CPUs. It lives in the
``lib/Target/X86`` directory.

LLVM
====

.. _x86-processors

Processors
----------

Use the ``clang -march=<Processor>`` option to specify the X86 processor.

  .. table:: X86 processors
     :name: x86-processor-table

     ================== ===================
     Processor          Alternative
                        Name
     ``i386``
     ``i486``
     ``i586``
     ``pentium``
     ``pentium-mmx``
     ``i686``
     ``pentiumpro``
     ``pentium2``
     ``pentium3``       - ``pentium3m``
     ``pentium-m``
     ``pentium4``       - ``pentium4m``
     ``lakemont``
     ``yonah``
     ``prescott``
     ``nocona``
     ``core2``
     ``penryn``
     ``bonnell``        - ``atom``
     ``silvermont``     - ``slm``
     ``goldmont``
     ``nehalem``        - ``corei7``
     ``westmere``
     ``sandybridge``    - ``corei7-avx``
     ``ivybridge``      - ``core-avx-i``
     ``haswell``        - ``core-avx2``
     ``broadwell``      - ``skylake``
     ``knl``
     ``knm``
     ``skylake-avx512`` - ``skx``
     ``cannonlake``
     ``icelake``
     ``k6``
     ``k6-2``
     ``k6-3``
     ``athlon``         - ``athlon-tbird``
     ``athlon-4``       - ``athlon-xp``
                        - ``athlon-mp``
     ``k8``             - ``opteron``
                        - ``athlon64``
                        - ``athlon-fx``
     ``k8-sse3``        - ``opteron-sse3``
                        - ``athlon64-sse3``
     ``amdfam10h``      - ``barcelona``
     ``btver1``
     ``btver2``
     ``bdver1``
     ``bdver2``
     ``bdver3``
     ``bdver4``
     ``znver1``
     ``geode``
     ``winchip-c6``
     ``winchip2``
     ``c3``
     ``c3-2``
     ================== ===================
