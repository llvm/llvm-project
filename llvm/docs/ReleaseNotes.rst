========================
LLVM 7.0.0 Release Notes
========================

.. contents::
    :local:

.. warning::
   These are in-progress notes for the upcoming LLVM 7 release.
   Release notes for previous releases can be found on
   `the Download Page <http://releases.llvm.org/download.html>`_.


Introduction
============

This document contains the release notes for the LLVM Compiler Infrastructure,
release 7.0.0.  Here we describe the status of LLVM, including major improvements
from the previous release, improvements in various subprojects of LLVM, and
some of the current users of the code.  All LLVM releases may be downloaded
from the `LLVM releases web site <http://llvm.org/releases/>`_.

For more information about LLVM, including information about the latest
release, please check out the `main LLVM web site <http://llvm.org/>`_.  If you
have questions or comments, the `LLVM Developer's Mailing List
<http://lists.llvm.org/mailman/listinfo/llvm-dev>`_ is a good place to send
them.

Note that if you are reading this file from a Subversion checkout or the main
LLVM web page, this document applies to the *next* release, not the current
one.  To see the release notes for a specific release, please see the `releases
page <http://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================
.. NOTE
   For small 1-3 sentence descriptions, just add an entry at the end of
   this list. If your description won't fit comfortably in one bullet
   point (e.g. maybe you would like to give an example of the
   functionality, or simply have a lot to talk about), see the `NOTE` below
   for adding a new subsection.

* Libraries have been renamed from 7.0 to 7. This change also impacts
  downstream libraries like lldb.

* The LoopInstSimplify pass (-loop-instsimplify) has been removed.

* Symbols starting with ``?`` are no longer mangled by LLVM when using the
  Windows ``x`` or ``w`` IR mangling schemes.

* A new tool named :doc:`llvm-exegesis <CommandGuide/llvm-exegesis>` has been
  added. :program:`llvm-exegesis` automatically measures instruction scheduling
  properties (latency/uops) and provides a principled way to edit scheduling
  models.

* A new tool named :doc:`llvm-mca <CommandGuide/llvm-mca>` has been added.
  :program:`llvm-mca` is a  static performance analysis tool that uses
  information available in LLVM to statically predict the performance of
  machine code for a specific CPU.

* The optimization flag to merge constants (-fmerge-all-constants) is no longer
  applied by default.

* Optimization of floating-point casts is improved. This may cause surprising
  results for code that is relying on the undefined behavior of overflowing 
  casts. The optimization can be disabled by specifying a function attribute:
  "strict-float-cast-overflow"="false". This attribute may be created by the
  clang option :option:`-fno-strict-float-cast-overflow`.
  Code sanitizers can be used to detect affected patterns. The option for
  detecting this problem alone is "-fsanitize=float-cast-overflow":

.. code-block:: c

    int main() {
      float x = 4294967296.0f;
      x = (float)((int)x);
      printf("junk in the ftrunc: %f\n", x);
      return 0;
    }

.. code-block:: bash

    clang -O1 ftrunc.c -fsanitize=float-cast-overflow ; ./a.out 
    ftrunc.c:5:15: runtime error: 4.29497e+09 is outside the range of representable values of type 'int'
    junk in the ftrunc: 0.000000

* ``LLVM_ON_WIN32`` is no longer set by ``llvm/Config/config.h`` and
  ``llvm/Config/llvm-config.h``.  If you used this macro, use the compiler-set
  ``_WIN32`` instead which is set exactly when ``LLVM_ON_WIN32`` used to be set.

* Note..

.. NOTE
   If you would like to document a larger change, then you can add a
   subsection about it right here. You can copy the following boilerplate
   and un-indent it (the indentation causes it to be inside this comment).

   Special New Feature
   -------------------

   Makes programs 10x faster by doing Special New Thing.

Changes to the LLVM IR
----------------------

* The signatures for the builtins @llvm.memcpy, @llvm.memmove, and @llvm.memset
  have changed. Alignment is no longer an argument, and are instead conveyed as
  parameter attributes.

* invariant.group.barrier has been renamed to launder.invariant.group.

* invariant.group metadata can now refer only empty metadata nodes.

Changes to the ARM Backend
--------------------------

 During this release ...


Changes to the MIPS Target
--------------------------

 During this release ...


Changes to the PowerPC Target
-----------------------------

 During this release ...

Changes to the X86 Target
-------------------------

 During this release ...

Changes to the AMDGPU Target
-----------------------------

 During this release ...

Changes to the AVR Target
-----------------------------

 During this release ...

Changes to the OCaml bindings
-----------------------------

* Remove ``add_bb_vectorize``.


Changes to the C API
--------------------

* Remove ``LLVMAddBBVectorizePass``. The implementation was removed and the C
  interface was made a deprecated no-op in LLVM 5. Use
  ``LLVMAddSLPVectorizePass`` instead to get the supported SLP vectorizer.

Changes to the DAG infrastructure
---------------------------------
* ADDC/ADDE/SUBC/SUBE are now deprecated and will default to expand. Backends
  that wish to continue to use these opcodes should explicitely request so
  using ``setOperationAction`` in their ``TargetLowering``. New backends
  should use UADDO/ADDCARRY/USUBO/SUBCARRY instead of the deprecated opcodes.

External Open Source Projects Using LLVM 7
==========================================

* A project...


Additional Information
======================

A wide variety of additional information is available on the `LLVM web page
<http://llvm.org/>`_, in particular in the `documentation
<http://llvm.org/docs/>`_ section.  The web page also contains versions of the
API documentation which is up-to-date with the Subversion version of the source
code.  You can access versions of these documents specific to this release by
going into the ``llvm/docs/`` directory in the LLVM tree.

If you have any questions or comments about LLVM, please feel free to contact
us via the `mailing lists <http://llvm.org/docs/#maillist>`_.
