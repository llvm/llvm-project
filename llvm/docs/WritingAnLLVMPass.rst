====================
Writing an LLVM Pass
====================

.. program:: opt

.. contents::
    :local:

Introduction --- What is a pass?
================================

The LLVM Pass Framework is an important part of the LLVM system, because LLVM
passes are where most of the interesting parts of the compiler exist.  Passes
perform the transformations and optimizations that make up the compiler, they
build the analysis results that are used by these transformations, and they
are, above all, a structuring technique for compiler code.

All LLVM passes are subclasses of the `Pass
<http://llvm.org/doxygen/classllvm_1_1Pass.html>`_ class, which implement
functionality by overriding virtual methods inherited from ``Pass``.  Depending
on how your pass works, you should inherit from the :ref:`ModulePass
<writing-an-llvm-pass-ModulePass>` , :ref:`CallGraphSCCPass
<writing-an-llvm-pass-CallGraphSCCPass>`, :ref:`FunctionPass
<writing-an-llvm-pass-FunctionPass>` , or :ref:`LoopPass
<writing-an-llvm-pass-LoopPass>`, or :ref:`RegionPass
<writing-an-llvm-pass-RegionPass>`, or :ref:`BasicBlockPass
<writing-an-llvm-pass-BasicBlockPass>` classes, which gives the system more
information about what your pass does, and how it can be combined with other
passes.  One of the main features of the LLVM Pass Framework is that it
schedules passes to run in an efficient way based on the constraints that your
pass meets (which are indicated by which class they derive from).

We start by showing you how to construct a pass, everything from setting up the
code, to compiling, loading, and executing it.  After the basics are down, more
advanced features are discussed.

Quick Start --- Writing hello world
===================================

Here we describe how to write the "hello world" of passes.  The "Hello" pass is
designed to simply print out the name of non-external functions that exist in
the program being compiled.  It does not modify the program at all, it just
inspects it.  The source code and files for this pass are available in the LLVM
source tree in the ``lib/Transforms/Hello`` directory.

.. _writing-an-llvm-pass-makefile:

Setting up the build environment
--------------------------------

First, configure and build LLVM.  Next, you need to create a new directory
somewhere in the LLVM source base.  For this example, we'll assume that you
made ``lib/Transforms/Hello``.  Finally, you must set up a build script
that will compile the source code for the new pass.  To do this,
copy the following into ``CMakeLists.txt``:

.. code-block:: cmake

  add_llvm_library( LLVMHello MODULE
    Hello.cpp
  
    PLUGIN_TOOL
    opt
    )

and the following line into ``lib/Transforms/CMakeLists.txt``:

.. code-block:: cmake

  add_subdirectory(Hello)

(Note that there is already a directory named ``Hello`` with a sample "Hello"
pass; you may play with it -- in which case you don't need to modify any
``CMakeLists.txt`` files -- or, if you want to create everything from scratch,
use another name.)

This build script specifies that ``Hello.cpp`` file in the current directory
is to be compiled and linked into a shared object ``$(LEVEL)/lib/LLVMHello.so`` that
can be dynamically loaded by the :program:`opt` tool via its :option:`-load`
option. If your operating system uses a suffix other than ``.so`` (such as
Windows or macOS), the appropriate extension will be used.

Now that we have the build scripts set up, we just need to write the code for
the pass itself.

.. _writing-an-llvm-pass-basiccode:

Basic code required
-------------------

Now that we have a way to compile our new pass, we just have to write it.
Start out with:

.. code-block:: c++

  #include "llvm/Pass.h"
  #include "llvm/IR/Function.h"
  #include "llvm/Support/raw_ostream.h"

Which are needed because we are writing a `Pass
<http://llvm.org/doxygen/classllvm_1_1Pass.html>`_, we are operating on
`Function <http://llvm.org/doxygen/classllvm_1_1Function.html>`_\ s, and we will
be doing some printing.

Next we have:

.. code-block:: c++

  using namespace llvm;

... which is required because the functions from the include files live in the
llvm namespace.

Next we have:

.. code-block:: c++

  namespace {

... which starts out an anonymous namespace.  Anonymous namespaces are to C++
what the "``static``" keyword is to C (at global scope).  It makes the things
declared inside of the anonymous namespace visible only to the current file.
If you're not familiar with them, consult a decent C++ book for more
information.

Next, we declare our pass itself:

.. code-block:: c++

  struct Hello : public FunctionPass {

This declares a "``Hello``" class that is a subclass of :ref:`FunctionPass
<writing-an-llvm-pass-FunctionPass>`.  The different builtin pass subclasses
are described in detail :ref:`later <writing-an-llvm-pass-pass-classes>`, but
for now, know that ``FunctionPass`` operates on a function at a time.

.. code-block:: c++

    static char ID;
    Hello() : FunctionPass(ID) {}

This declares pass identifier used by LLVM to identify pass.  This allows LLVM
to avoid using expensive C++ runtime information.

.. code-block:: c++

    bool runOnFunction(Function &F) override {
      errs() << "Hello: ";
      errs().write_escaped(F.getName()) << '\n';
      return false;
    }
  }; // end of struct Hello
  }  // end of anonymous namespace

We declare a :ref:`runOnFunction <writing-an-llvm-pass-runOnFunction>` method,
which overrides an abstract virtual method inherited from :ref:`FunctionPass
<writing-an-llvm-pass-FunctionPass>`.  This is where we are supposed to do our
thing, so we just print out our message with the name of each function.

.. code-block:: c++

  char Hello::ID = 0;

We initialize pass ID here.  LLVM uses ID's address to identify a pass, so
initialization value is not important.

.. code-block:: c++

  static RegisterPass<Hello> X("hello", "Hello World Pass",
                               false /* Only looks at CFG */,
                               false /* Analysis Pass */);

Lastly, we :ref:`register our class <writing-an-llvm-pass-registration>`
``Hello``, giving it a command line argument "``hello``", and a name "Hello
World Pass".  The last two arguments describe its behavior: if a pass walks CFG
without modifying it then the third argument is set to ``true``; if a pass is
an analysis pass, for example dominator tree pass, then ``true`` is supplied as
the fourth argument.

If we want to register the pass as a step of an existing pipeline, some extension
points are provided, e.g. ``PassManagerBuilder::EP_EarlyAsPossible`` to apply our
pass before any optimization, or ``PassManagerBuilder::EP_FullLinkTimeOptimizationLast``
to apply it after Link Time Optimizations.

.. code-block:: c++

    static llvm::RegisterStandardPasses Y(
        llvm::PassManagerBuilder::EP_EarlyAsPossible,
        [](const llvm::PassManagerBuilder &Builder,
           llvm::legacy::PassManagerBase &PM) { PM.add(new Hello()); });

As a whole, the ``.cpp`` file looks like:

.. code-block:: c++

  #include "llvm/Pass.h"
  #include "llvm/IR/Function.h"
  #include "llvm/Support/raw_ostream.h"

  #include "llvm/IR/LegacyPassManager.h"
  #include "llvm/Transforms/IPO/PassManagerBuilder.h"

  using namespace llvm;

  namespace {
  struct Hello : public FunctionPass {
    static char ID;
    Hello() : FunctionPass(ID) {}
  
    bool runOnFunction(Function &F) override {
      errs() << "Hello: ";
      errs().write_escaped(F.getName()) << '\n';
      return false;
    }
  }; // end of struct Hello
  }  // end of anonymous namespace

  char Hello::ID = 0;
  static RegisterPass<Hello> X("hello", "Hello World Pass",
                               false /* Only looks at CFG */,
                               false /* Analysis Pass */);

  static RegisterStandardPasses Y(
      PassManagerBuilder::EP_EarlyAsPossible,
      [](const PassManagerBuilder &Builder,
         legacy::PassManagerBase &PM) { PM.add(new Hello()); });

Now that it's all together, compile the file with a simple "``gmake``" command
from the top level of your build directory and you should get a new file
"``lib/LLVMHello.so``".  Note that everything in this file is
contained in an anonymous namespace --- this reflects the fact that passes
are self contained units that do not need external interfaces (although they
can have them) to be useful.

Running a pass with ``opt``
---------------------------

Now that you have a brand new shiny shared object file, we can use the
:program:`opt` command to run an LLVM program through your pass.  Because you
registered your pass with ``RegisterPass``, you will be able to use the
:program:`opt` tool to access it, once loaded.

To test it, follow the example at the end of the :doc:`GettingStarted` to
compile "Hello World" to LLVM.  We can now run the bitcode file (hello.bc) for
the program through our transformation like this (or course, any bitcode file
will work):

.. code-block:: console

  $ opt -load lib/LLVMHello.so -hello < hello.bc > /dev/null
  Hello: __main
  Hello: puts
  Hello: main

The :option:`-load` option specifies that :program:`opt` should load your pass
as a shared object, which makes "``-hello``" a valid command line argument
(which is one reason you need to :ref:`register your pass
<writing-an-llvm-pass-registration>`).  Because the Hello pass does not modify
the program in any interesting way, we just throw away the result of
:program:`opt` (sending it to ``/dev/null``).

To see what happened to the other string you registered, try running
:program:`opt` with the :option:`-help` option:

.. code-block:: console

  $ opt -load lib/LLVMHello.so -help
  OVERVIEW: llvm .bc -> .bc modular optimizer and analysis printer

  USAGE: opt [subcommand] [options] <input bitcode file>

  OPTIONS:
    Optimizations available:
  ...
      -guard-widening           - Widen guards
      -gvn                      - Global Value Numbering
      -gvn-hoist                - Early GVN Hoisting of Expressions
      -hello                    - Hello World Pass
      -indvars                  - Induction Variable Simplification
      -inferattrs               - Infer set function attributes
  ...

The pass name gets added as the information string for your pass, giving some
documentation to users of :program:`opt`.  Now that you have a working pass,
you would go ahead and make it do the cool transformations you want.  Once you
get it all working and tested, it may become useful to find out how fast your
pass is.  The :ref:`PassManager <writing-an-llvm-pass-passmanager>` provides a
nice command line option (:option:`-time-passes`) that allows you to get
information about the execution time of your pass along with the other passes
you queue up.  For example:

.. code-block:: console

  $ opt -load lib/LLVMHello.so -hello -time-passes < hello.bc > /dev/null
  Hello: __main
  Hello: puts
  Hello: main
  ===-------------------------------------------------------------------------===
                        ... Pass execution timing report ...
  ===-------------------------------------------------------------------------===
    Total Execution Time: 0.0007 seconds (0.0005 wall clock)
  
     ---User Time---   --User+System--   ---Wall Time---  --- Name ---
     0.0004 ( 55.3%)   0.0004 ( 55.3%)   0.0004 ( 75.7%)  Bitcode Writer
     0.0003 ( 44.7%)   0.0003 ( 44.7%)   0.0001 ( 13.6%)  Hello World Pass
     0.0000 (  0.0%)   0.0000 (  0.0%)   0.0001 ( 10.7%)  Module Verifier
     0.0007 (100.0%)   0.0007 (100.0%)   0.0005 (100.0%)  Total

As you can see, our implementation above is pretty fast.  The additional
passes listed are automatically inserted by the :program:`opt` tool to verify
that the LLVM emitted by your pass is still valid and well formed LLVM, which
hasn't been broken somehow.

Now that you have seen the basics of the mechanics behind passes, we can talk
about some more details of how they work and how to use them.

.. _writing-an-llvm-pass-pass-classes:

Pass classes and requirements
=============================

One of the first things that you should do when designing a new pass is to
decide what class you should subclass for your pass.  The :ref:`Hello World
<writing-an-llvm-pass-basiccode>` example uses the :ref:`FunctionPass
<writing-an-llvm-pass-FunctionPass>` class for its implementation, but we did
not discuss why or when this should occur.  Here we talk about the classes
available, from the most general to the most specific.

When choosing a superclass for your ``Pass``, you should choose the **most
specific** class possible, while still being able to meet the requirements
listed.  This gives the LLVM Pass Infrastructure information necessary to
optimize how passes are run, so that the resultant compiler isn't unnecessarily
slow.

The ``ImmutablePass`` class
---------------------------

The most plain and boring type of pass is the "`ImmutablePass
<http://llvm.org/doxygen/classllvm_1_1ImmutablePass.html>`_" class.  This pass
type is used for passes that do not have to be run, do not change state, and
never need to be updated.  This is not a normal type of transformation or
analysis, but can provide information about the current compiler configuration.

Although this pass class is very infrequently used, it is important for
providing information about the current target machine being compiled for, and
other static information that can affect the various transformations.

``ImmutablePass``\ es never invalidate other transformations, are never
invalidated, and are never "run".

.. _writing-an-llvm-pass-ModulePass:

The ``ModulePass`` class
------------------------

The `ModulePass <http://llvm.org/doxygen/classllvm_1_1ModulePass.html>`_ class
is the most general of all superclasses that you can use.  Deriving from
``ModulePass`` indicates that your pass uses the entire program as a unit,
referring to function bodies in no predictable order, or adding and removing
functions.  Because nothing is known about the behavior of ``ModulePass``
subclasses, no optimization can be done for their execution.

A module pass can use function level passes (e.g. dominators) using the
``getAnalysis`` interface ``getAnalysis<DominatorTree>(llvm::Function *)`` to
provide the function to retrieve analysis result for, if the function pass does
not require any module or immutable passes.  Note that this can only be done
for functions for which the analysis ran, e.g. in the case of dominators you
should only ask for the ``DominatorTree`` for function definitions, not
declarations.

To write a correct ``ModulePass`` subclass, derive from ``ModulePass`` and
overload the ``runOnModule`` method with the following signature:

The ``runOnModule`` method
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

  virtual bool runOnModule(Module &M) = 0;

The ``runOnModule`` method performs the interesting work of the pass.  It
should return ``true`` if the module was modified by the transformation and
``false`` otherwise.

.. _writing-an-llvm-pass-CallGraphSCCPass:

The ``CallGraphSCCPass`` class
------------------------------

The `CallGraphSCCPass
<http://llvm.org/doxygen/classllvm_1_1CallGraphSCCPass.html>`_ is used by
passes that need to traverse the program bottom-up on the call graph (callees
before callers).  Deriving from ``CallGraphSCCPass`` provides some mechanics
for building and traversing the ``CallGraph``, but also allows the system to
optimize execution of ``CallGraphSCCPass``\ es.  If your pass meets the
requirements outlined below, and doesn't meet the requirements of a
:ref:`FunctionPass <writing-an-llvm-pass-FunctionPass>` or :ref:`BasicBlockPass
<writing-an-llvm-pass-BasicBlockPass>`, you should derive from
``CallGraphSCCPass``.

``TODO``: explain briefly what SCC, Tarjan's algo, and B-U mean.

To be explicit, CallGraphSCCPass subclasses are:

#. ... *not allowed* to inspect or modify any ``Function``\ s other than those
   in the current SCC and the direct callers and direct callees of the SCC.
#. ... *required* to preserve the current ``CallGraph`` object, updating it to
   reflect any changes made to the program.
#. ... *not allowed* to add or remove SCC's from the current Module, though
   they may change the contents of an SCC.
#. ... *allowed* to add or remove global variables from the current Module.
#. ... *allowed* to maintain state across invocations of :ref:`runOnSCC
   <writing-an-llvm-pass-runOnSCC>` (including global data).

Implementing a ``CallGraphSCCPass`` is slightly tricky in some cases because it
has to handle SCCs with more than one node in it.  All of the virtual methods
described below should return ``true`` if they modified the program, or
``false`` if they didn't.

The ``doInitialization(CallGraph &)`` method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

  virtual bool doInitialization(CallGraph &CG);

The ``doInitialization`` method is allowed to do most of the things that
``CallGraphSCCPass``\ es are not allowed to do.  They can add and remove
functions, get pointers to functions, etc.  The ``doInitialization`` method is
designed to do simple initialization type of stuff that does not depend on the
SCCs being processed.  The ``doInitialization`` method call is not scheduled to
overlap with any other pass executions (thus it should be very fast).

.. _writing-an-llvm-pass-runOnSCC:

The ``runOnSCC`` method
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

  virtual bool runOnSCC(CallGraphSCC &SCC) = 0;

The ``runOnSCC`` method performs the interesting work of the pass, and should
return ``true`` if the module was modified by the transformation, ``false``
otherwise.

The ``doFinalization(CallGraph &)`` method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

  virtual bool doFinalization(CallGraph &CG);

The ``doFinalization`` method is an infrequently used method that is called
when the pass framework has finished calling :ref:`runOnSCC
<writing-an-llvm-pass-runOnSCC>` for every SCC in the program being compiled.

.. _writing-an-llvm-pass-FunctionPass:

The ``FunctionPass`` class
--------------------------

In contrast to ``ModulePass`` subclasses, `FunctionPass
<http://llvm.org/doxygen/classllvm_1_1Pass.html>`_ subclasses do have a
predictable, local behavior that can be expected by the system.  All
``FunctionPass`` execute on each function in the program independent of all of
the other functions in the program.  ``FunctionPass``\ es do not require that
they are executed in a particular order, and ``FunctionPass``\ es do not modify
external functions.

To be explicit, ``FunctionPass`` subclasses are not allowed to:

#. Inspect or modify a ``Function`` other than the one currently being processed.
#. Add or remove ``Function``\ s from the current ``Module``.
#. Add or remove global variables from the current ``Module``.
#. Maintain state across invocations of :ref:`runOnFunction
   <writing-an-llvm-pass-runOnFunction>` (including global data).

Implementing a ``FunctionPass`` is usually straightforward (See the :ref:`Hello
World <writing-an-llvm-pass-basiccode>` pass for example).
``FunctionPass``\ es may overload three virtual methods to do their work.  All
of these methods should return ``true`` if they modified the program, or
``false`` if they didn't.

.. _writing-an-llvm-pass-doInitialization-mod:

The ``doInitialization(Module &)`` method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

  virtual bool doInitialization(Module &M);

The ``doInitialization`` method is allowed to do most of the things that
``FunctionPass``\ es are not allowed to do.  They can add and remove functions,
get pointers to functions, etc.  The ``doInitialization`` method is designed to
do simple initialization type of stuff that does not depend on the functions
being processed.  The ``doInitialization`` method call is not scheduled to
overlap with any other pass executions (thus it should be very fast).

A good example of how this method should be used is the `LowerAllocations
<http://llvm.org/doxygen/LowerAllocations_8cpp-source.html>`_ pass.  This pass
converts ``malloc`` and ``free`` instructions into platform dependent
``malloc()`` and ``free()`` function calls.  It uses the ``doInitialization``
method to get a reference to the ``malloc`` and ``free`` functions that it
needs, adding prototypes to the module if necessary.

.. _writing-an-llvm-pass-runOnFunction:

The ``runOnFunction`` method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

  virtual bool runOnFunction(Function &F) = 0;

The ``runOnFunction`` method must be implemented by your subclass to do the
transformation or analysis work of your pass.  As usual, a ``true`` value
should be returned if the function is modified.

.. _writing-an-llvm-pass-doFinalization-mod:

The ``doFinalization(Module &)`` method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

  virtual bool doFinalization(Module &M);

The ``doFinalization`` method is an infrequently used method that is called
when the pass framework has finished calling :ref:`runOnFunction
<writing-an-llvm-pass-runOnFunction>` for every function in the program being
compiled.

.. _writing-an-llvm-pass-LoopPass:

The ``LoopPass`` class
----------------------

All ``LoopPass`` execute on each loop in the function independent of all of the
other loops in the function.  ``LoopPass`` processes loops in loop nest order
such that outer most loop is processed last.

``LoopPass`` subclasses are allowed to update loop nest using ``LPPassManager``
interface.  Implementing a loop pass is usually straightforward.
``LoopPass``\ es may overload three virtual methods to do their work.  All
these methods should return ``true`` if they modified the program, or ``false``
if they didn't.

A ``LoopPass`` subclass which is intended to run as part of the main loop pass
pipeline needs to preserve all of the same *function* analyses that the other
loop passes in its pipeline require. To make that easier,
a ``getLoopAnalysisUsage`` function is provided by ``LoopUtils.h``. It can be
called within the subclass's ``getAnalysisUsage`` override to get consistent
and correct behavior. Analogously, ``INITIALIZE_PASS_DEPENDENCY(LoopPass)``
will initialize this set of function analyses.

The ``doInitialization(Loop *, LPPassManager &)`` method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

  virtual bool doInitialization(Loop *, LPPassManager &LPM);

The ``doInitialization`` method is designed to do simple initialization type of
stuff that does not depend on the functions being processed.  The
``doInitialization`` method call is not scheduled to overlap with any other
pass executions (thus it should be very fast).  ``LPPassManager`` interface
should be used to access ``Function`` or ``Module`` level analysis information.

.. _writing-an-llvm-pass-runOnLoop:

The ``runOnLoop`` method
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

  virtual bool runOnLoop(Loop *, LPPassManager &LPM) = 0;

The ``runOnLoop`` method must be implemented by your subclass to do the
transformation or analysis work of your pass.  As usual, a ``true`` value
should be returned if the function is modified.  ``LPPassManager`` interface
should be used to update loop nest.

The ``doFinalization()`` method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

  virtual bool doFinalization();

The ``doFinalization`` method is an infrequently used method that is called
when the pass framework has finished calling :ref:`runOnLoop
<writing-an-llvm-pass-runOnLoop>` for every loop in the program being compiled.

.. _writing-an-llvm-pass-RegionPass:

The ``RegionPass`` class
------------------------

``RegionPass`` is similar to :ref:`LoopPass <writing-an-llvm-pass-LoopPass>`,
but executes on each single entry single exit region in the function.
``RegionPass`` processes regions in nested order such that the outer most
region is processed last.

``RegionPass`` subclasses are allowed to update the region tree by using the
``RGPassManager`` interface.  You may overload three virtual methods of
``RegionPass`` to implement your own region pass.  All these methods should
return ``true`` if they modified the program, or ``false`` if they did not.

The ``doInitialization(Region *, RGPassManager &)`` method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

  virtual bool doInitialization(Region *, RGPassManager &RGM);

The ``doInitialization`` method is designed to do simple initialization type of
stuff that does not depend on the functions being processed.  The
``doInitialization`` method call is not scheduled to overlap with any other
pass executions (thus it should be very fast).  ``RPPassManager`` interface
should be used to access ``Function`` or ``Module`` level analysis information.

.. _writing-an-llvm-pass-runOnRegion:

The ``runOnRegion`` method
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

  virtual bool runOnRegion(Region *, RGPassManager &RGM) = 0;

The ``runOnRegion`` method must be implemented by your subclass to do the
transformation or analysis work of your pass.  As usual, a true value should be
returned if the region is modified.  ``RGPassManager`` interface should be used to
update region tree.

The ``doFinalization()`` method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

  virtual bool doFinalization();

The ``doFinalization`` method is an infrequently used method that is called
when the pass framework has finished calling :ref:`runOnRegion
<writing-an-llvm-pass-runOnRegion>` for every region in the program being
compiled.

.. _writing-an-llvm-pass-BasicBlockPass:

The ``BasicBlockPass`` class
----------------------------

``BasicBlockPass``\ es are just like :ref:`FunctionPass's
<writing-an-llvm-pass-FunctionPass>` , except that they must limit their scope
of inspection and modification to a single basic block at a time.  As such,
they are **not** allowed to do any of the following:

#. Modify or inspect any basic blocks outside of the current one.
#. Maintain state across invocations of :ref:`runOnBasicBlock
   <writing-an-llvm-pass-runOnBasicBlock>`.
#. Modify the control flow graph (by altering terminator instructions)
#. Any of the things forbidden for :ref:`FunctionPasses
   <writing-an-llvm-pass-FunctionPass>`.

``BasicBlockPass``\ es are useful for traditional local and "peephole"
optimizations.  They may override the same :ref:`doInitialization(Module &)
<writing-an-llvm-pass-doInitialization-mod>` and :ref:`doFinalization(Module &)
<writing-an-llvm-pass-doFinalization-mod>` methods that :ref:`FunctionPass's
<writing-an-llvm-pass-FunctionPass>` have, but also have the following virtual
methods that may also be implemented:

The ``doInitialization(Function &)`` method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

  virtual bool doInitialization(Function &F);

The ``doInitialization`` method is allowed to do most of the things that
``BasicBlockPass``\ es are not allowed to do, but that ``FunctionPass``\ es
can.  The ``doInitialization`` method is designed to do simple initialization
that does not depend on the ``BasicBlock``\ s being processed.  The
``doInitialization`` method call is not scheduled to overlap with any other
pass executions (thus it should be very fast).

.. _writing-an-llvm-pass-runOnBasicBlock:

The ``runOnBasicBlock`` method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

  virtual bool runOnBasicBlock(BasicBlock &BB) = 0;

Override this function to do the work of the ``BasicBlockPass``.  This function
is not allowed to inspect or modify basic blocks other than the parameter, and
are not allowed to modify the CFG.  A ``true`` value must be returned if the
basic block is modified.

The ``doFinalization(Function &)`` method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

    virtual bool doFinalization(Function &F);

The ``doFinalization`` method is an infrequently used method that is called
when the pass framework has finished calling :ref:`runOnBasicBlock
<writing-an-llvm-pass-runOnBasicBlock>` for every ``BasicBlock`` in the program
being compiled.  This can be used to perform per-function finalization.

The ``MachineFunctionPass`` class
---------------------------------

A ``MachineFunctionPass`` is a part of the LLVM code generator that executes on
the machine-dependent representation of each LLVM function in the program.

Code generator passes are registered and initialized specially by
``TargetMachine::addPassesToEmitFile`` and similar routines, so they cannot
generally be run from the :program:`opt` or :program:`bugpoint` commands.

A ``MachineFunctionPass`` is also a ``FunctionPass``, so all the restrictions
that apply to a ``FunctionPass`` also apply to it.  ``MachineFunctionPass``\ es
also have additional restrictions.  In particular, ``MachineFunctionPass``\ es
are not allowed to do any of the following:

#. Modify or create any LLVM IR ``Instruction``\ s, ``BasicBlock``\ s,
   ``Argument``\ s, ``Function``\ s, ``GlobalVariable``\ s,
   ``GlobalAlias``\ es, or ``Module``\ s.
#. Modify a ``MachineFunction`` other than the one currently being processed.
#. Maintain state across invocations of :ref:`runOnMachineFunction
   <writing-an-llvm-pass-runOnMachineFunction>` (including global data).

.. _writing-an-llvm-pass-runOnMachineFunction:

The ``runOnMachineFunction(MachineFunction &MF)`` method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

  virtual bool runOnMachineFunction(MachineFunction &MF) = 0;

``runOnMachineFunction`` can be considered the main entry point of a
``MachineFunctionPass``; that is, you should override this method to do the
work of your ``MachineFunctionPass``.

The ``runOnMachineFunction`` method is called on every ``MachineFunction`` in a
``Module``, so that the ``MachineFunctionPass`` may perform optimizations on
the machine-dependent representation of the function.  If you want to get at
the LLVM ``Function`` for the ``MachineFunction`` you're working on, use
``MachineFunction``'s ``getFunction()`` accessor method --- but remember, you
may not modify the LLVM ``Function`` or its contents from a
``MachineFunctionPass``.

.. _writing-an-llvm-pass-registration:

Pass registration
-----------------

In the :ref:`Hello World <writing-an-llvm-pass-basiccode>` example pass we
illustrated how pass registration works, and discussed some of the reasons that
it is used and what it does.  Here we discuss how and why passes are
registered.

As we saw above, passes are registered with the ``RegisterPass`` template.  The
template parameter is the name of the pass that is to be used on the command
line to specify that the pass should be added to a program (for example, with
:program:`opt` or :program:`bugpoint`).  The first argument is the name of the
pass, which is to be used for the :option:`-help` output of programs, as well
as for debug output generated by the `--debug-pass` option.

If you want your pass to be easily dumpable, you should implement the virtual
print method:

The ``print`` method
^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

  virtual void print(llvm::raw_ostream &O, const Module *M) const;

The ``print`` method must be implemented by "analyses" in order to print a
human readable version of the analysis results.  This is useful for debugging
an analysis itself, as well as for other people to figure out how an analysis
works.  Use the opt ``-analyze`` argument to invoke this method.

The ``llvm::raw_ostream`` parameter specifies the stream to write the results
on, and the ``Module`` parameter gives a pointer to the top level module of the
program that has been analyzed.  Note however that this pointer may be ``NULL``
in certain circumstances (such as calling the ``Pass::dump()`` from a
debugger), so it should only be used to enhance debug output, it should not be
depended on.

.. _writing-an-llvm-pass-interaction:

Specifying interactions between passes
--------------------------------------

One of the main responsibilities of the ``PassManager`` is to make sure that
passes interact with each other correctly.  Because ``PassManager`` tries to
:ref:`optimize the execution of passes <writing-an-llvm-pass-passmanager>` it
must know how the passes interact with each other and what dependencies exist
between the various passes.  To track this, each pass can declare the set of
passes that are required to be executed before the current pass, and the passes
which are invalidated by the current pass.

Typically this functionality is used to require that analysis results are
computed before your pass is run.  Running arbitrary transformation passes can
invalidate the computed analysis results, which is what the invalidation set
specifies.  If a pass does not implement the :ref:`getAnalysisUsage
<writing-an-llvm-pass-getAnalysisUsage>` method, it defaults to not having any
prerequisite passes, and invalidating **all** other passes.

.. _writing-an-llvm-pass-getAnalysisUsage:

The ``getAnalysisUsage`` method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

  virtual void getAnalysisUsage(AnalysisUsage &Info) const;

By implementing the ``getAnalysisUsage`` method, the required and invalidated
sets may be specified for your transformation.  The implementation should fill
in the `AnalysisUsage
<http://llvm.org/doxygen/classllvm_1_1AnalysisUsage.html>`_ object with
information about which passes are required and not invalidated.  To do this, a
pass may call any of the following methods on the ``AnalysisUsage`` object:

The ``AnalysisUsage::addRequired<>`` and ``AnalysisUsage::addRequiredTransitive<>`` methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your pass requires a previous pass to be executed (an analysis for example),
it can use one of these methods to arrange for it to be run before your pass.
LLVM has many different types of analyses and passes that can be required,
spanning the range from ``DominatorSet`` to ``BreakCriticalEdges``.  Requiring
``BreakCriticalEdges``, for example, guarantees that there will be no critical
edges in the CFG when your pass has been run.

Some analyses chain to other analyses to do their job.  For example, an
`AliasAnalysis <AliasAnalysis>` implementation is required to :ref:`chain
<aliasanalysis-chaining>` to other alias analysis passes.  In cases where
analyses chain, the ``addRequiredTransitive`` method should be used instead of
the ``addRequired`` method.  This informs the ``PassManager`` that the
transitively required pass should be alive as long as the requiring pass is.

The ``AnalysisUsage::addPreserved<>`` method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One of the jobs of the ``PassManager`` is to optimize how and when analyses are
run.  In particular, it attempts to avoid recomputing data unless it needs to.
For this reason, passes are allowed to declare that they preserve (i.e., they
don't invalidate) an existing analysis if it's available.  For example, a
simple constant folding pass would not modify the CFG, so it can't possibly
affect the results of dominator analysis.  By default, all passes are assumed
to invalidate all others.

The ``AnalysisUsage`` class provides several methods which are useful in
certain circumstances that are related to ``addPreserved``.  In particular, the
``setPreservesAll`` method can be called to indicate that the pass does not
modify the LLVM program at all (which is true for analyses), and the
``setPreservesCFG`` method can be used by transformations that change
instructions in the program but do not modify the CFG or terminator
instructions (note that this property is implicitly set for
:ref:`BasicBlockPass <writing-an-llvm-pass-BasicBlockPass>`\ es).

``addPreserved`` is particularly useful for transformations like
``BreakCriticalEdges``.  This pass knows how to update a small set of loop and
dominator related analyses if they exist, so it can preserve them, despite the
fact that it hacks on the CFG.

Example implementations of ``getAnalysisUsage``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

  // This example modifies the program, but does not modify the CFG
  void LICM::getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesCFG();
    AU.addRequired<LoopInfoWrapperPass>();
  }

.. _writing-an-llvm-pass-getAnalysis:

The ``getAnalysis<>`` and ``getAnalysisIfAvailable<>`` methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Pass::getAnalysis<>`` method is automatically inherited by your class,
providing you with access to the passes that you declared that you required
with the :ref:`getAnalysisUsage <writing-an-llvm-pass-getAnalysisUsage>`
method.  It takes a single template argument that specifies which pass class
you want, and returns a reference to that pass.  For example:

.. code-block:: c++

  bool LICM::runOnFunction(Function &F) {
    LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    //...
  }

This method call returns a reference to the pass desired.  You may get a
runtime assertion failure if you attempt to get an analysis that you did not
declare as required in your :ref:`getAnalysisUsage
<writing-an-llvm-pass-getAnalysisUsage>` implementation.  This method can be
called by your ``run*`` method implementation, or by any other local method
invoked by your ``run*`` method.

A module level pass can use function level analysis info using this interface.
For example:

.. code-block:: c++

  bool ModuleLevelPass::runOnModule(Module &M) {
    //...
    DominatorTree &DT = getAnalysis<DominatorTree>(Func);
    //...
  }

In above example, ``runOnFunction`` for ``DominatorTree`` is called by pass
manager before returning a reference to the desired pass.

If your pass is capable of updating analyses if they exist (e.g.,
``BreakCriticalEdges``, as described above), you can use the
``getAnalysisIfAvailable`` method, which returns a pointer to the analysis if
it is active.  For example:

.. code-block:: c++

  if (DominatorSet *DS = getAnalysisIfAvailable<DominatorSet>()) {
    // A DominatorSet is active.  This code will update it.
  }

Implementing Analysis Groups
----------------------------

Now that we understand the basics of how passes are defined, how they are used,
and how they are required from other passes, it's time to get a little bit
fancier.  All of the pass relationships that we have seen so far are very
simple: one pass depends on one other specific pass to be run before it can
run.  For many applications, this is great, for others, more flexibility is
required.

In particular, some analyses are defined such that there is a single simple
interface to the analysis results, but multiple ways of calculating them.
Consider alias analysis for example.  The most trivial alias analysis returns
"may alias" for any alias query.  The most sophisticated analysis a
flow-sensitive, context-sensitive interprocedural analysis that can take a
significant amount of time to execute (and obviously, there is a lot of room
between these two extremes for other implementations).  To cleanly support
situations like this, the LLVM Pass Infrastructure supports the notion of
Analysis Groups.

Analysis Group Concepts
^^^^^^^^^^^^^^^^^^^^^^^

An Analysis Group is a single simple interface that may be implemented by
multiple different passes.  Analysis Groups can be given human readable names
just like passes, but unlike passes, they need not derive from the ``Pass``
class.  An analysis group may have one or more implementations, one of which is
the "default" implementation.

Analysis groups are used by client passes just like other passes are: the
``AnalysisUsage::addRequired()`` and ``Pass::getAnalysis()`` methods.  In order
to resolve this requirement, the :ref:`PassManager
<writing-an-llvm-pass-passmanager>` scans the available passes to see if any
implementations of the analysis group are available.  If none is available, the
default implementation is created for the pass to use.  All standard rules for
:ref:`interaction between passes <writing-an-llvm-pass-interaction>` still
apply.

Although :ref:`Pass Registration <writing-an-llvm-pass-registration>` is
optional for normal passes, all analysis group implementations must be
registered, and must use the :ref:`INITIALIZE_AG_PASS
<writing-an-llvm-pass-RegisterAnalysisGroup>` template to join the
implementation pool.  Also, a default implementation of the interface **must**
be registered with :ref:`RegisterAnalysisGroup
<writing-an-llvm-pass-RegisterAnalysisGroup>`.

As a concrete example of an Analysis Group in action, consider the
`AliasAnalysis <http://llvm.org/doxygen/classllvm_1_1AliasAnalysis.html>`_
analysis group.  The default implementation of the alias analysis interface
(the `basicaa <http://llvm.org/doxygen/structBasicAliasAnalysis.html>`_ pass)
just does a few simple checks that don't require significant analysis to
compute (such as: two different globals can never alias each other, etc).
Passes that use the `AliasAnalysis
<http://llvm.org/doxygen/classllvm_1_1AliasAnalysis.html>`_ interface (for
example the `gvn <http://llvm.org/doxygen/classllvm_1_1GVN.html>`_ pass), do not
care which implementation of alias analysis is actually provided, they just use
the designated interface.

From the user's perspective, commands work just like normal.  Issuing the
command ``opt -gvn ...`` will cause the ``basicaa`` class to be instantiated
and added to the pass sequence.  Issuing the command ``opt -somefancyaa -gvn
...`` will cause the ``gvn`` pass to use the ``somefancyaa`` alias analysis
(which doesn't actually exist, it's just a hypothetical example) instead.

.. _writing-an-llvm-pass-RegisterAnalysisGroup:

Using ``RegisterAnalysisGroup``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``RegisterAnalysisGroup`` template is used to register the analysis group
itself, while the ``INITIALIZE_AG_PASS`` is used to add pass implementations to
the analysis group.  First, an analysis group should be registered, with a
human readable name provided for it.  Unlike registration of passes, there is
no command line argument to be specified for the Analysis Group Interface
itself, because it is "abstract":

.. code-block:: c++

  static RegisterAnalysisGroup<AliasAnalysis> A("Alias Analysis");

Once the analysis is registered, passes can declare that they are valid
implementations of the interface by using the following code:

.. code-block:: c++

  namespace {
    // Declare that we implement the AliasAnalysis interface
    INITIALIZE_AG_PASS(FancyAA, AliasAnalysis , "somefancyaa",
        "A more complex alias analysis implementation",
        false,  // Is CFG Only?
        true,   // Is Analysis?
        false); // Is default Analysis Group implementation?
  }

This just shows a class ``FancyAA`` that uses the ``INITIALIZE_AG_PASS`` macro
both to register and to "join" the `AliasAnalysis
<http://llvm.org/doxygen/classllvm_1_1AliasAnalysis.html>`_ analysis group.
Every implementation of an analysis group should join using this macro.

.. code-block:: c++

  namespace {
    // Declare that we implement the AliasAnalysis interface
    INITIALIZE_AG_PASS(BasicAA, AliasAnalysis, "basicaa",
        "Basic Alias Analysis (default AA impl)",
        false, // Is CFG Only?
        true,  // Is Analysis?
        true); // Is default Analysis Group implementation?
  }

Here we show how the default implementation is specified (using the final
argument to the ``INITIALIZE_AG_PASS`` template).  There must be exactly one
default implementation available at all times for an Analysis Group to be used.
Only default implementation can derive from ``ImmutablePass``.  Here we declare
that the `BasicAliasAnalysis
<http://llvm.org/doxygen/structBasicAliasAnalysis.html>`_ pass is the default
implementation for the interface.

Pass Statistics
===============

The `Statistic <http://llvm.org/doxygen/Statistic_8h_source.html>`_ class is
designed to be an easy way to expose various success metrics from passes.
These statistics are printed at the end of a run, when the :option:`-stats`
command line option is enabled on the command line.  See the :ref:`Statistics
section <Statistic>` in the Programmer's Manual for details.

.. _writing-an-llvm-pass-passmanager:

What PassManager does
---------------------

The `PassManager <http://llvm.org/doxygen/PassManager_8h_source.html>`_ `class
<http://llvm.org/doxygen/classllvm_1_1PassManager.html>`_ takes a list of
passes, ensures their :ref:`prerequisites <writing-an-llvm-pass-interaction>`
are set up correctly, and then schedules passes to run efficiently.  All of the
LLVM tools that run passes use the PassManager for execution of these passes.

The PassManager does two main things to try to reduce the execution time of a
series of passes:

#. **Share analysis results.**  The ``PassManager`` attempts to avoid
   recomputing analysis results as much as possible.  This means keeping track
   of which analyses are available already, which analyses get invalidated, and
   which analyses are needed to be run for a pass.  An important part of work
   is that the ``PassManager`` tracks the exact lifetime of all analysis
   results, allowing it to :ref:`free memory
   <writing-an-llvm-pass-releaseMemory>` allocated to holding analysis results
   as soon as they are no longer needed.

#. **Pipeline the execution of passes on the program.**  The ``PassManager``
   attempts to get better cache and memory usage behavior out of a series of
   passes by pipelining the passes together.  This means that, given a series
   of consecutive :ref:`FunctionPass <writing-an-llvm-pass-FunctionPass>`, it
   will execute all of the :ref:`FunctionPass
   <writing-an-llvm-pass-FunctionPass>` on the first function, then all of the
   :ref:`FunctionPasses <writing-an-llvm-pass-FunctionPass>` on the second
   function, etc... until the entire program has been run through the passes.

   This improves the cache behavior of the compiler, because it is only
   touching the LLVM program representation for a single function at a time,
   instead of traversing the entire program.  It reduces the memory consumption
   of compiler, because, for example, only one `DominatorSet
   <http://llvm.org/doxygen/classllvm_1_1DominatorSet.html>`_ needs to be
   calculated at a time.  This also makes it possible to implement some
   :ref:`interesting enhancements <writing-an-llvm-pass-SMP>` in the future.

The effectiveness of the ``PassManager`` is influenced directly by how much
information it has about the behaviors of the passes it is scheduling.  For
example, the "preserved" set is intentionally conservative in the face of an
unimplemented :ref:`getAnalysisUsage <writing-an-llvm-pass-getAnalysisUsage>`
method.  Not implementing when it should be implemented will have the effect of
not allowing any analysis results to live across the execution of your pass.

The ``PassManager`` class exposes a ``--debug-pass`` command line options that
is useful for debugging pass execution, seeing how things work, and diagnosing
when you should be preserving more analyses than you currently are.  (To get
information about all of the variants of the ``--debug-pass`` option, just type
"``opt -help-hidden``").

By using the --debug-pass=Structure option, for example, we can see how our
:ref:`Hello World <writing-an-llvm-pass-basiccode>` pass interacts with other
passes.  Lets try it out with the gvn and licm passes:

.. code-block:: console

  $ opt -load lib/LLVMHello.so -gvn -licm --debug-pass=Structure < hello.bc > /dev/null
  ModulePass Manager
    FunctionPass Manager
      Dominator Tree Construction
      Basic Alias Analysis (stateless AA impl)
      Function Alias Analysis Results
      Memory Dependence Analysis
      Global Value Numbering
      Natural Loop Information
      Canonicalize natural loops
      Loop-Closed SSA Form Pass
      Basic Alias Analysis (stateless AA impl)
      Function Alias Analysis Results
      Scalar Evolution Analysis
      Loop Pass Manager
        Loop Invariant Code Motion
      Module Verifier
    Bitcode Writer

This output shows us when passes are constructed.
Here we see that GVN uses dominator tree information to do its job.  The LICM pass
uses natural loop information, which uses dominator tree as well.

After the LICM pass, the module verifier runs (which is automatically added by
the :program:`opt` tool), which uses the dominator tree to check that the
resultant LLVM code is well formed. Note that the dominator tree is computed
once, and shared by three passes.

Lets see how this changes when we run the :ref:`Hello World
<writing-an-llvm-pass-basiccode>` pass in between the two passes:

.. code-block:: console

  $ opt -load lib/LLVMHello.so -gvn -hello -licm --debug-pass=Structure < hello.bc > /dev/null
  ModulePass Manager
    FunctionPass Manager
      Dominator Tree Construction
      Basic Alias Analysis (stateless AA impl)
      Function Alias Analysis Results
      Memory Dependence Analysis
      Global Value Numbering
      Hello World Pass
      Dominator Tree Construction
      Natural Loop Information
      Canonicalize natural loops
      Loop-Closed SSA Form Pass
      Basic Alias Analysis (stateless AA impl)
      Function Alias Analysis Results
      Scalar Evolution Analysis
      Loop Pass Manager
        Loop Invariant Code Motion
      Module Verifier
    Bitcode Writer
  Hello: __main
  Hello: puts
  Hello: main

Here we see that the :ref:`Hello World <writing-an-llvm-pass-basiccode>` pass
has killed the Dominator Tree pass, even though it doesn't modify the code at
all!  To fix this, we need to add the following :ref:`getAnalysisUsage
<writing-an-llvm-pass-getAnalysisUsage>` method to our pass:

.. code-block:: c++

  // We don't modify the program, so we preserve all analyses
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

Now when we run our pass, we get this output:

.. code-block:: console

  $ opt -load lib/LLVMHello.so -gvn -hello -licm --debug-pass=Structure < hello.bc > /dev/null
  Pass Arguments:  -gvn -hello -licm
  ModulePass Manager
    FunctionPass Manager
      Dominator Tree Construction
      Basic Alias Analysis (stateless AA impl)
      Function Alias Analysis Results
      Memory Dependence Analysis
      Global Value Numbering
      Hello World Pass
      Natural Loop Information
      Canonicalize natural loops
      Loop-Closed SSA Form Pass
      Basic Alias Analysis (stateless AA impl)
      Function Alias Analysis Results
      Scalar Evolution Analysis
      Loop Pass Manager
        Loop Invariant Code Motion
      Module Verifier
    Bitcode Writer
  Hello: __main
  Hello: puts
  Hello: main

Which shows that we don't accidentally invalidate dominator information
anymore, and therefore do not have to compute it twice.

.. _writing-an-llvm-pass-releaseMemory:

The ``releaseMemory`` method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

  virtual void releaseMemory();

The ``PassManager`` automatically determines when to compute analysis results,
and how long to keep them around for.  Because the lifetime of the pass object
itself is effectively the entire duration of the compilation process, we need
some way to free analysis results when they are no longer useful.  The
``releaseMemory`` virtual method is the way to do this.

If you are writing an analysis or any other pass that retains a significant
amount of state (for use by another pass which "requires" your pass and uses
the :ref:`getAnalysis <writing-an-llvm-pass-getAnalysis>` method) you should
implement ``releaseMemory`` to, well, release the memory allocated to maintain
this internal state.  This method is called after the ``run*`` method for the
class, before the next call of ``run*`` in your pass.

Registering dynamically loaded passes
=====================================

*Size matters* when constructing production quality tools using LLVM, both for
the purposes of distribution, and for regulating the resident code size when
running on the target system.  Therefore, it becomes desirable to selectively
use some passes, while omitting others and maintain the flexibility to change
configurations later on.  You want to be able to do all this, and, provide
feedback to the user.  This is where pass registration comes into play.

The fundamental mechanisms for pass registration are the
``MachinePassRegistry`` class and subclasses of ``MachinePassRegistryNode``.

An instance of ``MachinePassRegistry`` is used to maintain a list of
``MachinePassRegistryNode`` objects.  This instance maintains the list and
communicates additions and deletions to the command line interface.

An instance of ``MachinePassRegistryNode`` subclass is used to maintain
information provided about a particular pass.  This information includes the
command line name, the command help string and the address of the function used
to create an instance of the pass.  A global static constructor of one of these
instances *registers* with a corresponding ``MachinePassRegistry``, the static
destructor *unregisters*.  Thus a pass that is statically linked in the tool
will be registered at start up.  A dynamically loaded pass will register on
load and unregister at unload.

Using existing registries
-------------------------

There are predefined registries to track instruction scheduling
(``RegisterScheduler``) and register allocation (``RegisterRegAlloc``) machine
passes.  Here we will describe how to *register* a register allocator machine
pass.

Implement your register allocator machine pass.  In your register allocator
``.cpp`` file add the following include:

.. code-block:: c++

  #include "llvm/CodeGen/RegAllocRegistry.h"

Also in your register allocator ``.cpp`` file, define a creator function in the
form:

.. code-block:: c++

  FunctionPass *createMyRegisterAllocator() {
    return new MyRegisterAllocator();
  }

Note that the signature of this function should match the type of
``RegisterRegAlloc::FunctionPassCtor``.  In the same file add the "installing"
declaration, in the form:

.. code-block:: c++

  static RegisterRegAlloc myRegAlloc("myregalloc",
                                     "my register allocator help string",
                                     createMyRegisterAllocator);

Note the two spaces prior to the help string produces a tidy result on the
:option:`-help` query.

.. code-block:: console

  $ llc -help
    ...
    -regalloc                    - Register allocator to use (default=linearscan)
      =linearscan                -   linear scan register allocator
      =local                     -   local register allocator
      =simple                    -   simple register allocator
      =myregalloc                -   my register allocator help string
    ...

And that's it.  The user is now free to use ``-regalloc=myregalloc`` as an
option.  Registering instruction schedulers is similar except use the
``RegisterScheduler`` class.  Note that the
``RegisterScheduler::FunctionPassCtor`` is significantly different from
``RegisterRegAlloc::FunctionPassCtor``.

To force the load/linking of your register allocator into the
:program:`llc`/:program:`lli` tools, add your creator function's global
declaration to ``Passes.h`` and add a "pseudo" call line to
``llvm/Codegen/LinkAllCodegenComponents.h``.

Creating new registries
-----------------------

The easiest way to get started is to clone one of the existing registries; we
recommend ``llvm/CodeGen/RegAllocRegistry.h``.  The key things to modify are
the class name and the ``FunctionPassCtor`` type.

Then you need to declare the registry.  Example: if your pass registry is
``RegisterMyPasses`` then define:

.. code-block:: c++

  MachinePassRegistry RegisterMyPasses::Registry;

And finally, declare the command line option for your passes.  Example:

.. code-block:: c++

  cl::opt<RegisterMyPasses::FunctionPassCtor, false,
          RegisterPassParser<RegisterMyPasses> >
  MyPassOpt("mypass",
            cl::init(&createDefaultMyPass),
            cl::desc("my pass option help"));

Here the command option is "``mypass``", with ``createDefaultMyPass`` as the
default creator.

Using GDB with dynamically loaded passes
----------------------------------------

Unfortunately, using GDB with dynamically loaded passes is not as easy as it
should be.  First of all, you can't set a breakpoint in a shared object that
has not been loaded yet, and second of all there are problems with inlined
functions in shared objects.  Here are some suggestions to debugging your pass
with GDB.

For sake of discussion, I'm going to assume that you are debugging a
transformation invoked by :program:`opt`, although nothing described here
depends on that.

Setting a breakpoint in your pass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First thing you do is start gdb on the opt process:

.. code-block:: console

  $ gdb opt
  GNU gdb 5.0
  Copyright 2000 Free Software Foundation, Inc.
  GDB is free software, covered by the GNU General Public License, and you are
  welcome to change it and/or distribute copies of it under certain conditions.
  Type "show copying" to see the conditions.
  There is absolutely no warranty for GDB.  Type "show warranty" for details.
  This GDB was configured as "sparc-sun-solaris2.6"...
  (gdb)

Note that :program:`opt` has a lot of debugging information in it, so it takes
time to load.  Be patient.  Since we cannot set a breakpoint in our pass yet
(the shared object isn't loaded until runtime), we must execute the process,
and have it stop before it invokes our pass, but after it has loaded the shared
object.  The most foolproof way of doing this is to set a breakpoint in
``PassManager::run`` and then run the process with the arguments you want:

.. code-block:: console

  $ (gdb) break llvm::PassManager::run
  Breakpoint 1 at 0x2413bc: file Pass.cpp, line 70.
  (gdb) run test.bc -load $(LLVMTOP)/llvm/Debug+Asserts/lib/[libname].so -[passoption]
  Starting program: opt test.bc -load $(LLVMTOP)/llvm/Debug+Asserts/lib/[libname].so -[passoption]
  Breakpoint 1, PassManager::run (this=0xffbef174, M=@0x70b298) at Pass.cpp:70
  70      bool PassManager::run(Module &M) { return PM->run(M); }
  (gdb)

Once the :program:`opt` stops in the ``PassManager::run`` method you are now
free to set breakpoints in your pass so that you can trace through execution or
do other standard debugging stuff.

Miscellaneous Problems
^^^^^^^^^^^^^^^^^^^^^^

Once you have the basics down, there are a couple of problems that GDB has,
some with solutions, some without.

* Inline functions have bogus stack information.  In general, GDB does a pretty
  good job getting stack traces and stepping through inline functions.  When a
  pass is dynamically loaded however, it somehow completely loses this
  capability.  The only solution I know of is to de-inline a function (move it
  from the body of a class to a ``.cpp`` file).

* Restarting the program breaks breakpoints.  After following the information
  above, you have succeeded in getting some breakpoints planted in your pass.
  Next thing you know, you restart the program (i.e., you type "``run``" again),
  and you start getting errors about breakpoints being unsettable.  The only
  way I have found to "fix" this problem is to delete the breakpoints that are
  already set in your pass, run the program, and re-set the breakpoints once
  execution stops in ``PassManager::run``.

Hopefully these tips will help with common case debugging situations.  If you'd
like to contribute some tips of your own, just contact `Chris
<mailto:sabre@nondot.org>`_.

Future extensions planned
-------------------------

Although the LLVM Pass Infrastructure is very capable as it stands, and does
some nifty stuff, there are things we'd like to add in the future.  Here is
where we are going:

.. _writing-an-llvm-pass-SMP:

Multithreaded LLVM
^^^^^^^^^^^^^^^^^^

Multiple CPU machines are becoming more common and compilation can never be
fast enough: obviously we should allow for a multithreaded compiler.  Because
of the semantics defined for passes above (specifically they cannot maintain
state across invocations of their ``run*`` methods), a nice clean way to
implement a multithreaded compiler would be for the ``PassManager`` class to
create multiple instances of each pass object, and allow the separate instances
to be hacking on different parts of the program at the same time.

This implementation would prevent each of the passes from having to implement
multithreaded constructs, requiring only the LLVM core to have locking in a few
places (for global resources).  Although this is a simple extension, we simply
haven't had time (or multiprocessor machines, thus a reason) to implement this.
Despite that, we have kept the LLVM passes SMP ready, and you should too.

