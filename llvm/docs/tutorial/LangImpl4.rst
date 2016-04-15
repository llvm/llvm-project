==============================================
Kaleidoscope: Adding JIT and Optimizer Support
==============================================

.. contents::
   :local:

Chapter 4 Introduction
======================

Welcome to Chapter 4 of the "`Implementing a language with
LLVM <index.html>`_" tutorial. Chapters 1-3 described the implementation
of a simple language and added support for generating LLVM IR. This
chapter describes two new techniques: adding optimizer support to your
language, and adding JIT compiler support. These additions will
demonstrate how to get nice, efficient code for the Kaleidoscope
language.

Trivial Constant Folding
========================

Our demonstration for Chapter 3 is elegant and easy to extend.
Unfortunately, it does not produce wonderful code. The IRBuilder,
however, does give us obvious optimizations when compiling simple code:

::

    ready> def test(x) 1+2+x;
    Read function definition:
    define double @test(double %x) {
    entry:
            %addtmp = fadd double 3.000000e+00, %x
            ret double %addtmp
    }

This code is not a literal transcription of the AST built by parsing the
input. That would be:

::

    ready> def test(x) 1+2+x;
    Read function definition:
    define double @test(double %x) {
    entry:
            %addtmp = fadd double 2.000000e+00, 1.000000e+00
            %addtmp1 = fadd double %addtmp, %x
            ret double %addtmp1
    }

Constant folding, as seen above, in particular, is a very common and
very important optimization: so much so that many language implementors
implement constant folding support in their AST representation.

With LLVM, you don't need this support in the AST. Since all calls to
build LLVM IR go through the LLVM IR builder, the builder itself checked
to see if there was a constant folding opportunity when you call it. If
so, it just does the constant fold and return the constant instead of
creating an instruction.

Well, that was easy :). In practice, we recommend always using
``IRBuilder`` when generating code like this. It has no "syntactic
overhead" for its use (you don't have to uglify your compiler with
constant checks everywhere) and it can dramatically reduce the amount of
LLVM IR that is generated in some cases (particular for languages with a
macro preprocessor or that use a lot of constants).

On the other hand, the ``IRBuilder`` is limited by the fact that it does
all of its analysis inline with the code as it is built. If you take a
slightly more complex example:

::

    ready> def test(x) (1+2+x)*(x+(1+2));
    ready> Read function definition:
    define double @test(double %x) {
    entry:
            %addtmp = fadd double 3.000000e+00, %x
            %addtmp1 = fadd double %x, 3.000000e+00
            %multmp = fmul double %addtmp, %addtmp1
            ret double %multmp
    }

In this case, the LHS and RHS of the multiplication are the same value.
We'd really like to see this generate "``tmp = x+3; result = tmp*tmp;``"
instead of computing "``x+3``" twice.

Unfortunately, no amount of local analysis will be able to detect and
correct this. This requires two transformations: reassociation of
expressions (to make the add's lexically identical) and Common
Subexpression Elimination (CSE) to delete the redundant add instruction.
Fortunately, LLVM provides a broad range of optimizations that you can
use, in the form of "passes".

LLVM Optimization Passes
========================

LLVM provides many optimization passes, which do many different sorts of
things and have different tradeoffs. Unlike other systems, LLVM doesn't
hold to the mistaken notion that one set of optimizations is right for
all languages and for all situations. LLVM allows a compiler implementor
to make complete decisions about what optimizations to use, in which
order, and in what situation.

As a concrete example, LLVM supports both "whole module" passes, which
look across as large of body of code as they can (often a whole file,
but if run at link time, this can be a substantial portion of the whole
program). It also supports and includes "per-function" passes which just
operate on a single function at a time, without looking at other
functions. For more information on passes and how they are run, see the
`How to Write a Pass <../WritingAnLLVMPass.html>`_ document and the
`List of LLVM Passes <../Passes.html>`_.

For Kaleidoscope, we are currently generating functions on the fly, one
at a time, as the user types them in. We aren't shooting for the
ultimate optimization experience in this setting, but we also want to
catch the easy and quick stuff where possible. As such, we will choose
to run a few per-function optimizations as the user types the function
in. If we wanted to make a "static Kaleidoscope compiler", we would use
exactly the code we have now, except that we would defer running the
optimizer until the entire file has been parsed.

In order to get per-function optimizations going, we need to set up a
`FunctionPassManager <../WritingAnLLVMPass.html#what-passmanager-doesr>`_ to hold
and organize the LLVM optimizations that we want to run. Once we have
that, we can add a set of optimizations to run. We'll need a new
FunctionPassManager for each module that we want to optimize, so we'll
write a function to create and initialize both the module and pass manager
for us:

.. code-block:: c++

    void InitializeModuleAndPassManager(void) {
      // Open a new module.
      Context LLVMContext;
      TheModule = llvm::make_unique<Module>("my cool jit", LLVMContext);
      TheModule->setDataLayout(TheJIT->getTargetMachine().createDataLayout());

      // Create a new pass manager attached to it.
      TheFPM = llvm::make_unique<FunctionPassManager>(TheModule.get());

      // Provide basic AliasAnalysis support for GVN.
      TheFPM.add(createBasicAliasAnalysisPass());
      // Do simple "peephole" optimizations and bit-twiddling optzns.
      TheFPM.add(createInstructionCombiningPass());
      // Reassociate expressions.
      TheFPM.add(createReassociatePass());
      // Eliminate Common SubExpressions.
      TheFPM.add(createGVNPass());
      // Simplify the control flow graph (deleting unreachable blocks, etc).
      TheFPM.add(createCFGSimplificationPass());

      TheFPM.doInitialization();
    }

This code initializes the global module ``TheModule``, and the function pass
manager ``TheFPM``, which is attached to ``TheModule``. Once the pass manager is
set up, we use a series of "add" calls to add a bunch of LLVM passes.

In this case, we choose to add five passes: one analysis pass (alias analysis),
and four optimization passes. The passes we choose here are a pretty standard set
of "cleanup" optimizations that are useful for a wide variety of code. I won't
delve into what they do but, believe me, they are a good starting place :).

Once the PassManager is set up, we need to make use of it. We do this by
running it after our newly created function is constructed (in
``FunctionAST::codegen()``), but before it is returned to the client:

.. code-block:: c++

      if (Value *RetVal = Body->codegen()) {
        // Finish off the function.
        Builder.CreateRet(RetVal);

        // Validate the generated code, checking for consistency.
        verifyFunction(*TheFunction);

        // Optimize the function.
        TheFPM->run(*TheFunction);

        return TheFunction;
      }

As you can see, this is pretty straightforward. The
``FunctionPassManager`` optimizes and updates the LLVM Function\* in
place, improving (hopefully) its body. With this in place, we can try
our test above again:

::

    ready> def test(x) (1+2+x)*(x+(1+2));
    ready> Read function definition:
    define double @test(double %x) {
    entry:
            %addtmp = fadd double %x, 3.000000e+00
            %multmp = fmul double %addtmp, %addtmp
            ret double %multmp
    }

As expected, we now get our nicely optimized code, saving a floating
point add instruction from every execution of this function.

LLVM provides a wide variety of optimizations that can be used in
certain circumstances. Some `documentation about the various
passes <../Passes.html>`_ is available, but it isn't very complete.
Another good source of ideas can come from looking at the passes that
``Clang`` runs to get started. The "``opt``" tool allows you to
experiment with passes from the command line, so you can see if they do
anything.

Now that we have reasonable code coming out of our front-end, lets talk
about executing it!

Adding a JIT Compiler
=====================

Code that is available in LLVM IR can have a wide variety of tools
applied to it. For example, you can run optimizations on it (as we did
above), you can dump it out in textual or binary forms, you can compile
the code to an assembly file (.s) for some target, or you can JIT
compile it. The nice thing about the LLVM IR representation is that it
is the "common currency" between many different parts of the compiler.

In this section, we'll add JIT compiler support to our interpreter. The
basic idea that we want for Kaleidoscope is to have the user enter
function bodies as they do now, but immediately evaluate the top-level
expressions they type in. For example, if they type in "1 + 2;", we
should evaluate and print out 3. If they define a function, they should
be able to call it from the command line.

In order to do this, we first declare and initialize the JIT. This is
done by adding a global variable ``TheJIT``, and initializing it in
``main``:

.. code-block:: c++

    static std::unique_ptr<KaleidoscopeJIT> TheJIT;
    ...
    int main() {
      ..
      TheJIT = llvm::make_unique<KaleidoscopeJIT>();

      // Run the main "interpreter loop" now.
      MainLoop();

      return 0;
    }

The KaleidoscopeJIT class is a simple JIT built specifically for these
tutorials. In later chapters we will look at how it works and extend it with
new features, but for now we will take it as given. Its API is very simple::
``addModule`` adds an LLVM IR module to the JIT, making its functions
available for execution; ``removeModule`` removes a module, freeing any
memory associated with the code in that module; and ``findSymbol`` allows us
to look up pointers to the compiled code.

We can take this simple API and change our code that parses top-level expressions to
look like this:

.. code-block:: c++

    static void HandleTopLevelExpression() {
      // Evaluate a top-level expression into an anonymous function.
      if (auto FnAST = ParseTopLevelExpr()) {
        if (FnAST->codegen()) {

          // JIT the module containing the anonymous expression, keeping a handle so
          // we can free it later.
          auto H = TheJIT->addModule(std::move(TheModule));
          InitializeModuleAndPassManager();

          // Search the JIT for the __anon_expr symbol.
          auto ExprSymbol = TheJIT->findSymbol("__anon_expr");
          assert(ExprSymbol && "Function not found");

          // Get the symbol's address and cast it to the right type (takes no
          // arguments, returns a double) so we can call it as a native function.
          double (*FP)() = (double (*)())(intptr_t)ExprSymbol.getAddress();
          fprintf(stderr, "Evaluated to %f\n", FP());

          // Delete the anonymous expression module from the JIT.
          TheJIT->removeModule(H);
        }

If parsing and codegen succeeed, the next step is to add the module containing
the top-level expression to the JIT. We do this by calling addModule, which
triggers code generation for all the functions in the module, and returns a
handle that can be used to remove the module from the JIT later. Once the module
has been added to the JIT it can no longer be modified, so we also open a new
module to hold subsequent code by calling ``InitializeModuleAndPassManager()``.

Once we've added the module to the JIT we need to get a pointer to the final
generated code. We do this by calling the JIT's findSymbol method, and passing
the name of the top-level expression function: ``__anon_expr``. Since we just
added this function, we assert that findSymbol returned a result.

Next, we get the in-memory address of the ``__anon_expr`` function by calling
``getAddress()`` on the symbol. Recall that we compile top-level expressions
into a self-contained LLVM function that takes no arguments and returns the
computed double. Because the LLVM JIT compiler matches the native platform ABI,
this means that you can just cast the result pointer to a function pointer of
that type and call it directly. This means, there is no difference between JIT
compiled code and native machine code that is statically linked into your
application.

Finally, since we don't support re-evaluation of top-level expressions, we
remove the module from the JIT when we're done to free the associated memory.
Recall, however, that the module we created a few lines earlier (via
``InitializeModuleAndPassManager``) is still open and waiting for new code to be
added.

With just these two changes, lets see how Kaleidoscope works now!

::

    ready> 4+5;
    Read top-level expression:
    define double @0() {
    entry:
      ret double 9.000000e+00
    }

    Evaluated to 9.000000

Well this looks like it is basically working. The dump of the function
shows the "no argument function that always returns double" that we
synthesize for each top-level expression that is typed in. This
demonstrates very basic functionality, but can we do more?

::

    ready> def testfunc(x y) x + y*2;
    Read function definition:
    define double @testfunc(double %x, double %y) {
    entry:
      %multmp = fmul double %y, 2.000000e+00
      %addtmp = fadd double %multmp, %x
      ret double %addtmp
    }

    ready> testfunc(4, 10);
    Read top-level expression:
    define double @1() {
    entry:
      %calltmp = call double @testfunc(double 4.000000e+00, double 1.000000e+01)
      ret double %calltmp
    }

    Evaluated to 24.000000

    ready> testfunc(5, 10);
    ready> LLVM ERROR: Program used external function 'testfunc' which could not be resolved!


Function definitions and calls also work, but something went very wrong on that
last line. The call looks valid, so what happened? As you may have guessed from
the the API a Module is a unit of allocation for the JIT, and testfunc was part
of the same module that contained anonymous expression. When we removed that
module from the JIT to free the memory for the anonymous expression, we deleted
the definition of ``testfunc`` along with it. Then, when we tried to call
testfunc a second time, the JIT could no longer find it.

The easiest way to fix this is to put the anonymous expression in a separate
module from the rest of the function definitions. The JIT will happily resolve
function calls across module boundaries, as long as each of the functions called
has a prototype, and is added to the JIT before it is called. By putting the
anonymous expression in a different module we can delete it without affecting
the rest of the functions.

In fact, we're going to go a step further and put every function in its own
module. Doing so allows us to exploit a useful property of the KaleidoscopeJIT
that will make our environment more REPL-like: Functions can be added to the
JIT more than once (unlike a module where every function must have a unique
definition). When you look up a symbol in KaleidoscopeJIT it will always return
the most recent definition:

::

    ready> def foo(x) x + 1;
    Read function definition:
    define double @foo(double %x) {
    entry:
      %addtmp = fadd double %x, 1.000000e+00
      ret double %addtmp
    }

    ready> foo(2);
    Evaluated to 3.000000

    ready> def foo(x) x + 2;
    define double @foo(double %x) {
    entry:
      %addtmp = fadd double %x, 2.000000e+00
      ret double %addtmp
    }

    ready> foo(2);
    Evaluated to 4.000000


To allow each function to live in its own module we'll need a way to
re-generate previous function declarations into each new module we open:

.. code-block:: c++

    static std::unique_ptr<KaleidoscopeJIT> TheJIT;

    ...

    Function *getFunction(std::string Name) {
      // First, see if the function has already been added to the current module.
      if (auto *F = TheModule->getFunction(Name))
        return F;

      // If not, check whether we can codegen the declaration from some existing
      // prototype.
      auto FI = FunctionProtos.find(Name);
      if (FI != FunctionProtos.end())
        return FI->second->codegen();

      // If no existing prototype exists, return null.
      return nullptr;
    }

    ...

    Value *CallExprAST::codegen() {
      // Look up the name in the global module table.
      Function *CalleeF = getFunction(Callee);

    ...

    Function *FunctionAST::codegen() {
      // Transfer ownership of the prototype to the FunctionProtos map, but keep a
      // reference to it for use below.
      auto &P = *Proto;
      FunctionProtos[Proto->getName()] = std::move(Proto);
      Function *TheFunction = getFunction(P.getName());
      if (!TheFunction)
        return nullptr;


To enable this, we'll start by adding a new global, ``FunctionProtos``, that
holds the most recent prototype for each function. We'll also add a convenience
method, ``getFunction()``, to replace calls to ``TheModule->getFunction()``.
Our convenience method searches ``TheModule`` for an existing function
declaration, falling back to generating a new declaration from FunctionProtos if
it doesn't find one. In ``CallExprAST::codegen()`` we just need to replace the
call to ``TheModule->getFunction()``. In ``FunctionAST::codegen()`` we need to
update the FunctionProtos map first, then call ``getFunction()``. With this
done, we can always obtain a function declaration in the current module for any
previously declared function.

We also need to update HandleDefinition and HandleExtern:

.. code-block:: c++

    static void HandleDefinition() {
      if (auto FnAST = ParseDefinition()) {
        if (auto *FnIR = FnAST->codegen()) {
          fprintf(stderr, "Read function definition:");
          FnIR->dump();
          TheJIT->addModule(std::move(TheModule));
          InitializeModuleAndPassManager();
        }
      } else {
        // Skip token for error recovery.
         getNextToken();
      }
    }

    static void HandleExtern() {
      if (auto ProtoAST = ParseExtern()) {
        if (auto *FnIR = ProtoAST->codegen()) {
          fprintf(stderr, "Read extern: ");
          FnIR->dump();
          FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
        }
      } else {
        // Skip token for error recovery.
        getNextToken();
      }
    }

In HandleDefinition, we add two lines to transfer the newly defined function to
the JIT and open a new module. In HandleExtern, we just need to add one line to
add the prototype to FunctionProtos.

With these changes made, lets try our REPL again (I removed the dump of the
anonymous functions this time, you should get the idea by now :) :

::

    ready> def foo(x) x + 1;
    ready> foo(2);
    Evaluated to 3.000000

    ready> def foo(x) x + 2;
    ready> foo(2);
    Evaluated to 4.000000

It works!

Even with this simple code, we get some surprisingly powerful capabilities -
check this out:

::

    ready> extern sin(x);
    Read extern:
    declare double @sin(double)

    ready> extern cos(x);
    Read extern:
    declare double @cos(double)

    ready> sin(1.0);
    Read top-level expression:
    define double @2() {
    entry:
      ret double 0x3FEAED548F090CEE
    }

    Evaluated to 0.841471

    ready> def foo(x) sin(x)*sin(x) + cos(x)*cos(x);
    Read function definition:
    define double @foo(double %x) {
    entry:
      %calltmp = call double @sin(double %x)
      %multmp = fmul double %calltmp, %calltmp
      %calltmp2 = call double @cos(double %x)
      %multmp4 = fmul double %calltmp2, %calltmp2
      %addtmp = fadd double %multmp, %multmp4
      ret double %addtmp
    }

    ready> foo(4.0);
    Read top-level expression:
    define double @3() {
    entry:
      %calltmp = call double @foo(double 4.000000e+00)
      ret double %calltmp
    }

    Evaluated to 1.000000

Whoa, how does the JIT know about sin and cos? The answer is surprisingly
simple: The KaleidoscopeJIT has a straightforward symbol resolution rule that
it uses to find symbols that aren't available in any given module: First
it searches all the modules that have already been added to the JIT, from the
most recent to the oldest, to find the newest definition. If no definition is
found inside the JIT, it falls back to calling "``dlsym("sin")``" on the
Kaleidoscope process itself. Since "``sin``" is defined within the JIT's
address space, it simply patches up calls in the module to call the libm
version of ``sin`` directly.

In the future we'll see how tweaking this symbol resolution rule can be used to
enable all sorts of useful features, from security (restricting the set of
symbols available to JIT'd code), to dynamic code generation based on symbol
names, and even lazy compilation.

One immediate benefit of the symbol resolution rule is that we can now extend
the language by writing arbitrary C++ code to implement operations. For example,
if we add:

.. code-block:: c++

    /// putchard - putchar that takes a double and returns 0.
    extern "C" double putchard(double X) {
      fputc((char)X, stderr);
      return 0;
    }

Now we can produce simple output to the console by using things like:
"``extern putchard(x); putchard(120);``", which prints a lowercase 'x'
on the console (120 is the ASCII code for 'x'). Similar code could be
used to implement file I/O, console input, and many other capabilities
in Kaleidoscope.

This completes the JIT and optimizer chapter of the Kaleidoscope
tutorial. At this point, we can compile a non-Turing-complete
programming language, optimize and JIT compile it in a user-driven way.
Next up we'll look into `extending the language with control flow
constructs <LangImpl5.html>`_, tackling some interesting LLVM IR issues
along the way.

Full Code Listing
=================

Here is the complete code listing for our running example, enhanced with
the LLVM JIT and optimizer. To build this example, use:

.. code-block:: bash

    # Compile
    clang++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core mcjit native` -O3 -o toy
    # Run
    ./toy

If you are compiling this on Linux, make sure to add the "-rdynamic"
option as well. This makes sure that the external functions are resolved
properly at runtime.

Here is the code:

.. literalinclude:: ../../examples/Kaleidoscope/Chapter4/toy.cpp
   :language: c++

`Next: Extending the language: control flow <LangImpl5.html>`_

