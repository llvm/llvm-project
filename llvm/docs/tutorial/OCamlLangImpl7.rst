=======================================================
Kaleidoscope: Extending the Language: Mutable Variables
=======================================================

.. contents::
   :local:

Chapter 7 Introduction
======================

Welcome to Chapter 7 of the "`Implementing a language with
LLVM <index.html>`_" tutorial. In chapters 1 through 6, we've built a
very respectable, albeit simple, `functional programming
language <http://en.wikipedia.org/wiki/Functional_programming>`_. In our
journey, we learned some parsing techniques, how to build and represent
an AST, how to build LLVM IR, and how to optimize the resultant code as
well as JIT compile it.

While Kaleidoscope is interesting as a functional language, the fact
that it is functional makes it "too easy" to generate LLVM IR for it. In
particular, a functional language makes it very easy to build LLVM IR
directly in `SSA
form <http://en.wikipedia.org/wiki/Static_single_assignment_form>`_.
Since LLVM requires that the input code be in SSA form, this is a very
nice property and it is often unclear to newcomers how to generate code
for an imperative language with mutable variables.

The short (and happy) summary of this chapter is that there is no need
for your front-end to build SSA form: LLVM provides highly tuned and
well tested support for this, though the way it works is a bit
unexpected for some.

Why is this a hard problem?
===========================

To understand why mutable variables cause complexities in SSA
construction, consider this extremely simple C example:

.. code-block:: c

    int G, H;
    int test(_Bool Condition) {
      int X;
      if (Condition)
        X = G;
      else
        X = H;
      return X;
    }

In this case, we have the variable "X", whose value depends on the path
executed in the program. Because there are two different possible values
for X before the return instruction, a PHI node is inserted to merge the
two values. The LLVM IR that we want for this example looks like this:

.. code-block:: llvm

    @G = weak global i32 0   ; type of @G is i32*
    @H = weak global i32 0   ; type of @H is i32*

    define i32 @test(i1 %Condition) {
    entry:
      br i1 %Condition, label %cond_true, label %cond_false

    cond_true:
      %X.0 = load i32* @G
      br label %cond_next

    cond_false:
      %X.1 = load i32* @H
      br label %cond_next

    cond_next:
      %X.2 = phi i32 [ %X.1, %cond_false ], [ %X.0, %cond_true ]
      ret i32 %X.2
    }

In this example, the loads from the G and H global variables are
explicit in the LLVM IR, and they live in the then/else branches of the
if statement (cond\_true/cond\_false). In order to merge the incoming
values, the X.2 phi node in the cond\_next block selects the right value
to use based on where control flow is coming from: if control flow comes
from the cond\_false block, X.2 gets the value of X.1. Alternatively, if
control flow comes from cond\_true, it gets the value of X.0. The intent
of this chapter is not to explain the details of SSA form. For more
information, see one of the many `online
references <http://en.wikipedia.org/wiki/Static_single_assignment_form>`_.

The question for this article is "who places the phi nodes when lowering
assignments to mutable variables?". The issue here is that LLVM
*requires* that its IR be in SSA form: there is no "non-ssa" mode for
it. However, SSA construction requires non-trivial algorithms and data
structures, so it is inconvenient and wasteful for every front-end to
have to reproduce this logic.

Memory in LLVM
==============

The 'trick' here is that while LLVM does require all register values to
be in SSA form, it does not require (or permit) memory objects to be in
SSA form. In the example above, note that the loads from G and H are
direct accesses to G and H: they are not renamed or versioned. This
differs from some other compiler systems, which do try to version memory
objects. In LLVM, instead of encoding dataflow analysis of memory into
the LLVM IR, it is handled with `Analysis
Passes <../WritingAnLLVMPass.html>`_ which are computed on demand.

With this in mind, the high-level idea is that we want to make a stack
variable (which lives in memory, because it is on the stack) for each
mutable object in a function. To take advantage of this trick, we need
to talk about how LLVM represents stack variables.

In LLVM, all memory accesses are explicit with load/store instructions,
and it is carefully designed not to have (or need) an "address-of"
operator. Notice how the type of the @G/@H global variables is actually
"i32\*" even though the variable is defined as "i32". What this means is
that @G defines *space* for an i32 in the global data area, but its
*name* actually refers to the address for that space. Stack variables
work the same way, except that instead of being declared with global
variable definitions, they are declared with the `LLVM alloca
instruction <../LangRef.html#alloca-instruction>`_:

.. code-block:: llvm

    define i32 @example() {
    entry:
      %X = alloca i32           ; type of %X is i32*.
      ...
      %tmp = load i32* %X       ; load the stack value %X from the stack.
      %tmp2 = add i32 %tmp, 1   ; increment it
      store i32 %tmp2, i32* %X  ; store it back
      ...

This code shows an example of how you can declare and manipulate a stack
variable in the LLVM IR. Stack memory allocated with the alloca
instruction is fully general: you can pass the address of the stack slot
to functions, you can store it in other variables, etc. In our example
above, we could rewrite the example to use the alloca technique to avoid
using a PHI node:

.. code-block:: llvm

    @G = weak global i32 0   ; type of @G is i32*
    @H = weak global i32 0   ; type of @H is i32*

    define i32 @test(i1 %Condition) {
    entry:
      %X = alloca i32           ; type of %X is i32*.
      br i1 %Condition, label %cond_true, label %cond_false

    cond_true:
      %X.0 = load i32* @G
            store i32 %X.0, i32* %X   ; Update X
      br label %cond_next

    cond_false:
      %X.1 = load i32* @H
            store i32 %X.1, i32* %X   ; Update X
      br label %cond_next

    cond_next:
      %X.2 = load i32* %X       ; Read X
      ret i32 %X.2
    }

With this, we have discovered a way to handle arbitrary mutable
variables without the need to create Phi nodes at all:

#. Each mutable variable becomes a stack allocation.
#. Each read of the variable becomes a load from the stack.
#. Each update of the variable becomes a store to the stack.
#. Taking the address of a variable just uses the stack address
   directly.

While this solution has solved our immediate problem, it introduced
another one: we have now apparently introduced a lot of stack traffic
for very simple and common operations, a major performance problem.
Fortunately for us, the LLVM optimizer has a highly-tuned optimization
pass named "mem2reg" that handles this case, promoting allocas like this
into SSA registers, inserting Phi nodes as appropriate. If you run this
example through the pass, for example, you'll get:

.. code-block:: bash

    $ llvm-as < example.ll | opt -mem2reg | llvm-dis
    @G = weak global i32 0
    @H = weak global i32 0

    define i32 @test(i1 %Condition) {
    entry:
      br i1 %Condition, label %cond_true, label %cond_false

    cond_true:
      %X.0 = load i32* @G
      br label %cond_next

    cond_false:
      %X.1 = load i32* @H
      br label %cond_next

    cond_next:
      %X.01 = phi i32 [ %X.1, %cond_false ], [ %X.0, %cond_true ]
      ret i32 %X.01
    }

The mem2reg pass implements the standard "iterated dominance frontier"
algorithm for constructing SSA form and has a number of optimizations
that speed up (very common) degenerate cases. The mem2reg optimization
pass is the answer to dealing with mutable variables, and we highly
recommend that you depend on it. Note that mem2reg only works on
variables in certain circumstances:

#. mem2reg is alloca-driven: it looks for allocas and if it can handle
   them, it promotes them. It does not apply to global variables or heap
   allocations.
#. mem2reg only looks for alloca instructions in the entry block of the
   function. Being in the entry block guarantees that the alloca is only
   executed once, which makes analysis simpler.
#. mem2reg only promotes allocas whose uses are direct loads and stores.
   If the address of the stack object is passed to a function, or if any
   funny pointer arithmetic is involved, the alloca will not be
   promoted.
#. mem2reg only works on allocas of `first
   class <../LangRef.html#first-class-types>`_ values (such as pointers,
   scalars and vectors), and only if the array size of the allocation is
   1 (or missing in the .ll file). mem2reg is not capable of promoting
   structs or arrays to registers. Note that the "scalarrepl" pass is
   more powerful and can promote structs, "unions", and arrays in many
   cases.

All of these properties are easy to satisfy for most imperative
languages, and we'll illustrate it below with Kaleidoscope. The final
question you may be asking is: should I bother with this nonsense for my
front-end? Wouldn't it be better if I just did SSA construction
directly, avoiding use of the mem2reg optimization pass? In short, we
strongly recommend that you use this technique for building SSA form,
unless there is an extremely good reason not to. Using this technique
is:

-  Proven and well tested: clang uses this technique
   for local mutable variables. As such, the most common clients of LLVM
   are using this to handle a bulk of their variables. You can be sure
   that bugs are found fast and fixed early.
-  Extremely Fast: mem2reg has a number of special cases that make it
   fast in common cases as well as fully general. For example, it has
   fast-paths for variables that are only used in a single block,
   variables that only have one assignment point, good heuristics to
   avoid insertion of unneeded phi nodes, etc.
-  Needed for debug info generation: `Debug information in
   LLVM <../SourceLevelDebugging.html>`_ relies on having the address of
   the variable exposed so that debug info can be attached to it. This
   technique dovetails very naturally with this style of debug info.

If nothing else, this makes it much easier to get your front-end up and
running, and is very simple to implement. Lets extend Kaleidoscope with
mutable variables now!

Mutable Variables in Kaleidoscope
=================================

Now that we know the sort of problem we want to tackle, lets see what
this looks like in the context of our little Kaleidoscope language.
We're going to add two features:

#. The ability to mutate variables with the '=' operator.
#. The ability to define new variables.

While the first item is really what this is about, we only have
variables for incoming arguments as well as for induction variables, and
redefining those only goes so far :). Also, the ability to define new
variables is a useful thing regardless of whether you will be mutating
them. Here's a motivating example that shows how we could use these:

::

    # Define ':' for sequencing: as a low-precedence operator that ignores operands
    # and just returns the RHS.
    def binary : 1 (x y) y;

    # Recursive fib, we could do this before.
    def fib(x)
      if (x < 3) then
        1
      else
        fib(x-1)+fib(x-2);

    # Iterative fib.
    def fibi(x)
      var a = 1, b = 1, c in
      (for i = 3, i < x in
         c = a + b :
         a = b :
         b = c) :
      b;

    # Call it.
    fibi(10);

In order to mutate variables, we have to change our existing variables
to use the "alloca trick". Once we have that, we'll add our new
operator, then extend Kaleidoscope to support new variable definitions.

Adjusting Existing Variables for Mutation
=========================================

The symbol table in Kaleidoscope is managed at code generation time by
the '``named_values``' map. This map currently keeps track of the LLVM
"Value\*" that holds the double value for the named variable. In order
to support mutation, we need to change this slightly, so that it
``named_values`` holds the *memory location* of the variable in
question. Note that this change is a refactoring: it changes the
structure of the code, but does not (by itself) change the behavior of
the compiler. All of these changes are isolated in the Kaleidoscope code
generator.

At this point in Kaleidoscope's development, it only supports variables
for two things: incoming arguments to functions and the induction
variable of 'for' loops. For consistency, we'll allow mutation of these
variables in addition to other user-defined variables. This means that
these will both need memory locations.

To start our transformation of Kaleidoscope, we'll change the
``named_values`` map so that it maps to AllocaInst\* instead of Value\*.
Once we do this, the C++ compiler will tell us what parts of the code we
need to update:

**Note:** the ocaml bindings currently model both ``Value*``'s and
``AllocInst*``'s as ``Llvm.llvalue``'s, but this may change in the future
to be more type safe.

.. code-block:: ocaml

    let named_values:(string, llvalue) Hashtbl.t = Hashtbl.create 10

Also, since we will need to create these alloca's, we'll use a helper
function that ensures that the allocas are created in the entry block of
the function:

.. code-block:: ocaml

    (* Create an alloca instruction in the entry block of the function. This
     * is used for mutable variables etc. *)
    let create_entry_block_alloca the_function var_name =
      let builder = builder_at (instr_begin (entry_block the_function)) in
      build_alloca double_type var_name builder

This funny looking code creates an ``Llvm.llbuilder`` object that is
pointing at the first instruction of the entry block. It then creates an
alloca with the expected name and returns it. Because all values in
Kaleidoscope are doubles, there is no need to pass in a type to use.

With this in place, the first functionality change we want to make is to
variable references. In our new scheme, variables live on the stack, so
code generating a reference to them actually needs to produce a load
from the stack slot:

.. code-block:: ocaml

    let rec codegen_expr = function
      ...
      | Ast.Variable name ->
          let v = try Hashtbl.find named_values name with
            | Not_found -> raise (Error "unknown variable name")
          in
          (* Load the value. *)
          build_load v name builder

As you can see, this is pretty straightforward. Now we need to update
the things that define the variables to set up the alloca. We'll start
with ``codegen_expr Ast.For ...`` (see the `full code listing <#id1>`_
for the unabridged code):

.. code-block:: ocaml

      | Ast.For (var_name, start, end_, step, body) ->
          let the_function = block_parent (insertion_block builder) in

          (* Create an alloca for the variable in the entry block. *)
          let alloca = create_entry_block_alloca the_function var_name in

          (* Emit the start code first, without 'variable' in scope. *)
          let start_val = codegen_expr start in

          (* Store the value into the alloca. *)
          ignore(build_store start_val alloca builder);

          ...

          (* Within the loop, the variable is defined equal to the PHI node. If it
           * shadows an existing variable, we have to restore it, so save it
           * now. *)
          let old_val =
            try Some (Hashtbl.find named_values var_name) with Not_found -> None
          in
          Hashtbl.add named_values var_name alloca;

          ...

          (* Compute the end condition. *)
          let end_cond = codegen_expr end_ in

          (* Reload, increment, and restore the alloca. This handles the case where
           * the body of the loop mutates the variable. *)
          let cur_var = build_load alloca var_name builder in
          let next_var = build_add cur_var step_val "nextvar" builder in
          ignore(build_store next_var alloca builder);
          ...

This code is virtually identical to the code `before we allowed mutable
variables <OCamlLangImpl5.html#code-generation-for-the-for-loop>`_. The big difference is that
we no longer have to construct a PHI node, and we use load/store to
access the variable as needed.

To support mutable argument variables, we need to also make allocas for
them. The code for this is also pretty simple:

.. code-block:: ocaml

    (* Create an alloca for each argument and register the argument in the symbol
     * table so that references to it will succeed. *)
    let create_argument_allocas the_function proto =
      let args = match proto with
        | Ast.Prototype (_, args) | Ast.BinOpPrototype (_, args, _) -> args
      in
      Array.iteri (fun i ai ->
        let var_name = args.(i) in
        (* Create an alloca for this variable. *)
        let alloca = create_entry_block_alloca the_function var_name in

        (* Store the initial value into the alloca. *)
        ignore(build_store ai alloca builder);

        (* Add arguments to variable symbol table. *)
        Hashtbl.add named_values var_name alloca;
      ) (params the_function)

For each argument, we make an alloca, store the input value to the
function into the alloca, and register the alloca as the memory location
for the argument. This method gets invoked by ``Codegen.codegen_func``
right after it sets up the entry block for the function.

The final missing piece is adding the mem2reg pass, which allows us to
get good codegen once again:

.. code-block:: ocaml

    let main () =
      ...
      let the_fpm = PassManager.create_function Codegen.the_module in

      (* Set up the optimizer pipeline.  Start with registering info about how the
       * target lays out data structures. *)
      DataLayout.add (ExecutionEngine.target_data the_execution_engine) the_fpm;

      (* Promote allocas to registers. *)
      add_memory_to_register_promotion the_fpm;

      (* Do simple "peephole" optimizations and bit-twiddling optzn. *)
      add_instruction_combining the_fpm;

      (* reassociate expressions. *)
      add_reassociation the_fpm;

It is interesting to see what the code looks like before and after the
mem2reg optimization runs. For example, this is the before/after code
for our recursive fib function. Before the optimization:

.. code-block:: llvm

    define double @fib(double %x) {
    entry:
      %x1 = alloca double
      store double %x, double* %x1
      %x2 = load double* %x1
      %cmptmp = fcmp ult double %x2, 3.000000e+00
      %booltmp = uitofp i1 %cmptmp to double
      %ifcond = fcmp one double %booltmp, 0.000000e+00
      br i1 %ifcond, label %then, label %else

    then:    ; preds = %entry
      br label %ifcont

    else:    ; preds = %entry
      %x3 = load double* %x1
      %subtmp = fsub double %x3, 1.000000e+00
      %calltmp = call double @fib(double %subtmp)
      %x4 = load double* %x1
      %subtmp5 = fsub double %x4, 2.000000e+00
      %calltmp6 = call double @fib(double %subtmp5)
      %addtmp = fadd double %calltmp, %calltmp6
      br label %ifcont

    ifcont:    ; preds = %else, %then
      %iftmp = phi double [ 1.000000e+00, %then ], [ %addtmp, %else ]
      ret double %iftmp
    }

Here there is only one variable (x, the input argument) but you can
still see the extremely simple-minded code generation strategy we are
using. In the entry block, an alloca is created, and the initial input
value is stored into it. Each reference to the variable does a reload
from the stack. Also, note that we didn't modify the if/then/else
expression, so it still inserts a PHI node. While we could make an
alloca for it, it is actually easier to create a PHI node for it, so we
still just make the PHI.

Here is the code after the mem2reg pass runs:

.. code-block:: llvm

    define double @fib(double %x) {
    entry:
      %cmptmp = fcmp ult double %x, 3.000000e+00
      %booltmp = uitofp i1 %cmptmp to double
      %ifcond = fcmp one double %booltmp, 0.000000e+00
      br i1 %ifcond, label %then, label %else

    then:
      br label %ifcont

    else:
      %subtmp = fsub double %x, 1.000000e+00
      %calltmp = call double @fib(double %subtmp)
      %subtmp5 = fsub double %x, 2.000000e+00
      %calltmp6 = call double @fib(double %subtmp5)
      %addtmp = fadd double %calltmp, %calltmp6
      br label %ifcont

    ifcont:    ; preds = %else, %then
      %iftmp = phi double [ 1.000000e+00, %then ], [ %addtmp, %else ]
      ret double %iftmp
    }

This is a trivial case for mem2reg, since there are no redefinitions of
the variable. The point of showing this is to calm your tension about
inserting such blatent inefficiencies :).

After the rest of the optimizers run, we get:

.. code-block:: llvm

    define double @fib(double %x) {
    entry:
      %cmptmp = fcmp ult double %x, 3.000000e+00
      %booltmp = uitofp i1 %cmptmp to double
      %ifcond = fcmp ueq double %booltmp, 0.000000e+00
      br i1 %ifcond, label %else, label %ifcont

    else:
      %subtmp = fsub double %x, 1.000000e+00
      %calltmp = call double @fib(double %subtmp)
      %subtmp5 = fsub double %x, 2.000000e+00
      %calltmp6 = call double @fib(double %subtmp5)
      %addtmp = fadd double %calltmp, %calltmp6
      ret double %addtmp

    ifcont:
      ret double 1.000000e+00
    }

Here we see that the simplifycfg pass decided to clone the return
instruction into the end of the 'else' block. This allowed it to
eliminate some branches and the PHI node.

Now that all symbol table references are updated to use stack variables,
we'll add the assignment operator.

New Assignment Operator
=======================

With our current framework, adding a new assignment operator is really
simple. We will parse it just like any other binary operator, but handle
it internally (instead of allowing the user to define it). The first
step is to set a precedence:

.. code-block:: ocaml

    let main () =
      (* Install standard binary operators.
       * 1 is the lowest precedence. *)
      Hashtbl.add Parser.binop_precedence '=' 2;
      Hashtbl.add Parser.binop_precedence '<' 10;
      Hashtbl.add Parser.binop_precedence '+' 20;
      Hashtbl.add Parser.binop_precedence '-' 20;
      ...

Now that the parser knows the precedence of the binary operator, it
takes care of all the parsing and AST generation. We just need to
implement codegen for the assignment operator. This looks like:

.. code-block:: ocaml

    let rec codegen_expr = function
          begin match op with
          | '=' ->
              (* Special case '=' because we don't want to emit the LHS as an
               * expression. *)
              let name =
                match lhs with
                | Ast.Variable name -> name
                | _ -> raise (Error "destination of '=' must be a variable")
              in

Unlike the rest of the binary operators, our assignment operator doesn't
follow the "emit LHS, emit RHS, do computation" model. As such, it is
handled as a special case before the other binary operators are handled.
The other strange thing is that it requires the LHS to be a variable. It
is invalid to have "(x+1) = expr" - only things like "x = expr" are
allowed.

.. code-block:: ocaml

              (* Codegen the rhs. *)
              let val_ = codegen_expr rhs in

              (* Lookup the name. *)
              let variable = try Hashtbl.find named_values name with
              | Not_found -> raise (Error "unknown variable name")
              in
              ignore(build_store val_ variable builder);
              val_
          | _ ->
                ...

Once we have the variable, codegen'ing the assignment is
straightforward: we emit the RHS of the assignment, create a store, and
return the computed value. Returning a value allows for chained
assignments like "X = (Y = Z)".

Now that we have an assignment operator, we can mutate loop variables
and arguments. For example, we can now run code like this:

::

    # Function to print a double.
    extern printd(x);

    # Define ':' for sequencing: as a low-precedence operator that ignores operands
    # and just returns the RHS.
    def binary : 1 (x y) y;

    def test(x)
      printd(x) :
      x = 4 :
      printd(x);

    test(123);

When run, this example prints "123" and then "4", showing that we did
actually mutate the value! Okay, we have now officially implemented our
goal: getting this to work requires SSA construction in the general
case. However, to be really useful, we want the ability to define our
own local variables, lets add this next!

User-defined Local Variables
============================

Adding var/in is just like any other other extensions we made to
Kaleidoscope: we extend the lexer, the parser, the AST and the code
generator. The first step for adding our new 'var/in' construct is to
extend the lexer. As before, this is pretty trivial, the code looks like
this:

.. code-block:: ocaml

    type token =
      ...
      (* var definition *)
      | Var

    ...

    and lex_ident buffer = parser
          ...
          | "in" -> [< 'Token.In; stream >]
          | "binary" -> [< 'Token.Binary; stream >]
          | "unary" -> [< 'Token.Unary; stream >]
          | "var" -> [< 'Token.Var; stream >]
          ...

The next step is to define the AST node that we will construct. For
var/in, it looks like this:

.. code-block:: ocaml

    type expr =
      ...
      (* variant for var/in. *)
      | Var of (string * expr option) array * expr
      ...

var/in allows a list of names to be defined all at once, and each name
can optionally have an initializer value. As such, we capture this
information in the VarNames vector. Also, var/in has a body, this body
is allowed to access the variables defined by the var/in.

With this in place, we can define the parser pieces. The first thing we
do is add it as a primary expression:

.. code-block:: ocaml

    (* primary
     *   ::= identifier
     *   ::= numberexpr
     *   ::= parenexpr
     *   ::= ifexpr
     *   ::= forexpr
     *   ::= varexpr *)
    let rec parse_primary = parser
      ...
      (* varexpr
       *   ::= 'var' identifier ('=' expression?
       *             (',' identifier ('=' expression)?)* 'in' expression *)
      | [< 'Token.Var;
           (* At least one variable name is required. *)
           'Token.Ident id ?? "expected identifier after var";
           init=parse_var_init;
           var_names=parse_var_names [(id, init)];
           (* At this point, we have to have 'in'. *)
           'Token.In ?? "expected 'in' keyword after 'var'";
           body=parse_expr >] ->
          Ast.Var (Array.of_list (List.rev var_names), body)

    ...

    and parse_var_init = parser
      (* read in the optional initializer. *)
      | [< 'Token.Kwd '='; e=parse_expr >] -> Some e
      | [< >] -> None

    and parse_var_names accumulator = parser
      | [< 'Token.Kwd ',';
           'Token.Ident id ?? "expected identifier list after var";
           init=parse_var_init;
           e=parse_var_names ((id, init) :: accumulator) >] -> e
      | [< >] -> accumulator

Now that we can parse and represent the code, we need to support
emission of LLVM IR for it. This code starts out with:

.. code-block:: ocaml

    let rec codegen_expr = function
      ...
      | Ast.Var (var_names, body)
          let old_bindings = ref [] in

          let the_function = block_parent (insertion_block builder) in

          (* Register all variables and emit their initializer. *)
          Array.iter (fun (var_name, init) ->

Basically it loops over all the variables, installing them one at a
time. For each variable we put into the symbol table, we remember the
previous value that we replace in OldBindings.

.. code-block:: ocaml

            (* Emit the initializer before adding the variable to scope, this
             * prevents the initializer from referencing the variable itself, and
             * permits stuff like this:
             *   var a = 1 in
             *     var a = a in ...   # refers to outer 'a'. *)
            let init_val =
              match init with
              | Some init -> codegen_expr init
              (* If not specified, use 0.0. *)
              | None -> const_float double_type 0.0
            in

            let alloca = create_entry_block_alloca the_function var_name in
            ignore(build_store init_val alloca builder);

            (* Remember the old variable binding so that we can restore the binding
             * when we unrecurse. *)

            begin
              try
                let old_value = Hashtbl.find named_values var_name in
                old_bindings := (var_name, old_value) :: !old_bindings;
              with Not_found > ()
            end;

            (* Remember this binding. *)
            Hashtbl.add named_values var_name alloca;
          ) var_names;

There are more comments here than code. The basic idea is that we emit
the initializer, create the alloca, then update the symbol table to
point to it. Once all the variables are installed in the symbol table,
we evaluate the body of the var/in expression:

.. code-block:: ocaml

          (* Codegen the body, now that all vars are in scope. *)
          let body_val = codegen_expr body in

Finally, before returning, we restore the previous variable bindings:

.. code-block:: ocaml

          (* Pop all our variables from scope. *)
          List.iter (fun (var_name, old_value) ->
            Hashtbl.add named_values var_name old_value
          ) !old_bindings;

          (* Return the body computation. *)
          body_val

The end result of all of this is that we get properly scoped variable
definitions, and we even (trivially) allow mutation of them :).

With this, we completed what we set out to do. Our nice iterative fib
example from the intro compiles and runs just fine. The mem2reg pass
optimizes all of our stack variables into SSA registers, inserting PHI
nodes where needed, and our front-end remains simple: no "iterated
dominance frontier" computation anywhere in sight.

Full Code Listing
=================

Here is the complete code listing for our running example, enhanced with
mutable variables and var/in support. To build this example, use:

.. code-block:: bash

    # Compile
    ocamlbuild toy.byte
    # Run
    ./toy.byte

Here is the code:

\_tags:
    ::

        <{lexer,parser}.ml>: use_camlp4, pp(camlp4of)
        <*.{byte,native}>: g++, use_llvm, use_llvm_analysis
        <*.{byte,native}>: use_llvm_executionengine, use_llvm_target
        <*.{byte,native}>: use_llvm_scalar_opts, use_bindings

myocamlbuild.ml:
    .. code-block:: ocaml

        open Ocamlbuild_plugin;;

        ocaml_lib ~extern:true "llvm";;
        ocaml_lib ~extern:true "llvm_analysis";;
        ocaml_lib ~extern:true "llvm_executionengine";;
        ocaml_lib ~extern:true "llvm_target";;
        ocaml_lib ~extern:true "llvm_scalar_opts";;

        flag ["link"; "ocaml"; "g++"] (S[A"-cc"; A"g++"; A"-cclib"; A"-rdynamic"]);;
        dep ["link"; "ocaml"; "use_bindings"] ["bindings.o"];;

token.ml:
    .. code-block:: ocaml

        (*===----------------------------------------------------------------------===
         * Lexer Tokens
         *===----------------------------------------------------------------------===*)

        (* The lexer returns these 'Kwd' if it is an unknown character, otherwise one of
         * these others for known things. *)
        type token =
          (* commands *)
          | Def | Extern

          (* primary *)
          | Ident of string | Number of float

          (* unknown *)
          | Kwd of char

          (* control *)
          | If | Then | Else
          | For | In

          (* operators *)
          | Binary | Unary

          (* var definition *)
          | Var

lexer.ml:
    .. code-block:: ocaml

        (*===----------------------------------------------------------------------===
         * Lexer
         *===----------------------------------------------------------------------===*)

        let rec lex = parser
          (* Skip any whitespace. *)
          | [< ' (' ' | '\n' | '\r' | '\t'); stream >] -> lex stream

          (* identifier: [a-zA-Z][a-zA-Z0-9] *)
          | [< ' ('A' .. 'Z' | 'a' .. 'z' as c); stream >] ->
              let buffer = Buffer.create 1 in
              Buffer.add_char buffer c;
              lex_ident buffer stream

          (* number: [0-9.]+ *)
          | [< ' ('0' .. '9' as c); stream >] ->
              let buffer = Buffer.create 1 in
              Buffer.add_char buffer c;
              lex_number buffer stream

          (* Comment until end of line. *)
          | [< ' ('#'); stream >] ->
              lex_comment stream

          (* Otherwise, just return the character as its ascii value. *)
          | [< 'c; stream >] ->
              [< 'Token.Kwd c; lex stream >]

          (* end of stream. *)
          | [< >] -> [< >]

        and lex_number buffer = parser
          | [< ' ('0' .. '9' | '.' as c); stream >] ->
              Buffer.add_char buffer c;
              lex_number buffer stream
          | [< stream=lex >] ->
              [< 'Token.Number (float_of_string (Buffer.contents buffer)); stream >]

        and lex_ident buffer = parser
          | [< ' ('A' .. 'Z' | 'a' .. 'z' | '0' .. '9' as c); stream >] ->
              Buffer.add_char buffer c;
              lex_ident buffer stream
          | [< stream=lex >] ->
              match Buffer.contents buffer with
              | "def" -> [< 'Token.Def; stream >]
              | "extern" -> [< 'Token.Extern; stream >]
              | "if" -> [< 'Token.If; stream >]
              | "then" -> [< 'Token.Then; stream >]
              | "else" -> [< 'Token.Else; stream >]
              | "for" -> [< 'Token.For; stream >]
              | "in" -> [< 'Token.In; stream >]
              | "binary" -> [< 'Token.Binary; stream >]
              | "unary" -> [< 'Token.Unary; stream >]
              | "var" -> [< 'Token.Var; stream >]
              | id -> [< 'Token.Ident id; stream >]

        and lex_comment = parser
          | [< ' ('\n'); stream=lex >] -> stream
          | [< 'c; e=lex_comment >] -> e
          | [< >] -> [< >]

ast.ml:
    .. code-block:: ocaml

        (*===----------------------------------------------------------------------===
         * Abstract Syntax Tree (aka Parse Tree)
         *===----------------------------------------------------------------------===*)

        (* expr - Base type for all expression nodes. *)
        type expr =
          (* variant for numeric literals like "1.0". *)
          | Number of float

          (* variant for referencing a variable, like "a". *)
          | Variable of string

          (* variant for a unary operator. *)
          | Unary of char * expr

          (* variant for a binary operator. *)
          | Binary of char * expr * expr

          (* variant for function calls. *)
          | Call of string * expr array

          (* variant for if/then/else. *)
          | If of expr * expr * expr

          (* variant for for/in. *)
          | For of string * expr * expr * expr option * expr

          (* variant for var/in. *)
          | Var of (string * expr option) array * expr

        (* proto - This type represents the "prototype" for a function, which captures
         * its name, and its argument names (thus implicitly the number of arguments the
         * function takes). *)
        type proto =
          | Prototype of string * string array
          | BinOpPrototype of string * string array * int

        (* func - This type represents a function definition itself. *)
        type func = Function of proto * expr

parser.ml:
    .. code-block:: ocaml

        (*===---------------------------------------------------------------------===
         * Parser
         *===---------------------------------------------------------------------===*)

        (* binop_precedence - This holds the precedence for each binary operator that is
         * defined *)
        let binop_precedence:(char, int) Hashtbl.t = Hashtbl.create 10

        (* precedence - Get the precedence of the pending binary operator token. *)
        let precedence c = try Hashtbl.find binop_precedence c with Not_found -> -1

        (* primary
         *   ::= identifier
         *   ::= numberexpr
         *   ::= parenexpr
         *   ::= ifexpr
         *   ::= forexpr
         *   ::= varexpr *)
        let rec parse_primary = parser
          (* numberexpr ::= number *)
          | [< 'Token.Number n >] -> Ast.Number n

          (* parenexpr ::= '(' expression ')' *)
          | [< 'Token.Kwd '('; e=parse_expr; 'Token.Kwd ')' ?? "expected ')'" >] -> e

          (* identifierexpr
           *   ::= identifier
           *   ::= identifier '(' argumentexpr ')' *)
          | [< 'Token.Ident id; stream >] ->
              let rec parse_args accumulator = parser
                | [< e=parse_expr; stream >] ->
                    begin parser
                      | [< 'Token.Kwd ','; e=parse_args (e :: accumulator) >] -> e
                      | [< >] -> e :: accumulator
                    end stream
                | [< >] -> accumulator
              in
              let rec parse_ident id = parser
                (* Call. *)
                | [< 'Token.Kwd '(';
                     args=parse_args [];
                     'Token.Kwd ')' ?? "expected ')'">] ->
                    Ast.Call (id, Array.of_list (List.rev args))

                (* Simple variable ref. *)
                | [< >] -> Ast.Variable id
              in
              parse_ident id stream

          (* ifexpr ::= 'if' expr 'then' expr 'else' expr *)
          | [< 'Token.If; c=parse_expr;
               'Token.Then ?? "expected 'then'"; t=parse_expr;
               'Token.Else ?? "expected 'else'"; e=parse_expr >] ->
              Ast.If (c, t, e)

          (* forexpr
                ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression *)
          | [< 'Token.For;
               'Token.Ident id ?? "expected identifier after for";
               'Token.Kwd '=' ?? "expected '=' after for";
               stream >] ->
              begin parser
                | [<
                     start=parse_expr;
                     'Token.Kwd ',' ?? "expected ',' after for";
                     end_=parse_expr;
                     stream >] ->
                    let step =
                      begin parser
                      | [< 'Token.Kwd ','; step=parse_expr >] -> Some step
                      | [< >] -> None
                      end stream
                    in
                    begin parser
                    | [< 'Token.In; body=parse_expr >] ->
                        Ast.For (id, start, end_, step, body)
                    | [< >] ->
                        raise (Stream.Error "expected 'in' after for")
                    end stream
                | [< >] ->
                    raise (Stream.Error "expected '=' after for")
              end stream

          (* varexpr
           *   ::= 'var' identifier ('=' expression?
           *             (',' identifier ('=' expression)?)* 'in' expression *)
          | [< 'Token.Var;
               (* At least one variable name is required. *)
               'Token.Ident id ?? "expected identifier after var";
               init=parse_var_init;
               var_names=parse_var_names [(id, init)];
               (* At this point, we have to have 'in'. *)
               'Token.In ?? "expected 'in' keyword after 'var'";
               body=parse_expr >] ->
              Ast.Var (Array.of_list (List.rev var_names), body)

          | [< >] -> raise (Stream.Error "unknown token when expecting an expression.")

        (* unary
         *   ::= primary
         *   ::= '!' unary *)
        and parse_unary = parser
          (* If this is a unary operator, read it. *)
          | [< 'Token.Kwd op when op != '(' && op != ')'; operand=parse_expr >] ->
              Ast.Unary (op, operand)

          (* If the current token is not an operator, it must be a primary expr. *)
          | [< stream >] -> parse_primary stream

        (* binoprhs
         *   ::= ('+' primary)* *)
        and parse_bin_rhs expr_prec lhs stream =
          match Stream.peek stream with
          (* If this is a binop, find its precedence. *)
          | Some (Token.Kwd c) when Hashtbl.mem binop_precedence c ->
              let token_prec = precedence c in

              (* If this is a binop that binds at least as tightly as the current binop,
               * consume it, otherwise we are done. *)
              if token_prec < expr_prec then lhs else begin
                (* Eat the binop. *)
                Stream.junk stream;

                (* Parse the primary expression after the binary operator. *)
                let rhs = parse_unary stream in

                (* Okay, we know this is a binop. *)
                let rhs =
                  match Stream.peek stream with
                  | Some (Token.Kwd c2) ->
                      (* If BinOp binds less tightly with rhs than the operator after
                       * rhs, let the pending operator take rhs as its lhs. *)
                      let next_prec = precedence c2 in
                      if token_prec < next_prec
                      then parse_bin_rhs (token_prec + 1) rhs stream
                      else rhs
                  | _ -> rhs
                in

                (* Merge lhs/rhs. *)
                let lhs = Ast.Binary (c, lhs, rhs) in
                parse_bin_rhs expr_prec lhs stream
              end
          | _ -> lhs

        and parse_var_init = parser
          (* read in the optional initializer. *)
          | [< 'Token.Kwd '='; e=parse_expr >] -> Some e
          | [< >] -> None

        and parse_var_names accumulator = parser
          | [< 'Token.Kwd ',';
               'Token.Ident id ?? "expected identifier list after var";
               init=parse_var_init;
               e=parse_var_names ((id, init) :: accumulator) >] -> e
          | [< >] -> accumulator

        (* expression
         *   ::= primary binoprhs *)
        and parse_expr = parser
          | [< lhs=parse_unary; stream >] -> parse_bin_rhs 0 lhs stream

        (* prototype
         *   ::= id '(' id* ')'
         *   ::= binary LETTER number? (id, id)
         *   ::= unary LETTER number? (id) *)
        let parse_prototype =
          let rec parse_args accumulator = parser
            | [< 'Token.Ident id; e=parse_args (id::accumulator) >] -> e
            | [< >] -> accumulator
          in
          let parse_operator = parser
            | [< 'Token.Unary >] -> "unary", 1
            | [< 'Token.Binary >] -> "binary", 2
          in
          let parse_binary_precedence = parser
            | [< 'Token.Number n >] -> int_of_float n
            | [< >] -> 30
          in
          parser
          | [< 'Token.Ident id;
               'Token.Kwd '(' ?? "expected '(' in prototype";
               args=parse_args [];
               'Token.Kwd ')' ?? "expected ')' in prototype" >] ->
              (* success. *)
              Ast.Prototype (id, Array.of_list (List.rev args))
          | [< (prefix, kind)=parse_operator;
               'Token.Kwd op ?? "expected an operator";
               (* Read the precedence if present. *)
               binary_precedence=parse_binary_precedence;
               'Token.Kwd '(' ?? "expected '(' in prototype";
                args=parse_args [];
               'Token.Kwd ')' ?? "expected ')' in prototype" >] ->
              let name = prefix ^ (String.make 1 op) in
              let args = Array.of_list (List.rev args) in

              (* Verify right number of arguments for operator. *)
              if Array.length args != kind
              then raise (Stream.Error "invalid number of operands for operator")
              else
                if kind == 1 then
                  Ast.Prototype (name, args)
                else
                  Ast.BinOpPrototype (name, args, binary_precedence)
          | [< >] ->
              raise (Stream.Error "expected function name in prototype")

        (* definition ::= 'def' prototype expression *)
        let parse_definition = parser
          | [< 'Token.Def; p=parse_prototype; e=parse_expr >] ->
              Ast.Function (p, e)

        (* toplevelexpr ::= expression *)
        let parse_toplevel = parser
          | [< e=parse_expr >] ->
              (* Make an anonymous proto. *)
              Ast.Function (Ast.Prototype ("", [||]), e)

        (*  external ::= 'extern' prototype *)
        let parse_extern = parser
          | [< 'Token.Extern; e=parse_prototype >] -> e

codegen.ml:
    .. code-block:: ocaml

        (*===----------------------------------------------------------------------===
         * Code Generation
         *===----------------------------------------------------------------------===*)

        open Llvm

        exception Error of string

        let context = global_context ()
        let the_module = create_module context "my cool jit"
        let builder = builder context
        let named_values:(string, llvalue) Hashtbl.t = Hashtbl.create 10
        let double_type = double_type context

        (* Create an alloca instruction in the entry block of the function. This
         * is used for mutable variables etc. *)
        let create_entry_block_alloca the_function var_name =
          let builder = builder_at context (instr_begin (entry_block the_function)) in
          build_alloca double_type var_name builder

        let rec codegen_expr = function
          | Ast.Number n -> const_float double_type n
          | Ast.Variable name ->
              let v = try Hashtbl.find named_values name with
                | Not_found -> raise (Error "unknown variable name")
              in
              (* Load the value. *)
              build_load v name builder
          | Ast.Unary (op, operand) ->
              let operand = codegen_expr operand in
              let callee = "unary" ^ (String.make 1 op) in
              let callee =
                match lookup_function callee the_module with
                | Some callee -> callee
                | None -> raise (Error "unknown unary operator")
              in
              build_call callee [|operand|] "unop" builder
          | Ast.Binary (op, lhs, rhs) ->
              begin match op with
              | '=' ->
                  (* Special case '=' because we don't want to emit the LHS as an
                   * expression. *)
                  let name =
                    match lhs with
                    | Ast.Variable name -> name
                    | _ -> raise (Error "destination of '=' must be a variable")
                  in

                  (* Codegen the rhs. *)
                  let val_ = codegen_expr rhs in

                  (* Lookup the name. *)
                  let variable = try Hashtbl.find named_values name with
                  | Not_found -> raise (Error "unknown variable name")
                  in
                  ignore(build_store val_ variable builder);
                  val_
              | _ ->
                  let lhs_val = codegen_expr lhs in
                  let rhs_val = codegen_expr rhs in
                  begin
                    match op with
                    | '+' -> build_add lhs_val rhs_val "addtmp" builder
                    | '-' -> build_sub lhs_val rhs_val "subtmp" builder
                    | '*' -> build_mul lhs_val rhs_val "multmp" builder
                    | '<' ->
                        (* Convert bool 0/1 to double 0.0 or 1.0 *)
                        let i = build_fcmp Fcmp.Ult lhs_val rhs_val "cmptmp" builder in
                        build_uitofp i double_type "booltmp" builder
                    | _ ->
                        (* If it wasn't a builtin binary operator, it must be a user defined
                         * one. Emit a call to it. *)
                        let callee = "binary" ^ (String.make 1 op) in
                        let callee =
                          match lookup_function callee the_module with
                          | Some callee -> callee
                          | None -> raise (Error "binary operator not found!")
                        in
                        build_call callee [|lhs_val; rhs_val|] "binop" builder
                  end
              end
          | Ast.Call (callee, args) ->
              (* Look up the name in the module table. *)
              let callee =
                match lookup_function callee the_module with
                | Some callee -> callee
                | None -> raise (Error "unknown function referenced")
              in
              let params = params callee in

              (* If argument mismatch error. *)
              if Array.length params == Array.length args then () else
                raise (Error "incorrect # arguments passed");
              let args = Array.map codegen_expr args in
              build_call callee args "calltmp" builder
          | Ast.If (cond, then_, else_) ->
              let cond = codegen_expr cond in

              (* Convert condition to a bool by comparing equal to 0.0 *)
              let zero = const_float double_type 0.0 in
              let cond_val = build_fcmp Fcmp.One cond zero "ifcond" builder in

              (* Grab the first block so that we might later add the conditional branch
               * to it at the end of the function. *)
              let start_bb = insertion_block builder in
              let the_function = block_parent start_bb in

              let then_bb = append_block context "then" the_function in

              (* Emit 'then' value. *)
              position_at_end then_bb builder;
              let then_val = codegen_expr then_ in

              (* Codegen of 'then' can change the current block, update then_bb for the
               * phi. We create a new name because one is used for the phi node, and the
               * other is used for the conditional branch. *)
              let new_then_bb = insertion_block builder in

              (* Emit 'else' value. *)
              let else_bb = append_block context "else" the_function in
              position_at_end else_bb builder;
              let else_val = codegen_expr else_ in

              (* Codegen of 'else' can change the current block, update else_bb for the
               * phi. *)
              let new_else_bb = insertion_block builder in

              (* Emit merge block. *)
              let merge_bb = append_block context "ifcont" the_function in
              position_at_end merge_bb builder;
              let incoming = [(then_val, new_then_bb); (else_val, new_else_bb)] in
              let phi = build_phi incoming "iftmp" builder in

              (* Return to the start block to add the conditional branch. *)
              position_at_end start_bb builder;
              ignore (build_cond_br cond_val then_bb else_bb builder);

              (* Set a unconditional branch at the end of the 'then' block and the
               * 'else' block to the 'merge' block. *)
              position_at_end new_then_bb builder; ignore (build_br merge_bb builder);
              position_at_end new_else_bb builder; ignore (build_br merge_bb builder);

              (* Finally, set the builder to the end of the merge block. *)
              position_at_end merge_bb builder;

              phi
          | Ast.For (var_name, start, end_, step, body) ->
              (* Output this as:
               *   var = alloca double
               *   ...
               *   start = startexpr
               *   store start -> var
               *   goto loop
               * loop:
               *   ...
               *   bodyexpr
               *   ...
               * loopend:
               *   step = stepexpr
               *   endcond = endexpr
               *
               *   curvar = load var
               *   nextvar = curvar + step
               *   store nextvar -> var
               *   br endcond, loop, endloop
               * outloop: *)

              let the_function = block_parent (insertion_block builder) in

              (* Create an alloca for the variable in the entry block. *)
              let alloca = create_entry_block_alloca the_function var_name in

              (* Emit the start code first, without 'variable' in scope. *)
              let start_val = codegen_expr start in

              (* Store the value into the alloca. *)
              ignore(build_store start_val alloca builder);

              (* Make the new basic block for the loop header, inserting after current
               * block. *)
              let loop_bb = append_block context "loop" the_function in

              (* Insert an explicit fall through from the current block to the
               * loop_bb. *)
              ignore (build_br loop_bb builder);

              (* Start insertion in loop_bb. *)
              position_at_end loop_bb builder;

              (* Within the loop, the variable is defined equal to the PHI node. If it
               * shadows an existing variable, we have to restore it, so save it
               * now. *)
              let old_val =
                try Some (Hashtbl.find named_values var_name) with Not_found -> None
              in
              Hashtbl.add named_values var_name alloca;

              (* Emit the body of the loop.  This, like any other expr, can change the
               * current BB.  Note that we ignore the value computed by the body, but
               * don't allow an error *)
              ignore (codegen_expr body);

              (* Emit the step value. *)
              let step_val =
                match step with
                | Some step -> codegen_expr step
                (* If not specified, use 1.0. *)
                | None -> const_float double_type 1.0
              in

              (* Compute the end condition. *)
              let end_cond = codegen_expr end_ in

              (* Reload, increment, and restore the alloca. This handles the case where
               * the body of the loop mutates the variable. *)
              let cur_var = build_load alloca var_name builder in
              let next_var = build_add cur_var step_val "nextvar" builder in
              ignore(build_store next_var alloca builder);

              (* Convert condition to a bool by comparing equal to 0.0. *)
              let zero = const_float double_type 0.0 in
              let end_cond = build_fcmp Fcmp.One end_cond zero "loopcond" builder in

              (* Create the "after loop" block and insert it. *)
              let after_bb = append_block context "afterloop" the_function in

              (* Insert the conditional branch into the end of loop_end_bb. *)
              ignore (build_cond_br end_cond loop_bb after_bb builder);

              (* Any new code will be inserted in after_bb. *)
              position_at_end after_bb builder;

              (* Restore the unshadowed variable. *)
              begin match old_val with
              | Some old_val -> Hashtbl.add named_values var_name old_val
              | None -> ()
              end;

              (* for expr always returns 0.0. *)
              const_null double_type
          | Ast.Var (var_names, body) ->
              let old_bindings = ref [] in

              let the_function = block_parent (insertion_block builder) in

              (* Register all variables and emit their initializer. *)
              Array.iter (fun (var_name, init) ->
                (* Emit the initializer before adding the variable to scope, this
                 * prevents the initializer from referencing the variable itself, and
                 * permits stuff like this:
                 *   var a = 1 in
                 *     var a = a in ...   # refers to outer 'a'. *)
                let init_val =
                  match init with
                  | Some init -> codegen_expr init
                  (* If not specified, use 0.0. *)
                  | None -> const_float double_type 0.0
                in

                let alloca = create_entry_block_alloca the_function var_name in
                ignore(build_store init_val alloca builder);

                (* Remember the old variable binding so that we can restore the binding
                 * when we unrecurse. *)
                begin
                  try
                    let old_value = Hashtbl.find named_values var_name in
                    old_bindings := (var_name, old_value) :: !old_bindings;
                  with Not_found -> ()
                end;

                (* Remember this binding. *)
                Hashtbl.add named_values var_name alloca;
              ) var_names;

              (* Codegen the body, now that all vars are in scope. *)
              let body_val = codegen_expr body in

              (* Pop all our variables from scope. *)
              List.iter (fun (var_name, old_value) ->
                Hashtbl.add named_values var_name old_value
              ) !old_bindings;

              (* Return the body computation. *)
              body_val

        let codegen_proto = function
          | Ast.Prototype (name, args) | Ast.BinOpPrototype (name, args, _) ->
              (* Make the function type: double(double,double) etc. *)
              let doubles = Array.make (Array.length args) double_type in
              let ft = function_type double_type doubles in
              let f =
                match lookup_function name the_module with
                | None -> declare_function name ft the_module

                (* If 'f' conflicted, there was already something named 'name'. If it
                 * has a body, don't allow redefinition or reextern. *)
                | Some f ->
                    (* If 'f' already has a body, reject this. *)
                    if block_begin f <> At_end f then
                      raise (Error "redefinition of function");

                    (* If 'f' took a different number of arguments, reject. *)
                    if element_type (type_of f) <> ft then
                      raise (Error "redefinition of function with different # args");
                    f
              in

              (* Set names for all arguments. *)
              Array.iteri (fun i a ->
                let n = args.(i) in
                set_value_name n a;
                Hashtbl.add named_values n a;
              ) (params f);
              f

        (* Create an alloca for each argument and register the argument in the symbol
         * table so that references to it will succeed. *)
        let create_argument_allocas the_function proto =
          let args = match proto with
            | Ast.Prototype (_, args) | Ast.BinOpPrototype (_, args, _) -> args
          in
          Array.iteri (fun i ai ->
            let var_name = args.(i) in
            (* Create an alloca for this variable. *)
            let alloca = create_entry_block_alloca the_function var_name in

            (* Store the initial value into the alloca. *)
            ignore(build_store ai alloca builder);

            (* Add arguments to variable symbol table. *)
            Hashtbl.add named_values var_name alloca;
          ) (params the_function)

        let codegen_func the_fpm = function
          | Ast.Function (proto, body) ->
              Hashtbl.clear named_values;
              let the_function = codegen_proto proto in

              (* If this is an operator, install it. *)
              begin match proto with
              | Ast.BinOpPrototype (name, args, prec) ->
                  let op = name.[String.length name - 1] in
                  Hashtbl.add Parser.binop_precedence op prec;
              | _ -> ()
              end;

              (* Create a new basic block to start insertion into. *)
              let bb = append_block context "entry" the_function in
              position_at_end bb builder;

              try
                (* Add all arguments to the symbol table and create their allocas. *)
                create_argument_allocas the_function proto;

                let ret_val = codegen_expr body in

                (* Finish off the function. *)
                let _ = build_ret ret_val builder in

                (* Validate the generated code, checking for consistency. *)
                Llvm_analysis.assert_valid_function the_function;

                (* Optimize the function. *)
                let _ = PassManager.run_function the_function the_fpm in

                the_function
              with e ->
                delete_function the_function;
                raise e

toplevel.ml:
    .. code-block:: ocaml

        (*===----------------------------------------------------------------------===
         * Top-Level parsing and JIT Driver
         *===----------------------------------------------------------------------===*)

        open Llvm
        open Llvm_executionengine

        (* top ::= definition | external | expression | ';' *)
        let rec main_loop the_fpm the_execution_engine stream =
          match Stream.peek stream with
          | None -> ()

          (* ignore top-level semicolons. *)
          | Some (Token.Kwd ';') ->
              Stream.junk stream;
              main_loop the_fpm the_execution_engine stream

          | Some token ->
              begin
                try match token with
                | Token.Def ->
                    let e = Parser.parse_definition stream in
                    print_endline "parsed a function definition.";
                    dump_value (Codegen.codegen_func the_fpm e);
                | Token.Extern ->
                    let e = Parser.parse_extern stream in
                    print_endline "parsed an extern.";
                    dump_value (Codegen.codegen_proto e);
                | _ ->
                    (* Evaluate a top-level expression into an anonymous function. *)
                    let e = Parser.parse_toplevel stream in
                    print_endline "parsed a top-level expr";
                    let the_function = Codegen.codegen_func the_fpm e in
                    dump_value the_function;

                    (* JIT the function, returning a function pointer. *)
                    let result = ExecutionEngine.run_function the_function [||]
                      the_execution_engine in

                    print_string "Evaluated to ";
                    print_float (GenericValue.as_float Codegen.double_type result);
                    print_newline ();
                with Stream.Error s | Codegen.Error s ->
                  (* Skip token for error recovery. *)
                  Stream.junk stream;
                  print_endline s;
              end;
              print_string "ready> "; flush stdout;
              main_loop the_fpm the_execution_engine stream

toy.ml:
    .. code-block:: ocaml

        (*===----------------------------------------------------------------------===
         * Main driver code.
         *===----------------------------------------------------------------------===*)

        open Llvm
        open Llvm_executionengine
        open Llvm_target
        open Llvm_scalar_opts

        let main () =
          ignore (initialize_native_target ());

          (* Install standard binary operators.
           * 1 is the lowest precedence. *)
          Hashtbl.add Parser.binop_precedence '=' 2;
          Hashtbl.add Parser.binop_precedence '<' 10;
          Hashtbl.add Parser.binop_precedence '+' 20;
          Hashtbl.add Parser.binop_precedence '-' 20;
          Hashtbl.add Parser.binop_precedence '*' 40;    (* highest. *)

          (* Prime the first token. *)
          print_string "ready> "; flush stdout;
          let stream = Lexer.lex (Stream.of_channel stdin) in

          (* Create the JIT. *)
          let the_execution_engine = ExecutionEngine.create Codegen.the_module in
          let the_fpm = PassManager.create_function Codegen.the_module in

          (* Set up the optimizer pipeline.  Start with registering info about how the
           * target lays out data structures. *)
          DataLayout.add (ExecutionEngine.target_data the_execution_engine) the_fpm;

          (* Promote allocas to registers. *)
          add_memory_to_register_promotion the_fpm;

          (* Do simple "peephole" optimizations and bit-twiddling optzn. *)
          add_instruction_combination the_fpm;

          (* reassociate expressions. *)
          add_reassociation the_fpm;

          (* Eliminate Common SubExpressions. *)
          add_gvn the_fpm;

          (* Simplify the control flow graph (deleting unreachable blocks, etc). *)
          add_cfg_simplification the_fpm;

          ignore (PassManager.initialize the_fpm);

          (* Run the main "interpreter loop" now. *)
          Toplevel.main_loop the_fpm the_execution_engine stream;

          (* Print out all the generated code. *)
          dump_module Codegen.the_module
        ;;

        main ()

bindings.c
    .. code-block:: c

        #include <stdio.h>

        /* putchard - putchar that takes a double and returns 0. */
        extern double putchard(double X) {
          putchar((char)X);
          return 0;
        }

        /* printd - printf that takes a double prints it as "%f\n", returning 0. */
        extern double printd(double X) {
          printf("%f\n", X);
          return 0;
        }

`Next: Conclusion and other useful LLVM tidbits <OCamlLangImpl8.html>`_

