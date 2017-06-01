==================================================
Kaleidoscope: Extending the Language: Control Flow
==================================================

.. contents::
   :local:

Chapter 5 Introduction
======================

Welcome to Chapter 5 of the "`Implementing a language with
LLVM <index.html>`_" tutorial. Parts 1-4 described the implementation of
the simple Kaleidoscope language and included support for generating
LLVM IR, followed by optimizations and a JIT compiler. Unfortunately, as
presented, Kaleidoscope is mostly useless: it has no control flow other
than call and return. This means that you can't have conditional
branches in the code, significantly limiting its power. In this episode
of "build that compiler", we'll extend Kaleidoscope to have an
if/then/else expression plus a simple 'for' loop.

If/Then/Else
============

Extending Kaleidoscope to support if/then/else is quite straightforward.
It basically requires adding lexer support for this "new" concept to the
lexer, parser, AST, and LLVM code emitter. This example is nice, because
it shows how easy it is to "grow" a language over time, incrementally
extending it as new ideas are discovered.

Before we get going on "how" we add this extension, lets talk about
"what" we want. The basic idea is that we want to be able to write this
sort of thing:

::

    def fib(x)
      if x < 3 then
        1
      else
        fib(x-1)+fib(x-2);

In Kaleidoscope, every construct is an expression: there are no
statements. As such, the if/then/else expression needs to return a value
like any other. Since we're using a mostly functional form, we'll have
it evaluate its conditional, then return the 'then' or 'else' value
based on how the condition was resolved. This is very similar to the C
"?:" expression.

The semantics of the if/then/else expression is that it evaluates the
condition to a boolean equality value: 0.0 is considered to be false and
everything else is considered to be true. If the condition is true, the
first subexpression is evaluated and returned, if the condition is
false, the second subexpression is evaluated and returned. Since
Kaleidoscope allows side-effects, this behavior is important to nail
down.

Now that we know what we "want", lets break this down into its
constituent pieces.

Lexer Extensions for If/Then/Else
---------------------------------

The lexer extensions are straightforward. First we add new variants for
the relevant tokens:

.. code-block:: ocaml

      (* control *)
      | If | Then | Else | For | In

Once we have that, we recognize the new keywords in the lexer. This is
pretty simple stuff:

.. code-block:: ocaml

          ...
          match Buffer.contents buffer with
          | "def" -> [< 'Token.Def; stream >]
          | "extern" -> [< 'Token.Extern; stream >]
          | "if" -> [< 'Token.If; stream >]
          | "then" -> [< 'Token.Then; stream >]
          | "else" -> [< 'Token.Else; stream >]
          | "for" -> [< 'Token.For; stream >]
          | "in" -> [< 'Token.In; stream >]
          | id -> [< 'Token.Ident id; stream >]

AST Extensions for If/Then/Else
-------------------------------

To represent the new expression we add a new AST variant for it:

.. code-block:: ocaml

    type expr =
      ...
      (* variant for if/then/else. *)
      | If of expr * expr * expr

The AST variant just has pointers to the various subexpressions.

Parser Extensions for If/Then/Else
----------------------------------

Now that we have the relevant tokens coming from the lexer and we have
the AST node to build, our parsing logic is relatively straightforward.
First we define a new parsing function:

.. code-block:: ocaml

    let rec parse_primary = parser
      ...
      (* ifexpr ::= 'if' expr 'then' expr 'else' expr *)
      | [< 'Token.If; c=parse_expr;
           'Token.Then ?? "expected 'then'"; t=parse_expr;
           'Token.Else ?? "expected 'else'"; e=parse_expr >] ->
          Ast.If (c, t, e)

Next we hook it up as a primary expression:

.. code-block:: ocaml

    let rec parse_primary = parser
      ...
      (* ifexpr ::= 'if' expr 'then' expr 'else' expr *)
      | [< 'Token.If; c=parse_expr;
           'Token.Then ?? "expected 'then'"; t=parse_expr;
           'Token.Else ?? "expected 'else'"; e=parse_expr >] ->
          Ast.If (c, t, e)

LLVM IR for If/Then/Else
------------------------

Now that we have it parsing and building the AST, the final piece is
adding LLVM code generation support. This is the most interesting part
of the if/then/else example, because this is where it starts to
introduce new concepts. All of the code above has been thoroughly
described in previous chapters.

To motivate the code we want to produce, lets take a look at a simple
example. Consider:

::

    extern foo();
    extern bar();
    def baz(x) if x then foo() else bar();

If you disable optimizations, the code you'll (soon) get from
Kaleidoscope looks like this:

.. code-block:: llvm

    declare double @foo()

    declare double @bar()

    define double @baz(double %x) {
    entry:
      %ifcond = fcmp one double %x, 0.000000e+00
      br i1 %ifcond, label %then, label %else

    then:    ; preds = %entry
      %calltmp = call double @foo()
      br label %ifcont

    else:    ; preds = %entry
      %calltmp1 = call double @bar()
      br label %ifcont

    ifcont:    ; preds = %else, %then
      %iftmp = phi double [ %calltmp, %then ], [ %calltmp1, %else ]
      ret double %iftmp
    }

To visualize the control flow graph, you can use a nifty feature of the
LLVM '`opt <http://llvm.org/cmds/opt.html>`_' tool. If you put this LLVM
IR into "t.ll" and run "``llvm-as < t.ll | opt -analyze -view-cfg``", `a
window will pop up <../ProgrammersManual.html#viewing-graphs-while-debugging-code>`_ and you'll
see this graph:

.. figure:: LangImpl05-cfg.png
   :align: center
   :alt: Example CFG

   Example CFG

Another way to get this is to call
"``Llvm_analysis.view_function_cfg f``" or
"``Llvm_analysis.view_function_cfg_only f``" (where ``f`` is a
"``Function``") either by inserting actual calls into the code and
recompiling or by calling these in the debugger. LLVM has many nice
features for visualizing various graphs.

Getting back to the generated code, it is fairly simple: the entry block
evaluates the conditional expression ("x" in our case here) and compares
the result to 0.0 with the "``fcmp one``" instruction ('one' is "Ordered
and Not Equal"). Based on the result of this expression, the code jumps
to either the "then" or "else" blocks, which contain the expressions for
the true/false cases.

Once the then/else blocks are finished executing, they both branch back
to the 'ifcont' block to execute the code that happens after the
if/then/else. In this case the only thing left to do is to return to the
caller of the function. The question then becomes: how does the code
know which expression to return?

The answer to this question involves an important SSA operation: the
`Phi
operation <http://en.wikipedia.org/wiki/Static_single_assignment_form>`_.
If you're not familiar with SSA, `the wikipedia
article <http://en.wikipedia.org/wiki/Static_single_assignment_form>`_
is a good introduction and there are various other introductions to it
available on your favorite search engine. The short version is that
"execution" of the Phi operation requires "remembering" which block
control came from. The Phi operation takes on the value corresponding to
the input control block. In this case, if control comes in from the
"then" block, it gets the value of "calltmp". If control comes from the
"else" block, it gets the value of "calltmp1".

At this point, you are probably starting to think "Oh no! This means my
simple and elegant front-end will have to start generating SSA form in
order to use LLVM!". Fortunately, this is not the case, and we strongly
advise *not* implementing an SSA construction algorithm in your
front-end unless there is an amazingly good reason to do so. In
practice, there are two sorts of values that float around in code
written for your average imperative programming language that might need
Phi nodes:

#. Code that involves user variables: ``x = 1; x = x + 1;``
#. Values that are implicit in the structure of your AST, such as the
   Phi node in this case.

In `Chapter 7 <OCamlLangImpl7.html>`_ of this tutorial ("mutable
variables"), we'll talk about #1 in depth. For now, just believe me that
you don't need SSA construction to handle this case. For #2, you have
the choice of using the techniques that we will describe for #1, or you
can insert Phi nodes directly, if convenient. In this case, it is really
really easy to generate the Phi node, so we choose to do it directly.

Okay, enough of the motivation and overview, lets generate code!

Code Generation for If/Then/Else
--------------------------------

In order to generate code for this, we implement the ``Codegen`` method
for ``IfExprAST``:

.. code-block:: ocaml

    let rec codegen_expr = function
      ...
      | Ast.If (cond, then_, else_) ->
          let cond = codegen_expr cond in

          (* Convert condition to a bool by comparing equal to 0.0 *)
          let zero = const_float double_type 0.0 in
          let cond_val = build_fcmp Fcmp.One cond zero "ifcond" builder in

This code is straightforward and similar to what we saw before. We emit
the expression for the condition, then compare that value to zero to get
a truth value as a 1-bit (bool) value.

.. code-block:: ocaml

          (* Grab the first block so that we might later add the conditional branch
           * to it at the end of the function. *)
          let start_bb = insertion_block builder in
          let the_function = block_parent start_bb in

          let then_bb = append_block context "then" the_function in
          position_at_end then_bb builder;

As opposed to the `C++ tutorial <LangImpl5.html>`_, we have to build our
basic blocks bottom up since we can't have dangling BasicBlocks. We
start off by saving a pointer to the first block (which might not be the
entry block), which we'll need to build a conditional branch later. We
do this by asking the ``builder`` for the current BasicBlock. The fourth
line gets the current Function object that is being built. It gets this
by the ``start_bb`` for its "parent" (the function it is currently
embedded into).

Once it has that, it creates one block. It is automatically appended
into the function's list of blocks.

.. code-block:: ocaml

          (* Emit 'then' value. *)
          position_at_end then_bb builder;
          let then_val = codegen_expr then_ in

          (* Codegen of 'then' can change the current block, update then_bb for the
           * phi. We create a new name because one is used for the phi node, and the
           * other is used for the conditional branch. *)
          let new_then_bb = insertion_block builder in

We move the builder to start inserting into the "then" block. Strictly
speaking, this call moves the insertion point to be at the end of the
specified block. However, since the "then" block is empty, it also
starts out by inserting at the beginning of the block. :)

Once the insertion point is set, we recursively codegen the "then"
expression from the AST.

The final line here is quite subtle, but is very important. The basic
issue is that when we create the Phi node in the merge block, we need to
set up the block/value pairs that indicate how the Phi will work.
Importantly, the Phi node expects to have an entry for each predecessor
of the block in the CFG. Why then, are we getting the current block when
we just set it to ThenBB 5 lines above? The problem is that the "Then"
expression may actually itself change the block that the Builder is
emitting into if, for example, it contains a nested "if/then/else"
expression. Because calling Codegen recursively could arbitrarily change
the notion of the current block, we are required to get an up-to-date
value for code that will set up the Phi node.

.. code-block:: ocaml

          (* Emit 'else' value. *)
          let else_bb = append_block context "else" the_function in
          position_at_end else_bb builder;
          let else_val = codegen_expr else_ in

          (* Codegen of 'else' can change the current block, update else_bb for the
           * phi. *)
          let new_else_bb = insertion_block builder in

Code generation for the 'else' block is basically identical to codegen
for the 'then' block.

.. code-block:: ocaml

          (* Emit merge block. *)
          let merge_bb = append_block context "ifcont" the_function in
          position_at_end merge_bb builder;
          let incoming = [(then_val, new_then_bb); (else_val, new_else_bb)] in
          let phi = build_phi incoming "iftmp" builder in

The first two lines here are now familiar: the first adds the "merge"
block to the Function object. The second changes the insertion
point so that newly created code will go into the "merge" block. Once
that is done, we need to create the PHI node and set up the block/value
pairs for the PHI.

.. code-block:: ocaml

          (* Return to the start block to add the conditional branch. *)
          position_at_end start_bb builder;
          ignore (build_cond_br cond_val then_bb else_bb builder);

Once the blocks are created, we can emit the conditional branch that
chooses between them. Note that creating new blocks does not implicitly
affect the IRBuilder, so it is still inserting into the block that the
condition went into. This is why we needed to save the "start" block.

.. code-block:: ocaml

          (* Set a unconditional branch at the end of the 'then' block and the
           * 'else' block to the 'merge' block. *)
          position_at_end new_then_bb builder; ignore (build_br merge_bb builder);
          position_at_end new_else_bb builder; ignore (build_br merge_bb builder);

          (* Finally, set the builder to the end of the merge block. *)
          position_at_end merge_bb builder;

          phi

To finish off the blocks, we create an unconditional branch to the merge
block. One interesting (and very important) aspect of the LLVM IR is
that it `requires all basic blocks to be
"terminated" <../LangRef.html#functionstructure>`_ with a `control flow
instruction <../LangRef.html#terminators>`_ such as return or branch.
This means that all control flow, *including fall throughs* must be made
explicit in the LLVM IR. If you violate this rule, the verifier will
emit an error.

Finally, the CodeGen function returns the phi node as the value computed
by the if/then/else expression. In our example above, this returned
value will feed into the code for the top-level function, which will
create the return instruction.

Overall, we now have the ability to execute conditional code in
Kaleidoscope. With this extension, Kaleidoscope is a fairly complete
language that can calculate a wide variety of numeric functions. Next up
we'll add another useful expression that is familiar from non-functional
languages...

'for' Loop Expression
=====================

Now that we know how to add basic control flow constructs to the
language, we have the tools to add more powerful things. Lets add
something more aggressive, a 'for' expression:

::

     extern putchard(char);
     def printstar(n)
       for i = 1, i < n, 1.0 in
         putchard(42);  # ascii 42 = '*'

     # print 100 '*' characters
     printstar(100);

This expression defines a new variable ("i" in this case) which iterates
from a starting value, while the condition ("i < n" in this case) is
true, incrementing by an optional step value ("1.0" in this case). If
the step value is omitted, it defaults to 1.0. While the loop is true,
it executes its body expression. Because we don't have anything better
to return, we'll just define the loop as always returning 0.0. In the
future when we have mutable variables, it will get more useful.

As before, lets talk about the changes that we need to Kaleidoscope to
support this.

Lexer Extensions for the 'for' Loop
-----------------------------------

The lexer extensions are the same sort of thing as for if/then/else:

.. code-block:: ocaml

      ... in Token.token ...
      (* control *)
      | If | Then | Else
      | For | In

      ... in Lexer.lex_ident...
          match Buffer.contents buffer with
          | "def" -> [< 'Token.Def; stream >]
          | "extern" -> [< 'Token.Extern; stream >]
          | "if" -> [< 'Token.If; stream >]
          | "then" -> [< 'Token.Then; stream >]
          | "else" -> [< 'Token.Else; stream >]
          | "for" -> [< 'Token.For; stream >]
          | "in" -> [< 'Token.In; stream >]
          | id -> [< 'Token.Ident id; stream >]

AST Extensions for the 'for' Loop
---------------------------------

The AST variant is just as simple. It basically boils down to capturing
the variable name and the constituent expressions in the node.

.. code-block:: ocaml

    type expr =
      ...
      (* variant for for/in. *)
      | For of string * expr * expr * expr option * expr

Parser Extensions for the 'for' Loop
------------------------------------

The parser code is also fairly standard. The only interesting thing here
is handling of the optional step value. The parser code handles it by
checking to see if the second comma is present. If not, it sets the step
value to null in the AST node:

.. code-block:: ocaml

    let rec parse_primary = parser
      ...
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

LLVM IR for the 'for' Loop
--------------------------

Now we get to the good part: the LLVM IR we want to generate for this
thing. With the simple example above, we get this LLVM IR (note that
this dump is generated with optimizations disabled for clarity):

.. code-block:: llvm

    declare double @putchard(double)

    define double @printstar(double %n) {
    entry:
            ; initial value = 1.0 (inlined into phi)
      br label %loop

    loop:    ; preds = %loop, %entry
      %i = phi double [ 1.000000e+00, %entry ], [ %nextvar, %loop ]
            ; body
      %calltmp = call double @putchard(double 4.200000e+01)
            ; increment
      %nextvar = fadd double %i, 1.000000e+00

            ; termination test
      %cmptmp = fcmp ult double %i, %n
      %booltmp = uitofp i1 %cmptmp to double
      %loopcond = fcmp one double %booltmp, 0.000000e+00
      br i1 %loopcond, label %loop, label %afterloop

    afterloop:    ; preds = %loop
            ; loop always returns 0.0
      ret double 0.000000e+00
    }

This loop contains all the same constructs we saw before: a phi node,
several expressions, and some basic blocks. Lets see how this fits
together.

Code Generation for the 'for' Loop
----------------------------------

The first part of Codegen is very simple: we just output the start
expression for the loop value:

.. code-block:: ocaml

    let rec codegen_expr = function
      ...
      | Ast.For (var_name, start, end_, step, body) ->
          (* Emit the start code first, without 'variable' in scope. *)
          let start_val = codegen_expr start in

With this out of the way, the next step is to set up the LLVM basic
block for the start of the loop body. In the case above, the whole loop
body is one block, but remember that the body code itself could consist
of multiple blocks (e.g. if it contains an if/then/else or a for/in
expression).

.. code-block:: ocaml

          (* Make the new basic block for the loop header, inserting after current
           * block. *)
          let preheader_bb = insertion_block builder in
          let the_function = block_parent preheader_bb in
          let loop_bb = append_block context "loop" the_function in

          (* Insert an explicit fall through from the current block to the
           * loop_bb. *)
          ignore (build_br loop_bb builder);

This code is similar to what we saw for if/then/else. Because we will
need it to create the Phi node, we remember the block that falls through
into the loop. Once we have that, we create the actual block that starts
the loop and create an unconditional branch for the fall-through between
the two blocks.

.. code-block:: ocaml

          (* Start insertion in loop_bb. *)
          position_at_end loop_bb builder;

          (* Start the PHI node with an entry for start. *)
          let variable = build_phi [(start_val, preheader_bb)] var_name builder in

Now that the "preheader" for the loop is set up, we switch to emitting
code for the loop body. To begin with, we move the insertion point and
create the PHI node for the loop induction variable. Since we already
know the incoming value for the starting value, we add it to the Phi
node. Note that the Phi will eventually get a second value for the
backedge, but we can't set it up yet (because it doesn't exist!).

.. code-block:: ocaml

          (* Within the loop, the variable is defined equal to the PHI node. If it
           * shadows an existing variable, we have to restore it, so save it
           * now. *)
          let old_val =
            try Some (Hashtbl.find named_values var_name) with Not_found -> None
          in
          Hashtbl.add named_values var_name variable;

          (* Emit the body of the loop.  This, like any other expr, can change the
           * current BB.  Note that we ignore the value computed by the body, but
           * don't allow an error *)
          ignore (codegen_expr body);

Now the code starts to get more interesting. Our 'for' loop introduces a
new variable to the symbol table. This means that our symbol table can
now contain either function arguments or loop variables. To handle this,
before we codegen the body of the loop, we add the loop variable as the
current value for its name. Note that it is possible that there is a
variable of the same name in the outer scope. It would be easy to make
this an error (emit an error and return null if there is already an
entry for VarName) but we choose to allow shadowing of variables. In
order to handle this correctly, we remember the Value that we are
potentially shadowing in ``old_val`` (which will be None if there is no
shadowed variable).

Once the loop variable is set into the symbol table, the code
recursively codegen's the body. This allows the body to use the loop
variable: any references to it will naturally find it in the symbol
table.

.. code-block:: ocaml

          (* Emit the step value. *)
          let step_val =
            match step with
            | Some step -> codegen_expr step
            (* If not specified, use 1.0. *)
            | None -> const_float double_type 1.0
          in

          let next_var = build_add variable step_val "nextvar" builder in

Now that the body is emitted, we compute the next value of the iteration
variable by adding the step value, or 1.0 if it isn't present.
'``next_var``' will be the value of the loop variable on the next
iteration of the loop.

.. code-block:: ocaml

          (* Compute the end condition. *)
          let end_cond = codegen_expr end_ in

          (* Convert condition to a bool by comparing equal to 0.0. *)
          let zero = const_float double_type 0.0 in
          let end_cond = build_fcmp Fcmp.One end_cond zero "loopcond" builder in

Finally, we evaluate the exit value of the loop, to determine whether
the loop should exit. This mirrors the condition evaluation for the
if/then/else statement.

.. code-block:: ocaml

          (* Create the "after loop" block and insert it. *)
          let loop_end_bb = insertion_block builder in
          let after_bb = append_block context "afterloop" the_function in

          (* Insert the conditional branch into the end of loop_end_bb. *)
          ignore (build_cond_br end_cond loop_bb after_bb builder);

          (* Any new code will be inserted in after_bb. *)
          position_at_end after_bb builder;

With the code for the body of the loop complete, we just need to finish
up the control flow for it. This code remembers the end block (for the
phi node), then creates the block for the loop exit ("afterloop"). Based
on the value of the exit condition, it creates a conditional branch that
chooses between executing the loop again and exiting the loop. Any
future code is emitted in the "afterloop" block, so it sets the
insertion position to it.

.. code-block:: ocaml

          (* Add a new entry to the PHI node for the backedge. *)
          add_incoming (next_var, loop_end_bb) variable;

          (* Restore the unshadowed variable. *)
          begin match old_val with
          | Some old_val -> Hashtbl.add named_values var_name old_val
          | None -> ()
          end;

          (* for expr always returns 0.0. *)
          const_null double_type

The final code handles various cleanups: now that we have the
"``next_var``" value, we can add the incoming value to the loop PHI
node. After that, we remove the loop variable from the symbol table, so
that it isn't in scope after the for loop. Finally, code generation of
the for loop always returns 0.0, so that is what we return from
``Codegen.codegen_expr``.

With this, we conclude the "adding control flow to Kaleidoscope" chapter
of the tutorial. In this chapter we added two control flow constructs,
and used them to motivate a couple of aspects of the LLVM IR that are
important for front-end implementors to know. In the next chapter of our
saga, we will get a bit crazier and add `user-defined
operators <OCamlLangImpl6.html>`_ to our poor innocent language.

Full Code Listing
=================

Here is the complete code listing for our running example, enhanced with
the if/then/else and for expressions.. To build this example, use:

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

        flag ["link"; "ocaml"; "g++"] (S[A"-cc"; A"g++"]);;
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

          (* variant for a binary operator. *)
          | Binary of char * expr * expr

          (* variant for function calls. *)
          | Call of string * expr array

          (* variant for if/then/else. *)
          | If of expr * expr * expr

          (* variant for for/in. *)
          | For of string * expr * expr * expr option * expr

        (* proto - This type represents the "prototype" for a function, which captures
         * its name, and its argument names (thus implicitly the number of arguments the
         * function takes). *)
        type proto = Prototype of string * string array

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
         *   ::= forexpr *)
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

          | [< >] -> raise (Stream.Error "unknown token when expecting an expression.")

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
                let rhs = parse_primary stream in

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

        (* expression
         *   ::= primary binoprhs *)
        and parse_expr = parser
          | [< lhs=parse_primary; stream >] -> parse_bin_rhs 0 lhs stream

        (* prototype
         *   ::= id '(' id* ')' *)
        let parse_prototype =
          let rec parse_args accumulator = parser
            | [< 'Token.Ident id; e=parse_args (id::accumulator) >] -> e
            | [< >] -> accumulator
          in

          parser
          | [< 'Token.Ident id;
               'Token.Kwd '(' ?? "expected '(' in prototype";
               args=parse_args [];
               'Token.Kwd ')' ?? "expected ')' in prototype" >] ->
              (* success. *)
              Ast.Prototype (id, Array.of_list (List.rev args))

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

        let rec codegen_expr = function
          | Ast.Number n -> const_float double_type n
          | Ast.Variable name ->
              (try Hashtbl.find named_values name with
                | Not_found -> raise (Error "unknown variable name"))
          | Ast.Binary (op, lhs, rhs) ->
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
                | _ -> raise (Error "invalid binary operator")
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
              (* Emit the start code first, without 'variable' in scope. *)
              let start_val = codegen_expr start in

              (* Make the new basic block for the loop header, inserting after current
               * block. *)
              let preheader_bb = insertion_block builder in
              let the_function = block_parent preheader_bb in
              let loop_bb = append_block context "loop" the_function in

              (* Insert an explicit fall through from the current block to the
               * loop_bb. *)
              ignore (build_br loop_bb builder);

              (* Start insertion in loop_bb. *)
              position_at_end loop_bb builder;

              (* Start the PHI node with an entry for start. *)
              let variable = build_phi [(start_val, preheader_bb)] var_name builder in

              (* Within the loop, the variable is defined equal to the PHI node. If it
               * shadows an existing variable, we have to restore it, so save it
               * now. *)
              let old_val =
                try Some (Hashtbl.find named_values var_name) with Not_found -> None
              in
              Hashtbl.add named_values var_name variable;

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

              let next_var = build_add variable step_val "nextvar" builder in

              (* Compute the end condition. *)
              let end_cond = codegen_expr end_ in

              (* Convert condition to a bool by comparing equal to 0.0. *)
              let zero = const_float double_type 0.0 in
              let end_cond = build_fcmp Fcmp.One end_cond zero "loopcond" builder in

              (* Create the "after loop" block and insert it. *)
              let loop_end_bb = insertion_block builder in
              let after_bb = append_block context "afterloop" the_function in

              (* Insert the conditional branch into the end of loop_end_bb. *)
              ignore (build_cond_br end_cond loop_bb after_bb builder);

              (* Any new code will be inserted in after_bb. *)
              position_at_end after_bb builder;

              (* Add a new entry to the PHI node for the backedge. *)
              add_incoming (next_var, loop_end_bb) variable;

              (* Restore the unshadowed variable. *)
              begin match old_val with
              | Some old_val -> Hashtbl.add named_values var_name old_val
              | None -> ()
              end;

              (* for expr always returns 0.0. *)
              const_null double_type

        let codegen_proto = function
          | Ast.Prototype (name, args) ->
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

        let codegen_func the_fpm = function
          | Ast.Function (proto, body) ->
              Hashtbl.clear named_values;
              let the_function = codegen_proto proto in

              (* Create a new basic block to start insertion into. *)
              let bb = append_block context "entry" the_function in
              position_at_end bb builder;

              try
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

`Next: Extending the language: user-defined
operators <OCamlLangImpl6.html>`_

