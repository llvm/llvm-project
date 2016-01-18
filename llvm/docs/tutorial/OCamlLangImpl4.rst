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

**Note:** the default ``IRBuilder`` now always includes the constant
folding optimisations below.

Our demonstration for Chapter 3 is elegant and easy to extend.
Unfortunately, it does not produce wonderful code. For example, when
compiling simple code, we don't get obvious optimizations:

::

    ready> def test(x) 1+2+x;
    Read function definition:
    define double @test(double %x) {
    entry:
            %addtmp = fadd double 1.000000e+00, 2.000000e+00
            %addtmp1 = fadd double %addtmp, %x
            ret double %addtmp1
    }

This code is a very, very literal transcription of the AST built by
parsing the input. As such, this transcription lacks optimizations like
constant folding (we'd like to get "``add x, 3.0``" in the example
above) as well as other more important optimizations. Constant folding,
in particular, is a very common and very important optimization: so much
so that many language implementors implement constant folding support in
their AST representation.

With LLVM, you don't need this support in the AST. Since all calls to
build LLVM IR go through the LLVM builder, it would be nice if the
builder itself checked to see if there was a constant folding
opportunity when you call it. If so, it could just do the constant fold
and return the constant instead of creating an instruction. This is
exactly what the ``LLVMFoldingBuilder`` class does.

All we did was switch from ``LLVMBuilder`` to ``LLVMFoldingBuilder``.
Though we change no other code, we now have all of our instructions
implicitly constant folded without us having to do anything about it.
For example, the input above now compiles to:

::

    ready> def test(x) 1+2+x;
    Read function definition:
    define double @test(double %x) {
    entry:
            %addtmp = fadd double 3.000000e+00, %x
            ret double %addtmp
    }

Well, that was easy :). In practice, we recommend always using
``LLVMFoldingBuilder`` when generating code like this. It has no
"syntactic overhead" for its use (you don't have to uglify your compiler
with constant checks everywhere) and it can dramatically reduce the
amount of LLVM IR that is generated in some cases (particular for
languages with a macro preprocessor or that use a lot of constants).

On the other hand, the ``LLVMFoldingBuilder`` is limited by the fact
that it does all of its analysis inline with the code as it is built. If
you take a slightly more complex example:

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
instead of computing "``x*3``" twice.

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
`Llvm.PassManager <../WritingAnLLVMPass.html#passmanager>`_ to hold and
organize the LLVM optimizations that we want to run. Once we have that,
we can add a set of optimizations to run. The code looks like this:

.. code-block:: ocaml

      (* Create the JIT. *)
      let the_execution_engine = ExecutionEngine.create Codegen.the_module in
      let the_fpm = PassManager.create_function Codegen.the_module in

      (* Set up the optimizer pipeline.  Start with registering info about how the
       * target lays out data structures. *)
      DataLayout.add (ExecutionEngine.target_data the_execution_engine) the_fpm;

      (* Do simple "peephole" optimizations and bit-twiddling optzn. *)
      add_instruction_combining the_fpm;

      (* reassociate expressions. *)
      add_reassociation the_fpm;

      (* Eliminate Common SubExpressions. *)
      add_gvn the_fpm;

      (* Simplify the control flow graph (deleting unreachable blocks, etc). *)
      add_cfg_simplification the_fpm;

      ignore (PassManager.initialize the_fpm);

      (* Run the main "interpreter loop" now. *)
      Toplevel.main_loop the_fpm the_execution_engine stream;

The meat of the matter here, is the definition of "``the_fpm``". It
requires a pointer to the ``the_module`` to construct itself. Once it is
set up, we use a series of "add" calls to add a bunch of LLVM passes.
The first pass is basically boilerplate, it adds a pass so that later
optimizations know how the data structures in the program are laid out.
The "``the_execution_engine``" variable is related to the JIT, which we
will get to in the next section.

In this case, we choose to add 4 optimization passes. The passes we
chose here are a pretty standard set of "cleanup" optimizations that are
useful for a wide variety of code. I won't delve into what they do but,
believe me, they are a good starting place :).

Once the ``Llvm.PassManager.`` is set up, we need to make use of it. We
do this by running it after our newly created function is constructed
(in ``Codegen.codegen_func``), but before it is returned to the client:

.. code-block:: ocaml

    let codegen_func the_fpm = function
          ...
          try
            let ret_val = codegen_expr body in

            (* Finish off the function. *)
            let _ = build_ret ret_val builder in

            (* Validate the generated code, checking for consistency. *)
            Llvm_analysis.assert_valid_function the_function;

            (* Optimize the function. *)
            let _ = PassManager.run_function the_function the_fpm in

            the_function

As you can see, this is pretty straightforward. The ``the_fpm``
optimizes and updates the LLVM Function\* in place, improving
(hopefully) its body. With this in place, we can try our test above
again:

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
done by adding a global variable and a call in ``main``:

.. code-block:: ocaml

    ...
    let main () =
      ...
      (* Create the JIT. *)
      let the_execution_engine = ExecutionEngine.create Codegen.the_module in
      ...

This creates an abstract "Execution Engine" which can be either a JIT
compiler or the LLVM interpreter. LLVM will automatically pick a JIT
compiler for you if one is available for your platform, otherwise it
will fall back to the interpreter.

Once the ``Llvm_executionengine.ExecutionEngine.t`` is created, the JIT
is ready to be used. There are a variety of APIs that are useful, but
the simplest one is the
"``Llvm_executionengine.ExecutionEngine.run_function``" function. This
method JIT compiles the specified LLVM Function and returns a function
pointer to the generated machine code. In our case, this means that we
can change the code that parses a top-level expression to look like
this:

.. code-block:: ocaml

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

Recall that we compile top-level expressions into a self-contained LLVM
function that takes no arguments and returns the computed double.
Because the LLVM JIT compiler matches the native platform ABI, this
means that you can just cast the result pointer to a function pointer of
that type and call it directly. This means, there is no difference
between JIT compiled code and native machine code that is statically
linked into your application.

With just these two changes, lets see how Kaleidoscope works now!

::

    ready> 4+5;
    define double @""() {
    entry:
            ret double 9.000000e+00
    }

    Evaluated to 9.000000

Well this looks like it is basically working. The dump of the function
shows the "no argument function that always returns double" that we
synthesize for each top level expression that is typed in. This
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
    define double @""() {
    entry:
            %calltmp = call double @testfunc(double 4.000000e+00, double 1.000000e+01)
            ret double %calltmp
    }

    Evaluated to 24.000000

This illustrates that we can now call user code, but there is something
a bit subtle going on here. Note that we only invoke the JIT on the
anonymous functions that *call testfunc*, but we never invoked it on
*testfunc* itself. What actually happened here is that the JIT scanned
for all non-JIT'd functions transitively called from the anonymous
function and compiled all of them before returning from
``run_function``.

The JIT provides a number of other more advanced interfaces for things
like freeing allocated machine code, rejit'ing functions to update them,
etc. However, even with this simple code, we get some surprisingly
powerful capabilities - check this out (I removed the dump of the
anonymous functions, you should get the idea by now :) :

::

    ready> extern sin(x);
    Read extern:
    declare double @sin(double)

    ready> extern cos(x);
    Read extern:
    declare double @cos(double)

    ready> sin(1.0);
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
    Evaluated to 1.000000

Whoa, how does the JIT know about sin and cos? The answer is
surprisingly simple: in this example, the JIT started execution of a
function and got to a function call. It realized that the function was
not yet JIT compiled and invoked the standard set of routines to resolve
the function. In this case, there is no body defined for the function,
so the JIT ended up calling "``dlsym("sin")``" on the Kaleidoscope
process itself. Since "``sin``" is defined within the JIT's address
space, it simply patches up calls in the module to call the libm version
of ``sin`` directly.

The LLVM JIT provides a number of interfaces (look in the
``llvm_executionengine.mli`` file) for controlling how unknown functions
get resolved. It allows you to establish explicit mappings between IR
objects and addresses (useful for LLVM global variables that you want to
map to static tables, for example), allows you to dynamically decide on
the fly based on the function name, and even allows you to have the JIT
compile functions lazily the first time they're called.

One interesting application of this is that we can now extend the
language by writing arbitrary C code to implement operations. For
example, if we add:

.. code-block:: c++

    /* putchard - putchar that takes a double and returns 0. */
    extern "C"
    double putchard(double X) {
      putchar((char)X);
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
constructs <OCamlLangImpl5.html>`_, tackling some interesting LLVM IR
issues along the way.

Full Code Listing
=================

Here is the complete code listing for our running example, enhanced with
the LLVM JIT and optimizer. To build this example, use:

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
         *   ::= parenexpr *)
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

`Next: Extending the language: control flow <OCamlLangImpl5.html>`_

