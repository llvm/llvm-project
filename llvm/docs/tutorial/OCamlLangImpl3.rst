========================================
Kaleidoscope: Code generation to LLVM IR
========================================

.. contents::
   :local:

Chapter 3 Introduction
======================

Welcome to Chapter 3 of the "`Implementing a language with
LLVM <index.html>`_" tutorial. This chapter shows you how to transform
the `Abstract Syntax Tree <OCamlLangImpl2.html>`_, built in Chapter 2,
into LLVM IR. This will teach you a little bit about how LLVM does
things, as well as demonstrate how easy it is to use. It's much more
work to build a lexer and parser than it is to generate LLVM IR code. :)

**Please note**: the code in this chapter and later require LLVM 2.3 or
LLVM SVN to work. LLVM 2.2 and before will not work with it.

Code Generation Setup
=====================

In order to generate LLVM IR, we want some simple setup to get started.
First we define virtual code generation (codegen) methods in each AST
class:

.. code-block:: ocaml

    let rec codegen_expr = function
      | Ast.Number n -> ...
      | Ast.Variable name -> ...

The ``Codegen.codegen_expr`` function says to emit IR for that AST node
along with all the things it depends on, and they all return an LLVM
Value object. "Value" is the class used to represent a "`Static Single
Assignment
(SSA) <http://en.wikipedia.org/wiki/Static_single_assignment_form>`_
register" or "SSA value" in LLVM. The most distinct aspect of SSA values
is that their value is computed as the related instruction executes, and
it does not get a new value until (and if) the instruction re-executes.
In other words, there is no way to "change" an SSA value. For more
information, please read up on `Static Single
Assignment <http://en.wikipedia.org/wiki/Static_single_assignment_form>`_
- the concepts are really quite natural once you grok them.

The second thing we want is an "Error" exception like we used for the
parser, which will be used to report errors found during code generation
(for example, use of an undeclared parameter):

.. code-block:: ocaml

    exception Error of string

    let context = global_context ()
    let the_module = create_module context "my cool jit"
    let builder = builder context
    let named_values:(string, llvalue) Hashtbl.t = Hashtbl.create 10
    let double_type = double_type context

The static variables will be used during code generation.
``Codegen.the_module`` is the LLVM construct that contains all of the
functions and global variables in a chunk of code. In many ways, it is
the top-level structure that the LLVM IR uses to contain code.

The ``Codegen.builder`` object is a helper object that makes it easy to
generate LLVM instructions. Instances of the
`IRBuilder <https://llvm.org/doxygen/IRBuilder_8h-source.html>`_
class keep track of the current place to insert instructions and has
methods to create new instructions.

The ``Codegen.named_values`` map keeps track of which values are defined
in the current scope and what their LLVM representation is. (In other
words, it is a symbol table for the code). In this form of Kaleidoscope,
the only things that can be referenced are function parameters. As such,
function parameters will be in this map when generating code for their
function body.

With these basics in place, we can start talking about how to generate
code for each expression. Note that this assumes that the
``Codegen.builder`` has been set up to generate code *into* something.
For now, we'll assume that this has already been done, and we'll just
use it to emit code.

Expression Code Generation
==========================

Generating LLVM code for expression nodes is very straightforward: less
than 30 lines of commented code for all four of our expression nodes.
First we'll do numeric literals:

.. code-block:: ocaml

      | Ast.Number n -> const_float double_type n

In the LLVM IR, numeric constants are represented with the
``ConstantFP`` class, which holds the numeric value in an ``APFloat``
internally (``APFloat`` has the capability of holding floating point
constants of Arbitrary Precision). This code basically just creates
and returns a ``ConstantFP``. Note that in the LLVM IR that constants
are all uniqued together and shared. For this reason, the API uses "the
foo::get(..)" idiom instead of "new foo(..)" or "foo::Create(..)".

.. code-block:: ocaml

      | Ast.Variable name ->
          (try Hashtbl.find named_values name with
            | Not_found -> raise (Error "unknown variable name"))

References to variables are also quite simple using LLVM. In the simple
version of Kaleidoscope, we assume that the variable has already been
emitted somewhere and its value is available. In practice, the only
values that can be in the ``Codegen.named_values`` map are function
arguments. This code simply checks to see that the specified name is in
the map (if not, an unknown variable is being referenced) and returns
the value for it. In future chapters, we'll add support for `loop
induction variables <LangImpl5.html#for-loop-expression>`_ in the symbol table, and for
`local variables <LangImpl7.html#user-defined-local-variables>`_.

.. code-block:: ocaml

      | Ast.Binary (op, lhs, rhs) ->
          let lhs_val = codegen_expr lhs in
          let rhs_val = codegen_expr rhs in
          begin
            match op with
            | '+' -> build_fadd lhs_val rhs_val "addtmp" builder
            | '-' -> build_fsub lhs_val rhs_val "subtmp" builder
            | '*' -> build_fmul lhs_val rhs_val "multmp" builder
            | '<' ->
                (* Convert bool 0/1 to double 0.0 or 1.0 *)
                let i = build_fcmp Fcmp.Ult lhs_val rhs_val "cmptmp" builder in
                build_uitofp i double_type "booltmp" builder
            | _ -> raise (Error "invalid binary operator")
          end

Binary operators start to get more interesting. The basic idea here is
that we recursively emit code for the left-hand side of the expression,
then the right-hand side, then we compute the result of the binary
expression. In this code, we do a simple switch on the opcode to create
the right LLVM instruction.

In the example above, the LLVM builder class is starting to show its
value. IRBuilder knows where to insert the newly created instruction,
all you have to do is specify what instruction to create (e.g. with
``Llvm.create_add``), which operands to use (``lhs`` and ``rhs`` here)
and optionally provide a name for the generated instruction.

One nice thing about LLVM is that the name is just a hint. For instance,
if the code above emits multiple "addtmp" variables, LLVM will
automatically provide each one with an increasing, unique numeric
suffix. Local value names for instructions are purely optional, but it
makes it much easier to read the IR dumps.

`LLVM instructions <../LangRef.html#instruction-reference>`_ are constrained by strict
rules: for example, the Left and Right operators of an `add
instruction <../LangRef.html#add-instruction>`_ must have the same type, and the
result type of the add must match the operand types. Because all values
in Kaleidoscope are doubles, this makes for very simple code for add,
sub and mul.

On the other hand, LLVM specifies that the `fcmp
instruction <../LangRef.html#fcmp-instruction>`_ always returns an 'i1' value (a
one bit integer). The problem with this is that Kaleidoscope wants the
value to be a 0.0 or 1.0 value. In order to get these semantics, we
combine the fcmp instruction with a `uitofp
instruction <../LangRef.html#uitofp-to-instruction>`_. This instruction converts its
input integer into a floating point value by treating the input as an
unsigned value. In contrast, if we used the `sitofp
instruction <../LangRef.html#sitofp-to-instruction>`_, the Kaleidoscope '<' operator
would return 0.0 and -1.0, depending on the input value.

.. code-block:: ocaml

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

Code generation for function calls is quite straightforward with LLVM.
The code above initially does a function name lookup in the LLVM
Module's symbol table. Recall that the LLVM Module is the container that
holds all of the functions we are JIT'ing. By giving each function the
same name as what the user specifies, we can use the LLVM symbol table
to resolve function names for us.

Once we have the function to call, we recursively codegen each argument
that is to be passed in, and create an LLVM `call
instruction <../LangRef.html#call-instruction>`_. Note that LLVM uses the native C
calling conventions by default, allowing these calls to also call into
standard library functions like "sin" and "cos", with no additional
effort.

This wraps up our handling of the four basic expressions that we have so
far in Kaleidoscope. Feel free to go in and add some more. For example,
by browsing the `LLVM language reference <../LangRef.html>`_ you'll find
several other interesting instructions that are really easy to plug into
our basic framework.

Function Code Generation
========================

Code generation for prototypes and functions must handle a number of
details, which make their code less beautiful than expression code
generation, but allows us to illustrate some important points. First,
lets talk about code generation for prototypes: they are used both for
function bodies and external function declarations. The code starts
with:

.. code-block:: ocaml

    let codegen_proto = function
      | Ast.Prototype (name, args) ->
          (* Make the function type: double(double,double) etc. *)
          let doubles = Array.make (Array.length args) double_type in
          let ft = function_type double_type doubles in
          let f =
            match lookup_function name the_module with

This code packs a lot of power into a few lines. Note first that this
function returns a "Function\*" instead of a "Value\*" (although at the
moment they both are modeled by ``llvalue`` in ocaml). Because a
"prototype" really talks about the external interface for a function
(not the value computed by an expression), it makes sense for it to
return the LLVM Function it corresponds to when codegen'd.

The call to ``Llvm.function_type`` creates the ``Llvm.llvalue`` that
should be used for a given Prototype. Since all function arguments in
Kaleidoscope are of type double, the first line creates a vector of "N"
LLVM double types. It then uses the ``Llvm.function_type`` method to
create a function type that takes "N" doubles as arguments, returns one
double as a result, and that is not vararg (that uses the function
``Llvm.var_arg_function_type``). Note that Types in LLVM are uniqued
just like ``Constant``'s are, so you don't "new" a type, you "get" it.

The final line above checks if the function has already been defined in
``Codegen.the_module``. If not, we will create it.

.. code-block:: ocaml

            | None -> declare_function name ft the_module

This indicates the type and name to use, as well as which module to
insert into. By default we assume a function has
``Llvm.Linkage.ExternalLinkage``. "`external
linkage <../LangRef.html#linkage>`_" means that the function may be defined
outside the current module and/or that it is callable by functions
outside the module. The "``name``" passed in is the name the user
specified: this name is registered in "``Codegen.the_module``"s symbol
table, which is used by the function call code above.

In Kaleidoscope, I choose to allow redefinitions of functions in two
cases: first, we want to allow 'extern'ing a function more than once, as
long as the prototypes for the externs match (since all arguments have
the same type, we just have to check that the number of arguments
match). Second, we want to allow 'extern'ing a function and then
defining a body for it. This is useful when defining mutually recursive
functions.

.. code-block:: ocaml

            (* If 'f' conflicted, there was already something named 'name'. If it
             * has a body, don't allow redefinition or reextern. *)
            | Some f ->
                (* If 'f' already has a body, reject this. *)
                if Array.length (basic_blocks f) == 0 then () else
                  raise (Error "redefinition of function");

                (* If 'f' took a different number of arguments, reject. *)
                if Array.length (params f) == Array.length args then () else
                  raise (Error "redefinition of function with different # args");
                f
          in

In order to verify the logic above, we first check to see if the
pre-existing function is "empty". In this case, empty means that it has
no basic blocks in it, which means it has no body. If it has no body, it
is a forward declaration. Since we don't allow anything after a full
definition of the function, the code rejects this case. If the previous
reference to a function was an 'extern', we simply verify that the
number of arguments for that definition and this one match up. If not,
we emit an error.

.. code-block:: ocaml

          (* Set names for all arguments. *)
          Array.iteri (fun i a ->
            let n = args.(i) in
            set_value_name n a;
            Hashtbl.add named_values n a;
          ) (params f);
          f

The last bit of code for prototypes loops over all of the arguments in
the function, setting the name of the LLVM Argument objects to match,
and registering the arguments in the ``Codegen.named_values`` map for
future use by the ``Ast.Variable`` variant. Once this is set up, it
returns the Function object to the caller. Note that we don't check for
conflicting argument names here (e.g. "extern foo(a b a)"). Doing so
would be very straight-forward with the mechanics we have already used
above.

.. code-block:: ocaml

    let codegen_func = function
      | Ast.Function (proto, body) ->
          Hashtbl.clear named_values;
          let the_function = codegen_proto proto in

Code generation for function definitions starts out simply enough: we
just codegen the prototype (Proto) and verify that it is ok. We then
clear out the ``Codegen.named_values`` map to make sure that there isn't
anything in it from the last function we compiled. Code generation of
the prototype ensures that there is an LLVM Function object that is
ready to go for us.

.. code-block:: ocaml

          (* Create a new basic block to start insertion into. *)
          let bb = append_block context "entry" the_function in
          position_at_end bb builder;

          try
            let ret_val = codegen_expr body in

Now we get to the point where the ``Codegen.builder`` is set up. The
first line creates a new `basic
block <http://en.wikipedia.org/wiki/Basic_block>`_ (named "entry"),
which is inserted into ``the_function``. The second line then tells the
builder that new instructions should be inserted into the end of the new
basic block. Basic blocks in LLVM are an important part of functions
that define the `Control Flow
Graph <http://en.wikipedia.org/wiki/Control_flow_graph>`_. Since we
don't have any control flow, our functions will only contain one block
at this point. We'll fix this in `Chapter 5 <OCamlLangImpl5.html>`_ :).

.. code-block:: ocaml

            let ret_val = codegen_expr body in

            (* Finish off the function. *)
            let _ = build_ret ret_val builder in

            (* Validate the generated code, checking for consistency. *)
            Llvm_analysis.assert_valid_function the_function;

            the_function

Once the insertion point is set up, we call the ``Codegen.codegen_func``
method for the root expression of the function. If no error happens,
this emits code to compute the expression into the entry block and
returns the value that was computed. Assuming no error, we then create
an LLVM `ret instruction <../LangRef.html#ret-instruction>`_, which completes the
function. Once the function is built, we call
``Llvm_analysis.assert_valid_function``, which is provided by LLVM. This
function does a variety of consistency checks on the generated code, to
determine if our compiler is doing everything right. Using this is
important: it can catch a lot of bugs. Once the function is finished and
validated, we return it.

.. code-block:: ocaml

          with e ->
            delete_function the_function;
            raise e

The only piece left here is handling of the error case. For simplicity,
we handle this by merely deleting the function we produced with the
``Llvm.delete_function`` method. This allows the user to redefine a
function that they incorrectly typed in before: if we didn't delete it,
it would live in the symbol table, with a body, preventing future
redefinition.

This code does have a bug, though. Since the ``Codegen.codegen_proto``
can return a previously defined forward declaration, our code can
actually delete a forward declaration. There are a number of ways to fix
this bug, see what you can come up with! Here is a testcase:

::

    extern foo(a b);     # ok, defines foo.
    def foo(a b) c;      # error, 'c' is invalid.
    def bar() foo(1, 2); # error, unknown function "foo"

Driver Changes and Closing Thoughts
===================================

For now, code generation to LLVM doesn't really get us much, except that
we can look at the pretty IR calls. The sample code inserts calls to
Codegen into the "``Toplevel.main_loop``", and then dumps out the LLVM
IR. This gives a nice way to look at the LLVM IR for simple functions.
For example:

::

    ready> 4+5;
    Read top-level expression:
    define double @""() {
    entry:
            %addtmp = fadd double 4.000000e+00, 5.000000e+00
            ret double %addtmp
    }

Note how the parser turns the top-level expression into anonymous
functions for us. This will be handy when we add `JIT
support <OCamlLangImpl4.html#adding-a-jit-compiler>`_ in the next chapter. Also note that
the code is very literally transcribed, no optimizations are being
performed. We will `add
optimizations <OCamlLangImpl4.html#trivial-constant-folding>`_ explicitly in the
next chapter.

::

    ready> def foo(a b) a*a + 2*a*b + b*b;
    Read function definition:
    define double @foo(double %a, double %b) {
    entry:
            %multmp = fmul double %a, %a
            %multmp1 = fmul double 2.000000e+00, %a
            %multmp2 = fmul double %multmp1, %b
            %addtmp = fadd double %multmp, %multmp2
            %multmp3 = fmul double %b, %b
            %addtmp4 = fadd double %addtmp, %multmp3
            ret double %addtmp4
    }

This shows some simple arithmetic. Notice the striking similarity to the
LLVM builder calls that we use to create the instructions.

::

    ready> def bar(a) foo(a, 4.0) + bar(31337);
    Read function definition:
    define double @bar(double %a) {
    entry:
            %calltmp = call double @foo(double %a, double 4.000000e+00)
            %calltmp1 = call double @bar(double 3.133700e+04)
            %addtmp = fadd double %calltmp, %calltmp1
            ret double %addtmp
    }

This shows some function calls. Note that this function will take a long
time to execute if you call it. In the future we'll add conditional
control flow to actually make recursion useful :).

::

    ready> extern cos(x);
    Read extern:
    declare double @cos(double)

    ready> cos(1.234);
    Read top-level expression:
    define double @""() {
    entry:
            %calltmp = call double @cos(double 1.234000e+00)
            ret double %calltmp
    }

This shows an extern for the libm "cos" function, and a call to it.

::

    ready> ^D
    ; ModuleID = 'my cool jit'

    define double @""() {
    entry:
            %addtmp = fadd double 4.000000e+00, 5.000000e+00
            ret double %addtmp
    }

    define double @foo(double %a, double %b) {
    entry:
            %multmp = fmul double %a, %a
            %multmp1 = fmul double 2.000000e+00, %a
            %multmp2 = fmul double %multmp1, %b
            %addtmp = fadd double %multmp, %multmp2
            %multmp3 = fmul double %b, %b
            %addtmp4 = fadd double %addtmp, %multmp3
            ret double %addtmp4
    }

    define double @bar(double %a) {
    entry:
            %calltmp = call double @foo(double %a, double 4.000000e+00)
            %calltmp1 = call double @bar(double 3.133700e+04)
            %addtmp = fadd double %calltmp, %calltmp1
            ret double %addtmp
    }

    declare double @cos(double)

    define double @""() {
    entry:
            %calltmp = call double @cos(double 1.234000e+00)
            ret double %calltmp
    }

When you quit the current demo, it dumps out the IR for the entire
module generated. Here you can see the big picture with all the
functions referencing each other.

This wraps up the third chapter of the Kaleidoscope tutorial. Up next,
we'll describe how to `add JIT codegen and optimizer
support <OCamlLangImpl4.html>`_ to this so we can actually start running
code!

Full Code Listing
=================

Here is the complete code listing for our running example, enhanced with
the LLVM code generator. Because this uses the LLVM libraries, we need
to link them in. To do this, we use the
`llvm-config <https://llvm.org/cmds/llvm-config.html>`_ tool to inform
our makefile/command line about which options to use:

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

myocamlbuild.ml:
    .. code-block:: ocaml

        open Ocamlbuild_plugin;;

        ocaml_lib ~extern:true "llvm";;
        ocaml_lib ~extern:true "llvm_analysis";;

        flag ["link"; "ocaml"; "g++"] (S[A"-cc"; A"g++"]);;

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

        let codegen_func = function
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

        (* top ::= definition | external | expression | ';' *)
        let rec main_loop stream =
          match Stream.peek stream with
          | None -> ()

          (* ignore top-level semicolons. *)
          | Some (Token.Kwd ';') ->
              Stream.junk stream;
              main_loop stream

          | Some token ->
              begin
                try match token with
                | Token.Def ->
                    let e = Parser.parse_definition stream in
                    print_endline "parsed a function definition.";
                    dump_value (Codegen.codegen_func e);
                | Token.Extern ->
                    let e = Parser.parse_extern stream in
                    print_endline "parsed an extern.";
                    dump_value (Codegen.codegen_proto e);
                | _ ->
                    (* Evaluate a top-level expression into an anonymous function. *)
                    let e = Parser.parse_toplevel stream in
                    print_endline "parsed a top-level expr";
                    dump_value (Codegen.codegen_func e);
                with Stream.Error s | Codegen.Error s ->
                  (* Skip token for error recovery. *)
                  Stream.junk stream;
                  print_endline s;
              end;
              print_string "ready> "; flush stdout;
              main_loop stream

toy.ml:
    .. code-block:: ocaml

        (*===----------------------------------------------------------------------===
         * Main driver code.
         *===----------------------------------------------------------------------===*)

        open Llvm

        let main () =
          (* Install standard binary operators.
           * 1 is the lowest precedence. *)
          Hashtbl.add Parser.binop_precedence '<' 10;
          Hashtbl.add Parser.binop_precedence '+' 20;
          Hashtbl.add Parser.binop_precedence '-' 20;
          Hashtbl.add Parser.binop_precedence '*' 40;    (* highest. *)

          (* Prime the first token. *)
          print_string "ready> "; flush stdout;
          let stream = Lexer.lex (Stream.of_channel stdin) in

          (* Run the main "interpreter loop" now. *)
          Toplevel.main_loop stream;

          (* Print out all the generated code. *)
          dump_module Codegen.the_module
        ;;

        main ()

`Next: Adding JIT and Optimizer Support <OCamlLangImpl4.html>`_

