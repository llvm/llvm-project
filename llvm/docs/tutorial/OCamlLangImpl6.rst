============================================================
Kaleidoscope: Extending the Language: User-defined Operators
============================================================

.. contents::
   :local:

Chapter 6 Introduction
======================

Welcome to Chapter 6 of the "`Implementing a language with
LLVM <index.html>`_" tutorial. At this point in our tutorial, we now
have a fully functional language that is fairly minimal, but also
useful. There is still one big problem with it, however. Our language
doesn't have many useful operators (like division, logical negation, or
even any comparisons besides less-than).

This chapter of the tutorial takes a wild digression into adding
user-defined operators to the simple and beautiful Kaleidoscope
language. This digression now gives us a simple and ugly language in
some ways, but also a powerful one at the same time. One of the great
things about creating your own language is that you get to decide what
is good or bad. In this tutorial we'll assume that it is okay to use
this as a way to show some interesting parsing techniques.

At the end of this tutorial, we'll run through an example Kaleidoscope
application that `renders the Mandelbrot set <#kicking-the-tires>`_. This gives an
example of what you can build with Kaleidoscope and its feature set.

User-defined Operators: the Idea
================================

The "operator overloading" that we will add to Kaleidoscope is more
general than languages like C++. In C++, you are only allowed to
redefine existing operators: you can't programatically change the
grammar, introduce new operators, change precedence levels, etc. In this
chapter, we will add this capability to Kaleidoscope, which will let the
user round out the set of operators that are supported.

The point of going into user-defined operators in a tutorial like this
is to show the power and flexibility of using a hand-written parser.
Thus far, the parser we have been implementing uses recursive descent
for most parts of the grammar and operator precedence parsing for the
expressions. See `Chapter 2 <OCamlLangImpl2.html>`_ for details. Without
using operator precedence parsing, it would be very difficult to allow
the programmer to introduce new operators into the grammar: the grammar
is dynamically extensible as the JIT runs.

The two specific features we'll add are programmable unary operators
(right now, Kaleidoscope has no unary operators at all) as well as
binary operators. An example of this is:

::

    # Logical unary not.
    def unary!(v)
      if v then
        0
      else
        1;

    # Define > with the same precedence as <.
    def binary> 10 (LHS RHS)
      RHS < LHS;

    # Binary "logical or", (note that it does not "short circuit")
    def binary| 5 (LHS RHS)
      if LHS then
        1
      else if RHS then
        1
      else
        0;

    # Define = with slightly lower precedence than relationals.
    def binary= 9 (LHS RHS)
      !(LHS < RHS | LHS > RHS);

Many languages aspire to being able to implement their standard runtime
library in the language itself. In Kaleidoscope, we can implement
significant parts of the language in the library!

We will break down implementation of these features into two parts:
implementing support for user-defined binary operators and adding unary
operators.

User-defined Binary Operators
=============================

Adding support for user-defined binary operators is pretty simple with
our current framework. We'll first add support for the unary/binary
keywords:

.. code-block:: ocaml

    type token =
      ...
      (* operators *)
      | Binary | Unary

    ...

    and lex_ident buffer = parser
      ...
          | "for" -> [< 'Token.For; stream >]
          | "in" -> [< 'Token.In; stream >]
          | "binary" -> [< 'Token.Binary; stream >]
          | "unary" -> [< 'Token.Unary; stream >]

This just adds lexer support for the unary and binary keywords, like we
did in `previous chapters <OCamlLangImpl5.html#lexer-extensions-for-if-then-else>`_. One nice
thing about our current AST, is that we represent binary operators with
full generalisation by using their ASCII code as the opcode. For our
extended operators, we'll use this same representation, so we don't need
any new AST or parser support.

On the other hand, we have to be able to represent the definitions of
these new operators, in the "def binary\| 5" part of the function
definition. In our grammar so far, the "name" for the function
definition is parsed as the "prototype" production and into the
``Ast.Prototype`` AST node. To represent our new user-defined operators
as prototypes, we have to extend the ``Ast.Prototype`` AST node like
this:

.. code-block:: ocaml

    (* proto - This type represents the "prototype" for a function, which captures
     * its name, and its argument names (thus implicitly the number of arguments the
     * function takes). *)
    type proto =
      | Prototype of string * string array
      | BinOpPrototype of string * string array * int

Basically, in addition to knowing a name for the prototype, we now keep
track of whether it was an operator, and if it was, what precedence
level the operator is at. The precedence is only used for binary
operators (as you'll see below, it just doesn't apply for unary
operators). Now that we have a way to represent the prototype for a
user-defined operator, we need to parse it:

.. code-block:: ocaml

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

This is all fairly straightforward parsing code, and we have already
seen a lot of similar code in the past. One interesting part about the
code above is the couple lines that set up ``name`` for binary
operators. This builds names like "binary@" for a newly defined "@"
operator. This then takes advantage of the fact that symbol names in the
LLVM symbol table are allowed to have any character in them, including
embedded nul characters.

The next interesting thing to add, is codegen support for these binary
operators. Given our current structure, this is a simple addition of a
default case for our existing binary operator node:

.. code-block:: ocaml

    let codegen_expr = function
      ...
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

As you can see above, the new code is actually really simple. It just
does a lookup for the appropriate operator in the symbol table and
generates a function call to it. Since user-defined operators are just
built as normal functions (because the "prototype" boils down to a
function with the right name) everything falls into place.

The final piece of code we are missing, is a bit of top level magic:

.. code-block:: ocaml

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
          ...

Basically, before codegening a function, if it is a user-defined
operator, we register it in the precedence table. This allows the binary
operator parsing logic we already have in place to handle it. Since we
are working on a fully-general operator precedence parser, this is all
we need to do to "extend the grammar".

Now we have useful user-defined binary operators. This builds a lot on
the previous framework we built for other operators. Adding unary
operators is a bit more challenging, because we don't have any framework
for it yet - lets see what it takes.

User-defined Unary Operators
============================

Since we don't currently support unary operators in the Kaleidoscope
language, we'll need to add everything to support them. Above, we added
simple support for the 'unary' keyword to the lexer. In addition to
that, we need an AST node:

.. code-block:: ocaml

    type expr =
      ...
      (* variant for a unary operator. *)
      | Unary of char * expr
      ...

This AST node is very simple and obvious by now. It directly mirrors the
binary operator AST node, except that it only has one child. With this,
we need to add the parsing logic. Parsing a unary operator is pretty
simple: we'll add a new function to do it:

.. code-block:: ocaml

    (* unary
     *   ::= primary
     *   ::= '!' unary *)
    and parse_unary = parser
      (* If this is a unary operator, read it. *)
      | [< 'Token.Kwd op when op != '(' && op != ')'; operand=parse_expr >] ->
          Ast.Unary (op, operand)

      (* If the current token is not an operator, it must be a primary expr. *)
      | [< stream >] -> parse_primary stream

The grammar we add is pretty straightforward here. If we see a unary
operator when parsing a primary operator, we eat the operator as a
prefix and parse the remaining piece as another unary operator. This
allows us to handle multiple unary operators (e.g. "!!x"). Note that
unary operators can't have ambiguous parses like binary operators can,
so there is no need for precedence information.

The problem with this function, is that we need to call ParseUnary from
somewhere. To do this, we change previous callers of ParsePrimary to
call ``parse_unary`` instead:

.. code-block:: ocaml

    (* binoprhs
     *   ::= ('+' primary)* *)
    and parse_bin_rhs expr_prec lhs stream =
            ...
            (* Parse the unary expression after the binary operator. *)
            let rhs = parse_unary stream in
            ...

    ...

    (* expression
     *   ::= primary binoprhs *)
    and parse_expr = parser
      | [< lhs=parse_unary; stream >] -> parse_bin_rhs 0 lhs stream

With these two simple changes, we are now able to parse unary operators
and build the AST for them. Next up, we need to add parser support for
prototypes, to parse the unary operator prototype. We extend the binary
operator code above with:

.. code-block:: ocaml

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

As with binary operators, we name unary operators with a name that
includes the operator character. This assists us at code generation
time. Speaking of, the final piece we need to add is codegen support for
unary operators. It looks like this:

.. code-block:: ocaml

    let rec codegen_expr = function
      ...
      | Ast.Unary (op, operand) ->
          let operand = codegen_expr operand in
          let callee = "unary" ^ (String.make 1 op) in
          let callee =
            match lookup_function callee the_module with
            | Some callee -> callee
            | None -> raise (Error "unknown unary operator")
          in
          build_call callee [|operand|] "unop" builder

This code is similar to, but simpler than, the code for binary
operators. It is simpler primarily because it doesn't need to handle any
predefined operators.

Kicking the Tires
=================

It is somewhat hard to believe, but with a few simple extensions we've
covered in the last chapters, we have grown a real-ish language. With
this, we can do a lot of interesting things, including I/O, math, and a
bunch of other things. For example, we can now add a nice sequencing
operator (printd is defined to print out the specified value and a
newline):

::

    ready> extern printd(x);
    Read extern: declare double @printd(double)
    ready> def binary : 1 (x y) 0;  # Low-precedence operator that ignores operands.
    ..
    ready> printd(123) : printd(456) : printd(789);
    123.000000
    456.000000
    789.000000
    Evaluated to 0.000000

We can also define a bunch of other "primitive" operations, such as:

::

    # Logical unary not.
    def unary!(v)
      if v then
        0
      else
        1;

    # Unary negate.
    def unary-(v)
      0-v;

    # Define > with the same precedence as <.
    def binary> 10 (LHS RHS)
      RHS < LHS;

    # Binary logical or, which does not short circuit.
    def binary| 5 (LHS RHS)
      if LHS then
        1
      else if RHS then
        1
      else
        0;

    # Binary logical and, which does not short circuit.
    def binary& 6 (LHS RHS)
      if !LHS then
        0
      else
        !!RHS;

    # Define = with slightly lower precedence than relationals.
    def binary = 9 (LHS RHS)
      !(LHS < RHS | LHS > RHS);

Given the previous if/then/else support, we can also define interesting
functions for I/O. For example, the following prints out a character
whose "density" reflects the value passed in: the lower the value, the
denser the character:

::

    ready>

    extern putchard(char)
    def printdensity(d)
      if d > 8 then
        putchard(32)  # ' '
      else if d > 4 then
        putchard(46)  # '.'
      else if d > 2 then
        putchard(43)  # '+'
      else
        putchard(42); # '*'
    ...
    ready> printdensity(1): printdensity(2): printdensity(3) :
              printdensity(4): printdensity(5): printdensity(9): putchard(10);
    *++..
    Evaluated to 0.000000

Based on these simple primitive operations, we can start to define more
interesting things. For example, here's a little function that solves
for the number of iterations it takes a function in the complex plane to
converge:

::

    # determine whether the specific location diverges.
    # Solve for z = z^2 + c in the complex plane.
    def mandelconverger(real imag iters creal cimag)
      if iters > 255 | (real*real + imag*imag > 4) then
        iters
      else
        mandelconverger(real*real - imag*imag + creal,
                        2*real*imag + cimag,
                        iters+1, creal, cimag);

    # return the number of iterations required for the iteration to escape
    def mandelconverge(real imag)
      mandelconverger(real, imag, 0, real, imag);

This "z = z\ :sup:`2`\  + c" function is a beautiful little creature
that is the basis for computation of the `Mandelbrot
Set <http://en.wikipedia.org/wiki/Mandelbrot_set>`_. Our
``mandelconverge`` function returns the number of iterations that it
takes for a complex orbit to escape, saturating to 255. This is not a
very useful function by itself, but if you plot its value over a
two-dimensional plane, you can see the Mandelbrot set. Given that we are
limited to using putchard here, our amazing graphical output is limited,
but we can whip together something using the density plotter above:

::

    # compute and plot the mandelbrot set with the specified 2 dimensional range
    # info.
    def mandelhelp(xmin xmax xstep   ymin ymax ystep)
      for y = ymin, y < ymax, ystep in (
        (for x = xmin, x < xmax, xstep in
           printdensity(mandelconverge(x,y)))
        : putchard(10)
      )

    # mandel - This is a convenient helper function for plotting the mandelbrot set
    # from the specified position with the specified Magnification.
    def mandel(realstart imagstart realmag imagmag)
      mandelhelp(realstart, realstart+realmag*78, realmag,
                 imagstart, imagstart+imagmag*40, imagmag);

Given this, we can try plotting out the mandelbrot set! Lets try it out:

::

    ready> mandel(-2.3, -1.3, 0.05, 0.07);
    *******************************+++++++++++*************************************
    *************************+++++++++++++++++++++++*******************************
    **********************+++++++++++++++++++++++++++++****************************
    *******************+++++++++++++++++++++.. ...++++++++*************************
    *****************++++++++++++++++++++++.... ...+++++++++***********************
    ***************+++++++++++++++++++++++.....   ...+++++++++*********************
    **************+++++++++++++++++++++++....     ....+++++++++********************
    *************++++++++++++++++++++++......      .....++++++++*******************
    ************+++++++++++++++++++++.......       .......+++++++******************
    ***********+++++++++++++++++++....                ... .+++++++*****************
    **********+++++++++++++++++.......                     .+++++++****************
    *********++++++++++++++...........                    ...+++++++***************
    ********++++++++++++............                      ...++++++++**************
    ********++++++++++... ..........                        .++++++++**************
    *******+++++++++.....                                   .+++++++++*************
    *******++++++++......                                  ..+++++++++*************
    *******++++++.......                                   ..+++++++++*************
    *******+++++......                                     ..+++++++++*************
    *******.... ....                                      ...+++++++++*************
    *******.... .                                         ...+++++++++*************
    *******+++++......                                    ...+++++++++*************
    *******++++++.......                                   ..+++++++++*************
    *******++++++++......                                   .+++++++++*************
    *******+++++++++.....                                  ..+++++++++*************
    ********++++++++++... ..........                        .++++++++**************
    ********++++++++++++............                      ...++++++++**************
    *********++++++++++++++..........                     ...+++++++***************
    **********++++++++++++++++........                     .+++++++****************
    **********++++++++++++++++++++....                ... ..+++++++****************
    ***********++++++++++++++++++++++.......       .......++++++++*****************
    ************+++++++++++++++++++++++......      ......++++++++******************
    **************+++++++++++++++++++++++....      ....++++++++********************
    ***************+++++++++++++++++++++++.....   ...+++++++++*********************
    *****************++++++++++++++++++++++....  ...++++++++***********************
    *******************+++++++++++++++++++++......++++++++*************************
    *********************++++++++++++++++++++++.++++++++***************************
    *************************+++++++++++++++++++++++*******************************
    ******************************+++++++++++++************************************
    *******************************************************************************
    *******************************************************************************
    *******************************************************************************
    Evaluated to 0.000000
    ready> mandel(-2, -1, 0.02, 0.04);
    **************************+++++++++++++++++++++++++++++++++++++++++++++++++++++
    ***********************++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    *********************+++++++++++++++++++++++++++++++++++++++++++++++++++++++++.
    *******************+++++++++++++++++++++++++++++++++++++++++++++++++++++++++...
    *****************+++++++++++++++++++++++++++++++++++++++++++++++++++++++++.....
    ***************++++++++++++++++++++++++++++++++++++++++++++++++++++++++........
    **************++++++++++++++++++++++++++++++++++++++++++++++++++++++...........
    ************+++++++++++++++++++++++++++++++++++++++++++++++++++++..............
    ***********++++++++++++++++++++++++++++++++++++++++++++++++++........        .
    **********++++++++++++++++++++++++++++++++++++++++++++++.............
    ********+++++++++++++++++++++++++++++++++++++++++++..................
    *******+++++++++++++++++++++++++++++++++++++++.......................
    ******+++++++++++++++++++++++++++++++++++...........................
    *****++++++++++++++++++++++++++++++++............................
    *****++++++++++++++++++++++++++++...............................
    ****++++++++++++++++++++++++++......   .........................
    ***++++++++++++++++++++++++.........     ......    ...........
    ***++++++++++++++++++++++............
    **+++++++++++++++++++++..............
    **+++++++++++++++++++................
    *++++++++++++++++++.................
    *++++++++++++++++............ ...
    *++++++++++++++..............
    *+++....++++................
    *..........  ...........
    *
    *..........  ...........
    *+++....++++................
    *++++++++++++++..............
    *++++++++++++++++............ ...
    *++++++++++++++++++.................
    **+++++++++++++++++++................
    **+++++++++++++++++++++..............
    ***++++++++++++++++++++++............
    ***++++++++++++++++++++++++.........     ......    ...........
    ****++++++++++++++++++++++++++......   .........................
    *****++++++++++++++++++++++++++++...............................
    *****++++++++++++++++++++++++++++++++............................
    ******+++++++++++++++++++++++++++++++++++...........................
    *******+++++++++++++++++++++++++++++++++++++++.......................
    ********+++++++++++++++++++++++++++++++++++++++++++..................
    Evaluated to 0.000000
    ready> mandel(-0.9, -1.4, 0.02, 0.03);
    *******************************************************************************
    *******************************************************************************
    *******************************************************************************
    **********+++++++++++++++++++++************************************************
    *+++++++++++++++++++++++++++++++++++++++***************************************
    +++++++++++++++++++++++++++++++++++++++++++++**********************************
    ++++++++++++++++++++++++++++++++++++++++++++++++++*****************************
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++*************************
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++**********************
    +++++++++++++++++++++++++++++++++.........++++++++++++++++++*******************
    +++++++++++++++++++++++++++++++....   ......+++++++++++++++++++****************
    +++++++++++++++++++++++++++++.......  ........+++++++++++++++++++**************
    ++++++++++++++++++++++++++++........   ........++++++++++++++++++++************
    +++++++++++++++++++++++++++.........     ..  ...+++++++++++++++++++++**********
    ++++++++++++++++++++++++++...........        ....++++++++++++++++++++++********
    ++++++++++++++++++++++++.............       .......++++++++++++++++++++++******
    +++++++++++++++++++++++.............        ........+++++++++++++++++++++++****
    ++++++++++++++++++++++...........           ..........++++++++++++++++++++++***
    ++++++++++++++++++++...........                .........++++++++++++++++++++++*
    ++++++++++++++++++............                  ...........++++++++++++++++++++
    ++++++++++++++++...............                 .............++++++++++++++++++
    ++++++++++++++.................                 ...............++++++++++++++++
    ++++++++++++..................                  .................++++++++++++++
    +++++++++..................                      .................+++++++++++++
    ++++++........        .                               .........  ..++++++++++++
    ++............                                         ......    ....++++++++++
    ..............                                                    ...++++++++++
    ..............                                                    ....+++++++++
    ..............                                                    .....++++++++
    .............                                                    ......++++++++
    ...........                                                     .......++++++++
    .........                                                       ........+++++++
    .........                                                       ........+++++++
    .........                                                           ....+++++++
    ........                                                             ...+++++++
    .......                                                              ...+++++++
                                                                        ....+++++++
                                                                       .....+++++++
                                                                        ....+++++++
                                                                        ....+++++++
                                                                        ....+++++++
    Evaluated to 0.000000
    ready> ^D

At this point, you may be starting to realize that Kaleidoscope is a
real and powerful language. It may not be self-similar :), but it can be
used to plot things that are!

With this, we conclude the "adding user-defined operators" chapter of
the tutorial. We have successfully augmented our language, adding the
ability to extend the language in the library, and we have shown how
this can be used to build a simple but interesting end-user application
in Kaleidoscope. At this point, Kaleidoscope can build a variety of
applications that are functional and can call functions with
side-effects, but it can't actually define and mutate a variable itself.

Strikingly, variable mutation is an important feature of some languages,
and it is not at all obvious how to `add support for mutable
variables <OCamlLangImpl7.html>`_ without having to add an "SSA
construction" phase to your front-end. In the next chapter, we will
describe how you can add variable mutation without building SSA in your
front-end.

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

                (* Parse the unary expression after the binary operator. *)
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

        let rec codegen_expr = function
          | Ast.Number n -> const_float double_type n
          | Ast.Variable name ->
              (try Hashtbl.find named_values name with
                | Not_found -> raise (Error "unknown variable name"))
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

        /* printd - printf that takes a double prints it as "%f\n", returning 0. */
        extern double printd(double X) {
          printf("%f\n", X);
          return 0;
        }

`Next: Extending the language: mutable variables / SSA
construction <OCamlLangImpl7.html>`_

