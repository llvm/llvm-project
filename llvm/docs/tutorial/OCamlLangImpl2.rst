===========================================
Kaleidoscope: Implementing a Parser and AST
===========================================

.. contents::
   :local:

Chapter 2 Introduction
======================

Welcome to Chapter 2 of the "`Implementing a language with LLVM in
Objective Caml <index.html>`_" tutorial. This chapter shows you how to
use the lexer, built in `Chapter 1 <OCamlLangImpl1.html>`_, to build a
full `parser <http://en.wikipedia.org/wiki/Parsing>`_ for our
Kaleidoscope language. Once we have a parser, we'll define and build an
`Abstract Syntax
Tree <http://en.wikipedia.org/wiki/Abstract_syntax_tree>`_ (AST).

The parser we will build uses a combination of `Recursive Descent
Parsing <http://en.wikipedia.org/wiki/Recursive_descent_parser>`_ and
`Operator-Precedence
Parsing <http://en.wikipedia.org/wiki/Operator-precedence_parser>`_ to
parse the Kaleidoscope language (the latter for binary expressions and
the former for everything else). Before we get to parsing though, lets
talk about the output of the parser: the Abstract Syntax Tree.

The Abstract Syntax Tree (AST)
==============================

The AST for a program captures its behavior in such a way that it is
easy for later stages of the compiler (e.g. code generation) to
interpret. We basically want one object for each construct in the
language, and the AST should closely model the language. In
Kaleidoscope, we have expressions, a prototype, and a function object.
We'll start with expressions first:

.. code-block:: ocaml

    (* expr - Base type for all expression nodes. *)
    type expr =
      (* variant for numeric literals like "1.0". *)
      | Number of float

The code above shows the definition of the base ExprAST class and one
subclass which we use for numeric literals. The important thing to note
about this code is that the Number variant captures the numeric value of
the literal as an instance variable. This allows later phases of the
compiler to know what the stored numeric value is.

Right now we only create the AST, so there are no useful functions on
them. It would be very easy to add a function to pretty print the code,
for example. Here are the other expression AST node definitions that
we'll use in the basic form of the Kaleidoscope language:

.. code-block:: ocaml

      (* variant for referencing a variable, like "a". *)
      | Variable of string

      (* variant for a binary operator. *)
      | Binary of char * expr * expr

      (* variant for function calls. *)
      | Call of string * expr array

This is all (intentionally) rather straight-forward: variables capture
the variable name, binary operators capture their opcode (e.g. '+'), and
calls capture a function name as well as a list of any argument
expressions. One thing that is nice about our AST is that it captures
the language features without talking about the syntax of the language.
Note that there is no discussion about precedence of binary operators,
lexical structure, etc.

For our basic language, these are all of the expression nodes we'll
define. Because it doesn't have conditional control flow, it isn't
Turing-complete; we'll fix that in a later installment. The two things
we need next are a way to talk about the interface to a function, and a
way to talk about functions themselves:

.. code-block:: ocaml

    (* proto - This type represents the "prototype" for a function, which captures
     * its name, and its argument names (thus implicitly the number of arguments the
     * function takes). *)
    type proto = Prototype of string * string array

    (* func - This type represents a function definition itself. *)
    type func = Function of proto * expr

In Kaleidoscope, functions are typed with just a count of their
arguments. Since all values are double precision floating point, the
type of each argument doesn't need to be stored anywhere. In a more
aggressive and realistic language, the "expr" variants would probably
have a type field.

With this scaffolding, we can now talk about parsing expressions and
function bodies in Kaleidoscope.

Parser Basics
=============

Now that we have an AST to build, we need to define the parser code to
build it. The idea here is that we want to parse something like "x+y"
(which is returned as three tokens by the lexer) into an AST that could
be generated with calls like this:

.. code-block:: ocaml

      let x = Variable "x" in
      let y = Variable "y" in
      let result = Binary ('+', x, y) in
      ...

The error handling routines make use of the builtin ``Stream.Failure``
and ``Stream.Error``s. ``Stream.Failure`` is raised when the parser is
unable to find any matching token in the first position of a pattern.
``Stream.Error`` is raised when the first token matches, but the rest do
not. The error recovery in our parser will not be the best and is not
particular user-friendly, but it will be enough for our tutorial. These
exceptions make it easier to handle errors in routines that have various
return types.

With these basic types and exceptions, we can implement the first piece
of our grammar: numeric literals.

Basic Expression Parsing
========================

We start with numeric literals, because they are the simplest to
process. For each production in our grammar, we'll define a function
which parses that production. We call this class of expressions
"primary" expressions, for reasons that will become more clear `later in
the tutorial <OCamlLangImpl6.html#unary>`_. In order to parse an
arbitrary primary expression, we need to determine what sort of
expression it is. For numeric literals, we have:

.. code-block:: ocaml

    (* primary
     *   ::= identifier
     *   ::= numberexpr
     *   ::= parenexpr *)
    parse_primary = parser
      (* numberexpr ::= number *)
      | [< 'Token.Number n >] -> Ast.Number n

This routine is very simple: it expects to be called when the current
token is a ``Token.Number`` token. It takes the current number value,
creates a ``Ast.Number`` node, advances the lexer to the next token, and
finally returns.

There are some interesting aspects to this. The most important one is
that this routine eats all of the tokens that correspond to the
production and returns the lexer buffer with the next token (which is
not part of the grammar production) ready to go. This is a fairly
standard way to go for recursive descent parsers. For a better example,
the parenthesis operator is defined like this:

.. code-block:: ocaml

      (* parenexpr ::= '(' expression ')' *)
      | [< 'Token.Kwd '('; e=parse_expr; 'Token.Kwd ')' ?? "expected ')'" >] -> e

This function illustrates a number of interesting things about the
parser:

1) It shows how we use the ``Stream.Error`` exception. When called, this
function expects that the current token is a '(' token, but after
parsing the subexpression, it is possible that there is no ')' waiting.
For example, if the user types in "(4 x" instead of "(4)", the parser
should emit an error. Because errors can occur, the parser needs a way
to indicate that they happened. In our parser, we use the camlp4
shortcut syntax ``token ?? "parse error"``, where if the token before
the ``??`` does not match, then ``Stream.Error "parse error"`` will be
raised.

2) Another interesting aspect of this function is that it uses recursion
by calling ``Parser.parse_primary`` (we will soon see that
``Parser.parse_primary`` can call ``Parser.parse_primary``). This is
powerful because it allows us to handle recursive grammars, and keeps
each production very simple. Note that parentheses do not cause
construction of AST nodes themselves. While we could do it this way, the
most important role of parentheses are to guide the parser and provide
grouping. Once the parser constructs the AST, parentheses are not
needed.

The next simple production is for handling variable references and
function calls:

.. code-block:: ocaml

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

This routine follows the same style as the other routines. (It expects
to be called if the current token is a ``Token.Ident`` token). It also
has recursion and error handling. One interesting aspect of this is that
it uses *look-ahead* to determine if the current identifier is a stand
alone variable reference or if it is a function call expression. It
handles this by checking to see if the token after the identifier is a
'(' token, constructing either a ``Ast.Variable`` or ``Ast.Call`` node
as appropriate.

We finish up by raising an exception if we received a token we didn't
expect:

.. code-block:: ocaml

      | [< >] -> raise (Stream.Error "unknown token when expecting an expression.")

Now that basic expressions are handled, we need to handle binary
expressions. They are a bit more complex.

Binary Expression Parsing
=========================

Binary expressions are significantly harder to parse because they are
often ambiguous. For example, when given the string "x+y\*z", the parser
can choose to parse it as either "(x+y)\*z" or "x+(y\*z)". With common
definitions from mathematics, we expect the later parse, because "\*"
(multiplication) has higher *precedence* than "+" (addition).

There are many ways to handle this, but an elegant and efficient way is
to use `Operator-Precedence
Parsing <http://en.wikipedia.org/wiki/Operator-precedence_parser>`_.
This parsing technique uses the precedence of binary operators to guide
recursion. To start with, we need a table of precedences:

.. code-block:: ocaml

    (* binop_precedence - This holds the precedence for each binary operator that is
     * defined *)
    let binop_precedence:(char, int) Hashtbl.t = Hashtbl.create 10

    (* precedence - Get the precedence of the pending binary operator token. *)
    let precedence c = try Hashtbl.find binop_precedence c with Not_found -> -1

    ...

    let main () =
      (* Install standard binary operators.
       * 1 is the lowest precedence. *)
      Hashtbl.add Parser.binop_precedence '<' 10;
      Hashtbl.add Parser.binop_precedence '+' 20;
      Hashtbl.add Parser.binop_precedence '-' 20;
      Hashtbl.add Parser.binop_precedence '*' 40;    (* highest. *)
      ...

For the basic form of Kaleidoscope, we will only support 4 binary
operators (this can obviously be extended by you, our brave and intrepid
reader). The ``Parser.precedence`` function returns the precedence for
the current token, or -1 if the token is not a binary operator. Having a
``Hashtbl.t`` makes it easy to add new operators and makes it clear that
the algorithm doesn't depend on the specific operators involved, but it
would be easy enough to eliminate the ``Hashtbl.t`` and do the
comparisons in the ``Parser.precedence`` function. (Or just use a
fixed-size array).

With the helper above defined, we can now start parsing binary
expressions. The basic idea of operator precedence parsing is to break
down an expression with potentially ambiguous binary operators into
pieces. Consider, for example, the expression "a+b+(c+d)\*e\*f+g".
Operator precedence parsing considers this as a stream of primary
expressions separated by binary operators. As such, it will first parse
the leading primary expression "a", then it will see the pairs [+, b]
[+, (c+d)] [\*, e] [\*, f] and [+, g]. Note that because parentheses are
primary expressions, the binary expression parser doesn't need to worry
about nested subexpressions like (c+d) at all.

To start, an expression is a primary expression potentially followed by
a sequence of [binop,primaryexpr] pairs:

.. code-block:: ocaml

    (* expression
     *   ::= primary binoprhs *)
    and parse_expr = parser
      | [< lhs=parse_primary; stream >] -> parse_bin_rhs 0 lhs stream

``Parser.parse_bin_rhs`` is the function that parses the sequence of
pairs for us. It takes a precedence and a pointer to an expression for
the part that has been parsed so far. Note that "x" is a perfectly valid
expression: As such, "binoprhs" is allowed to be empty, in which case it
returns the expression that is passed into it. In our example above, the
code passes the expression for "a" into ``Parser.parse_bin_rhs`` and the
current token is "+".

The precedence value passed into ``Parser.parse_bin_rhs`` indicates the
*minimal operator precedence* that the function is allowed to eat. For
example, if the current pair stream is [+, x] and
``Parser.parse_bin_rhs`` is passed in a precedence of 40, it will not
consume any tokens (because the precedence of '+' is only 20). With this
in mind, ``Parser.parse_bin_rhs`` starts with:

.. code-block:: ocaml

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

This code gets the precedence of the current token and checks to see if
if is too low. Because we defined invalid tokens to have a precedence of
-1, this check implicitly knows that the pair-stream ends when the token
stream runs out of binary operators. If this check succeeds, we know
that the token is a binary operator and that it will be included in this
expression:

.. code-block:: ocaml

            (* Eat the binop. *)
            Stream.junk stream;

            (* Parse the primary expression after the binary operator *)
            let rhs = parse_primary stream in

            (* Okay, we know this is a binop. *)
            let rhs =
              match Stream.peek stream with
              | Some (Token.Kwd c2) ->

As such, this code eats (and remembers) the binary operator and then
parses the primary expression that follows. This builds up the whole
pair, the first of which is [+, b] for the running example.

Now that we parsed the left-hand side of an expression and one pair of
the RHS sequence, we have to decide which way the expression associates.
In particular, we could have "(a+b) binop unparsed" or "a + (b binop
unparsed)". To determine this, we look ahead at "binop" to determine its
precedence and compare it to BinOp's precedence (which is '+' in this
case):

.. code-block:: ocaml

                  (* If BinOp binds less tightly with rhs than the operator after
                   * rhs, let the pending operator take rhs as its lhs. *)
                  let next_prec = precedence c2 in
                  if token_prec < next_prec

If the precedence of the binop to the right of "RHS" is lower or equal
to the precedence of our current operator, then we know that the
parentheses associate as "(a+b) binop ...". In our example, the current
operator is "+" and the next operator is "+", we know that they have the
same precedence. In this case we'll create the AST node for "a+b", and
then continue parsing:

.. code-block:: ocaml

              ... if body omitted ...
            in

            (* Merge lhs/rhs. *)
            let lhs = Ast.Binary (c, lhs, rhs) in
            parse_bin_rhs expr_prec lhs stream
          end

In our example above, this will turn "a+b+" into "(a+b)" and execute the
next iteration of the loop, with "+" as the current token. The code
above will eat, remember, and parse "(c+d)" as the primary expression,
which makes the current pair equal to [+, (c+d)]. It will then evaluate
the 'if' conditional above with "\*" as the binop to the right of the
primary. In this case, the precedence of "\*" is higher than the
precedence of "+" so the if condition will be entered.

The critical question left here is "how can the if condition parse the
right hand side in full"? In particular, to build the AST correctly for
our example, it needs to get all of "(c+d)\*e\*f" as the RHS expression
variable. The code to do this is surprisingly simple (code from the
above two blocks duplicated for context):

.. code-block:: ocaml

              match Stream.peek stream with
              | Some (Token.Kwd c2) ->
                  (* If BinOp binds less tightly with rhs than the operator after
                   * rhs, let the pending operator take rhs as its lhs. *)
                  if token_prec < precedence c2
                  then parse_bin_rhs (token_prec + 1) rhs stream
                  else rhs
              | _ -> rhs
            in

            (* Merge lhs/rhs. *)
            let lhs = Ast.Binary (c, lhs, rhs) in
            parse_bin_rhs expr_prec lhs stream
          end

At this point, we know that the binary operator to the RHS of our
primary has higher precedence than the binop we are currently parsing.
As such, we know that any sequence of pairs whose operators are all
higher precedence than "+" should be parsed together and returned as
"RHS". To do this, we recursively invoke the ``Parser.parse_bin_rhs``
function specifying "token\_prec+1" as the minimum precedence required
for it to continue. In our example above, this will cause it to return
the AST node for "(c+d)\*e\*f" as RHS, which is then set as the RHS of
the '+' expression.

Finally, on the next iteration of the while loop, the "+g" piece is
parsed and added to the AST. With this little bit of code (14
non-trivial lines), we correctly handle fully general binary expression
parsing in a very elegant way. This was a whirlwind tour of this code,
and it is somewhat subtle. I recommend running through it with a few
tough examples to see how it works.

This wraps up handling of expressions. At this point, we can point the
parser at an arbitrary token stream and build an expression from it,
stopping at the first token that is not part of the expression. Next up
we need to handle function definitions, etc.

Parsing the Rest
================

The next thing missing is handling of function prototypes. In
Kaleidoscope, these are used both for 'extern' function declarations as
well as function body definitions. The code to do this is
straight-forward and not very interesting (once you've survived
expressions):

.. code-block:: ocaml

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

Given this, a function definition is very simple, just a prototype plus
an expression to implement the body:

.. code-block:: ocaml

    (* definition ::= 'def' prototype expression *)
    let parse_definition = parser
      | [< 'Token.Def; p=parse_prototype; e=parse_expr >] ->
          Ast.Function (p, e)

In addition, we support 'extern' to declare functions like 'sin' and
'cos' as well as to support forward declaration of user functions. These
'extern's are just prototypes with no body:

.. code-block:: ocaml

    (*  external ::= 'extern' prototype *)
    let parse_extern = parser
      | [< 'Token.Extern; e=parse_prototype >] -> e

Finally, we'll also let the user type in arbitrary top-level expressions
and evaluate them on the fly. We will handle this by defining anonymous
nullary (zero argument) functions for them:

.. code-block:: ocaml

    (* toplevelexpr ::= expression *)
    let parse_toplevel = parser
      | [< e=parse_expr >] ->
          (* Make an anonymous proto. *)
          Ast.Function (Ast.Prototype ("", [||]), e)

Now that we have all the pieces, let's build a little driver that will
let us actually *execute* this code we've built!

The Driver
==========

The driver for this simply invokes all of the parsing pieces with a
top-level dispatch loop. There isn't much interesting here, so I'll just
include the top-level loop. See `below <#code>`_ for full code in the
"Top-Level Parsing" section.

.. code-block:: ocaml

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
                ignore(Parser.parse_definition stream);
                print_endline "parsed a function definition.";
            | Token.Extern ->
                ignore(Parser.parse_extern stream);
                print_endline "parsed an extern.";
            | _ ->
                (* Evaluate a top-level expression into an anonymous function. *)
                ignore(Parser.parse_toplevel stream);
                print_endline "parsed a top-level expr";
            with Stream.Error s ->
              (* Skip token for error recovery. *)
              Stream.junk stream;
              print_endline s;
          end;
          print_string "ready> "; flush stdout;
          main_loop stream

The most interesting part of this is that we ignore top-level
semicolons. Why is this, you ask? The basic reason is that if you type
"4 + 5" at the command line, the parser doesn't know whether that is the
end of what you will type or not. For example, on the next line you
could type "def foo..." in which case 4+5 is the end of a top-level
expression. Alternatively you could type "\* 6", which would continue
the expression. Having top-level semicolons allows you to type "4+5;",
and the parser will know you are done.

Conclusions
===========

With just under 300 lines of commented code (240 lines of non-comment,
non-blank code), we fully defined our minimal language, including a
lexer, parser, and AST builder. With this done, the executable will
validate Kaleidoscope code and tell us if it is grammatically invalid.
For example, here is a sample interaction:

.. code-block:: bash

    $ ./toy.byte
    ready> def foo(x y) x+foo(y, 4.0);
    Parsed a function definition.
    ready> def foo(x y) x+y y;
    Parsed a function definition.
    Parsed a top-level expr
    ready> def foo(x y) x+y );
    Parsed a function definition.
    Error: unknown token when expecting an expression
    ready> extern sin(a);
    ready> Parsed an extern
    ready> ^D
    $

There is a lot of room for extension here. You can define new AST nodes,
extend the language in many ways, etc. In the `next
installment <OCamlLangImpl3.html>`_, we will describe how to generate
LLVM Intermediate Representation (IR) from the AST.

Full Code Listing
=================

Here is the complete code listing for this and the previous chapter.
Note that it is fully self-contained: you don't need LLVM or any
external libraries at all for this. (Besides the ocaml standard
libraries, of course.) To build this, just compile with:

.. code-block:: bash

    # Compile
    ocamlbuild toy.byte
    # Run
    ./toy.byte

Here is the code:

\_tags:
    ::

        <{lexer,parser}.ml>: use_camlp4, pp(camlp4of)

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

toplevel.ml:
    .. code-block:: ocaml

        (*===----------------------------------------------------------------------===
         * Top-Level parsing and JIT Driver
         *===----------------------------------------------------------------------===*)

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
                    ignore(Parser.parse_definition stream);
                    print_endline "parsed a function definition.";
                | Token.Extern ->
                    ignore(Parser.parse_extern stream);
                    print_endline "parsed an extern.";
                | _ ->
                    (* Evaluate a top-level expression into an anonymous function. *)
                    ignore(Parser.parse_toplevel stream);
                    print_endline "parsed a top-level expr";
                with Stream.Error s ->
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
        ;;

        main ()

`Next: Implementing Code Generation to LLVM IR <OCamlLangImpl3.html>`_

