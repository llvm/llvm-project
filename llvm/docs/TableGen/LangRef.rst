===========================
TableGen Language Reference
===========================

.. contents::
   :local:

.. warning::
   This document is extremely rough. If you find something lacking, please
   fix it, file a documentation bug, or ask about it on llvm-dev.

Introduction
============

This document is meant to be a normative spec about the TableGen language
in and of itself (i.e. how to understand a given construct in terms of how
it affects the final set of records represented by the TableGen file). If
you are unsure if this document is really what you are looking for, please
read the :doc:`introduction to TableGen <index>` first.

Notation
========

The lexical and syntax notation used here is intended to imitate
`Python's`_. In particular, for lexical definitions, the productions
operate at the character level and there is no implied whitespace between
elements. The syntax definitions operate at the token level, so there is
implied whitespace between tokens.

.. _`Python's`: http://docs.python.org/py3k/reference/introduction.html#notation

Lexical Analysis
================

TableGen supports BCPL (``// ...``) and nestable C-style (``/* ... */``)
comments.  TableGen also provides simple `Preprocessing Support`_.

The following is a listing of the basic punctuation tokens::

   - + [ ] { } ( ) < > : ; .  = ? #

Numeric literals take one of the following forms:

.. TableGen actually will lex some pretty strange sequences an interpret
   them as numbers. What is shown here is an attempt to approximate what it
   "should" accept.

.. productionlist::
   TokInteger: `DecimalInteger` | `HexInteger` | `BinInteger`
   DecimalInteger: ["+" | "-"] ("0"..."9")+
   HexInteger: "0x" ("0"..."9" | "a"..."f" | "A"..."F")+
   BinInteger: "0b" ("0" | "1")+

One aspect to note is that the :token:`DecimalInteger` token *includes* the
``+`` or ``-``, as opposed to having ``+`` and ``-`` be unary operators as
most languages do.

Also note that :token:`BinInteger` creates a value of type ``bits<n>``
(where ``n`` is the number of bits).  This will implicitly convert to
integers when needed.

TableGen has identifier-like tokens:

.. productionlist::
   ualpha: "a"..."z" | "A"..."Z" | "_"
   TokIdentifier: ("0"..."9")* `ualpha` (`ualpha` | "0"..."9")*
   TokVarName: "$" `ualpha` (`ualpha` |  "0"..."9")*

Note that unlike most languages, TableGen allows :token:`TokIdentifier` to
begin with a number. In case of ambiguity, a token will be interpreted as a
numeric literal rather than an identifier.

TableGen also has two string-like literals:

.. productionlist::
   TokString: '"' <non-'"' characters and C-like escapes> '"'
   TokCodeFragment: "[{" <shortest text not containing "}]"> "}]"

:token:`TokCodeFragment` is essentially a multiline string literal
delimited by ``[{`` and ``}]``.

.. note::
   The current implementation accepts the following C-like escapes::

      \\ \' \" \t \n

TableGen also has the following keywords::

   bit   bits      class   code         dag
   def   foreach   defm    field        in
   int   let       list    multiclass   string

TableGen also has "bang operators" which have a
wide variety of meanings:

.. productionlist::
   BangOperator: one of
               :!eq     !if      !head    !tail      !con
               :!add    !shl     !sra     !srl       !and
               :!or     !empty   !subst   !foreach   !strconcat
               :!cast   !listconcat       !size      !foldl
               :!isa    !dag     !le      !lt        !ge
               :!gt     !ne      !mul

TableGen also has !cond operator that needs a slightly different
syntax compared to other "bang operators":

.. productionlist::
   CondOperator: !cond


Syntax
======

TableGen has an ``include`` mechanism. It does not play a role in the
syntax per se, since it is lexically replaced with the contents of the
included file.

.. productionlist::
   IncludeDirective: "include" `TokString`

TableGen's top-level production consists of "objects".

.. productionlist::
   TableGenFile: `Object`*
   Object: `Class` | `Def` | `Defm` | `Defset` | `Let` | `MultiClass` |
           `Foreach`

``class``\es
------------

.. productionlist::
   Class: "class" `TokIdentifier` [`TemplateArgList`] `ObjectBody`
   TemplateArgList: "<" `Declaration` ("," `Declaration`)* ">"

A ``class`` declaration creates a record which other records can inherit
from. A class can be parametrized by a list of "template arguments", whose
values can be used in the class body.

A given class can only be defined once. A ``class`` declaration is
considered to define the class if any of the following is true:

.. break ObjectBody into its consituents so that they are present here?

#. The :token:`TemplateArgList` is present.
#. The :token:`Body` in the :token:`ObjectBody` is present and is not empty.
#. The :token:`BaseClassList` in the :token:`ObjectBody` is present.

You can declare an empty class by giving an empty :token:`TemplateArgList`
and an empty :token:`ObjectBody`. This can serve as a restricted form of
forward declaration: note that records deriving from the forward-declared
class will inherit no fields from it since the record expansion is done
when the record is parsed.

Every class has an implicit template argument called ``NAME``, which is set
to the name of the instantiating ``def`` or ``defm``. The result is undefined
if the class is instantiated by an anonymous record.

Declarations
------------

.. Omitting mention of arcane "field" prefix to discourage its use.

The declaration syntax is pretty much what you would expect as a C++
programmer.

.. productionlist::
   Declaration: `Type` `TokIdentifier` ["=" `Value`]

It assigns the value to the identifier.

Types
-----

.. productionlist::
   Type: "string" | "code" | "bit" | "int" | "dag"
       :| "bits" "<" `TokInteger` ">"
       :| "list" "<" `Type` ">"
       :| `ClassID`
   ClassID: `TokIdentifier`

Both ``string`` and ``code`` correspond to the string type; the difference
is purely to indicate programmer intention.

The :token:`ClassID` must identify a class that has been previously
declared or defined.

Values
------

.. productionlist::
   Value: `SimpleValue` `ValueSuffix`*
   ValueSuffix: "{" `RangeList` "}"
              :| "[" `RangeList` "]"
              :| "." `TokIdentifier`
   RangeList: `RangePiece` ("," `RangePiece`)*
   RangePiece: `TokInteger`
             :| `TokInteger` "-" `TokInteger`
             :| `TokInteger` `TokInteger`

The peculiar last form of :token:`RangePiece` is due to the fact that the
"``-``" is included in the :token:`TokInteger`, hence ``1-5`` gets lexed as
two consecutive :token:`TokInteger`'s, with values ``1`` and ``-5``,
instead of "1", "-", and "5".
The :token:`RangeList` can be thought of as specifying "list slice" in some
contexts.


:token:`SimpleValue` has a number of forms:


.. productionlist::
   SimpleValue: `TokIdentifier`

The value will be the variable referenced by the identifier. It can be one
of:

.. The code for this is exceptionally abstruse. These examples are a
   best-effort attempt.

* name of a ``def``, such as the use of ``Bar`` in::

     def Bar : SomeClass {
       int X = 5;
     }

     def Foo {
       SomeClass Baz = Bar;
     }

* value local to a ``def``, such as the use of ``Bar`` in::

     def Foo {
       int Bar = 5;
       int Baz = Bar;
     }

  Values defined in superclasses can be accessed the same way.

* a template arg of a ``class``, such as the use of ``Bar`` in::

     class Foo<int Bar> {
       int Baz = Bar;
     }

* value local to a ``class``, such as the use of ``Bar`` in::

     class Foo {
       int Bar = 5;
       int Baz = Bar;
     }

* a template arg to a ``multiclass``, such as the use of ``Bar`` in::

     multiclass Foo<int Bar> {
       def : SomeClass<Bar>;
     }

* the iteration variable of a ``foreach``, such as the use of ``i`` in::

     foreach i = 0-5 in
     def Foo#i;

* a variable defined by ``defset``

* the implicit template argument ``NAME`` in a ``class`` or ``multiclass``

.. productionlist::
   SimpleValue: `TokInteger`

This represents the numeric value of the integer.

.. productionlist::
   SimpleValue: `TokString`+

Multiple adjacent string literals are concatenated like in C/C++. The value
is the concatenation of the strings.

.. productionlist::
   SimpleValue: `TokCodeFragment`

The value is the string value of the code fragment.

.. productionlist::
   SimpleValue: "?"

``?`` represents an "unset" initializer.

.. productionlist::
   SimpleValue: "{" `ValueList` "}"
   ValueList: [`ValueListNE`]
   ValueListNE: `Value` ("," `Value`)*

This represents a sequence of bits, as would be used to initialize a
``bits<n>`` field (where ``n`` is the number of bits).

.. productionlist::
   SimpleValue: `ClassID` "<" `ValueListNE` ">"

This generates a new anonymous record definition (as would be created by an
unnamed ``def`` inheriting from the given class with the given template
arguments) and the value is the value of that record definition.

.. productionlist::
   SimpleValue: "[" `ValueList` "]" ["<" `Type` ">"]

A list initializer. The optional :token:`Type` can be used to indicate a
specific element type, otherwise the element type will be deduced from the
given values.

.. The initial `DagArg` of the dag must start with an identifier or
   !cast, but this is more of an implementation detail and so for now just
   leave it out.

.. productionlist::
   SimpleValue: "(" `DagArg` [`DagArgList`] ")"
   DagArgList: `DagArg` ("," `DagArg`)*
   DagArg: `Value` [":" `TokVarName`] | `TokVarName`

The initial :token:`DagArg` is called the "operator" of the dag.

.. productionlist::
   SimpleValue: `BangOperator` ["<" `Type` ">"] "(" `ValueListNE` ")"
              :| `CondOperator` "(" `CondVal` ("," `CondVal`)* ")"
   CondVal: `Value` ":" `Value`

Bodies
------

.. productionlist::
   ObjectBody: `BaseClassList` `Body`
   BaseClassList: [":" `BaseClassListNE`]
   BaseClassListNE: `SubClassRef` ("," `SubClassRef`)*
   SubClassRef: (`ClassID` | `MultiClassID`) ["<" `ValueList` ">"]
   DefmID: `TokIdentifier`

The version with the :token:`MultiClassID` is only valid in the
:token:`BaseClassList` of a ``defm``.
The :token:`MultiClassID` should be the name of a ``multiclass``.

.. put this somewhere else

It is after parsing the base class list that the "let stack" is applied.

.. productionlist::
   Body: ";" | "{" BodyList "}"
   BodyList: BodyItem*
   BodyItem: `Declaration` ";"
           :| "let" `TokIdentifier` [ "{" `RangeList` "}" ] "=" `Value` ";"

The ``let`` form allows overriding the value of an inherited field.

``def``
-------

.. productionlist::
   Def: "def" [`Value`] `ObjectBody`

Defines a record whose name is given by the optional :token:`Value`. The value
is parsed in a special mode where global identifiers (records and variables
defined by ``defset``) are not recognized, and all unrecognized identifiers
are interpreted as strings.

If no name is given, the record is anonymous. The final name of anonymous
records is undefined, but globally unique.

Special handling occurs if this ``def`` appears inside a ``multiclass`` or
a ``foreach``.

When a non-anonymous record is defined in a multiclass and the given name
does not contain a reference to the implicit template argument ``NAME``, such
a reference will automatically be prepended. That is, the following are
equivalent inside a multiclass::

    def Foo;
    def NAME#Foo;

``defm``
--------

.. productionlist::
   Defm: "defm" [`Value`] ":" `BaseClassListNE` ";"

The :token:`BaseClassList` is a list of at least one ``multiclass`` and any
number of ``class``'s. The ``multiclass``'s must occur before any ``class``'s.

Instantiates all records defined in all given ``multiclass``'s and adds the
given ``class``'s as superclasses.

The name is parsed in the same special mode used by ``def``. If the name is
missing, a globally unique string is used instead (but instantiated records
are not considered to be anonymous, unless they were originally defined by an
anonymous ``def``) That is, the following have different semantics::

    defm : SomeMultiClass<...>;    // some globally unique name
    defm "" : SomeMultiClass<...>; // empty name string

When it occurs inside a multiclass, the second variant is equivalent to
``defm NAME : ...``. More generally, when ``defm`` occurs in a multiclass and
its name does not contain a reference to the implicit template argument
``NAME``, such a reference will automatically be prepended. That is, the
following are equivalent inside a multiclass::

    defm Foo : SomeMultiClass<...>;
    defm NAME#Foo : SomeMultiClass<...>;

``defset``
----------
.. productionlist::
   Defset: "defset" `Type` `TokIdentifier` "=" "{" `Object`* "}"

All records defined inside the braces via ``def`` and ``defm`` are collected
in a globally accessible list of the given name (in addition to being added
to the global collection of records as usual). Anonymous records created inside
initializier expressions using the ``Class<args...>`` syntax are never collected
in a defset.

The given type must be ``list<A>``, where ``A`` is some class. It is an error
to define a record (via ``def`` or ``defm``) inside the braces which doesn't
derive from ``A``.

``foreach``
-----------

.. productionlist::
   Foreach: "foreach" `ForeachDeclaration` "in" "{" `Object`* "}"
          :| "foreach" `ForeachDeclaration` "in" `Object`
   ForeachDeclaration: ID "=" ( "{" `RangeList` "}" | `RangePiece` | `Value` )

The value assigned to the variable in the declaration is iterated over and
the object or object list is reevaluated with the variable set at each
iterated value.

Note that the productions involving RangeList and RangePiece have precedence
over the more generic value parsing based on the first token.

Top-Level ``let``
-----------------

.. productionlist::
   Let:  "let" `LetList` "in" "{" `Object`* "}"
      :| "let" `LetList` "in" `Object`
   LetList: `LetItem` ("," `LetItem`)*
   LetItem: `TokIdentifier` [`RangeList`] "=" `Value`

This is effectively equivalent to ``let`` inside the body of a record
except that it applies to multiple records at a time. The bindings are
applied at the end of parsing the base classes of a record.

``multiclass``
--------------

.. productionlist::
   MultiClass: "multiclass" `TokIdentifier` [`TemplateArgList`]
             : [":" `BaseMultiClassList`] "{" `MultiClassObject`+ "}"
   BaseMultiClassList: `MultiClassID` ("," `MultiClassID`)*
   MultiClassID: `TokIdentifier`
   MultiClassObject: `Def` | `Defm` | `Let` | `Foreach`

Preprocessing Support
=====================

TableGen's embedded preprocessor is only intended for conditional compilation.
It supports the following directives:

.. productionlist::
   LineBegin: ^
   LineEnd: "\n" | "\r" | EOF
   WhiteSpace: " " | "\t"
   CStyleComment: "/*" (.* - "*/") "*/"
   BCPLComment: "//" (.* - `LineEnd`) `LineEnd`
   WhiteSpaceOrCStyleComment: `WhiteSpace` | `CStyleComment`
   WhiteSpaceOrAnyComment: `WhiteSpace` | `CStyleComment` | `BCPLComment`
   MacroName: `ualpha` (`ualpha` | "0"..."9")*
   PrepDefine: `LineBegin` (`WhiteSpaceOrCStyleComment`)*
             : "#define" (`WhiteSpace`)+ `MacroName`
             : (`WhiteSpaceOrAnyComment`)* `LineEnd`
   PrepIfdef: `LineBegin` (`WhiteSpaceOrCStyleComment`)*
            : "#ifdef" (`WhiteSpace`)+ `MacroName`
            : (`WhiteSpaceOrAnyComment`)* `LineEnd`
   PrepElse: `LineBegin` (`WhiteSpaceOrCStyleComment`)*
           : "#else" (`WhiteSpaceOrAnyComment`)* `LineEnd`
   PrepEndif: `LineBegin` (`WhiteSpaceOrCStyleComment`)*
            : "#endif" (`WhiteSpaceOrAnyComment`)* `LineEnd`
   PrepRegContentException: `PredIfdef` | `PredElse` | `PredEndif` | EOF
   PrepRegion: .* - `PrepRegContentException`
             :| `PrepIfDef`
             :  (`PrepRegion`)*
             :  [`PrepElse`]
             :  (`PrepRegion`)*
             :  `PrepEndif`

:token:`PrepRegion` may occur anywhere in a TD file, as long as it matches
the grammar specification.

:token:`PrepDefine` allows defining a :token:`MacroName` so that any following
:token:`PrepIfdef` - :token:`PrepElse` preprocessing region part and
:token:`PrepIfdef` - :token:`PrepEndif` preprocessing region
are enabled for TableGen tokens parsing.

A preprocessing region, starting (i.e. having its :token:`PrepIfdef`) in a file,
must end (i.e. have its :token:`PrepEndif`) in the same file.

A :token:`MacroName` may be defined externally by using ``{ -D<NAME> }``
option of TableGen.
