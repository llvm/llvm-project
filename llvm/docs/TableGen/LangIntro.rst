==============================
TableGen Language Introduction
==============================

.. contents::
   :local:

.. warning::
   This document is extremely rough. If you find something lacking, please
   fix it, file a documentation bug, or ask about it on llvm-dev.

Introduction
============

This document is not meant to be a normative spec about the TableGen language
in and of itself (i.e. how to understand a given construct in terms of how
it affects the final set of records represented by the TableGen file). For
the formal language specification, see :doc:`LangRef`.

TableGen syntax
===============

TableGen doesn't care about the meaning of data (that is up to the backend to
define), but it does care about syntax, and it enforces a simple type system.
This section describes the syntax and the constructs allowed in a TableGen file.

TableGen primitives
-------------------

TableGen comments
^^^^^^^^^^^^^^^^^

TableGen supports C++ style "``//``" comments, which run to the end of the
line, and it also supports **nestable** "``/* */``" comments.

.. _TableGen type:

The TableGen type system
^^^^^^^^^^^^^^^^^^^^^^^^

TableGen files are strongly typed, in a simple (but complete) type-system.
These types are used to perform automatic conversions, check for errors, and to
help interface designers constrain the input that they allow.  Every `value
definition`_ is required to have an associated type.

TableGen supports a mixture of very low-level types (such as ``bit``) and very
high-level types (such as ``dag``).  This flexibility is what allows it to
describe a wide range of information conveniently and compactly.  The TableGen
types are:

``bit``
    A 'bit' is a boolean value that can hold either 0 or 1.

``int``
    The 'int' type represents a simple 32-bit integer value, such as 5.

``string``
    The 'string' type represents an ordered sequence of characters of arbitrary
    length.

``code``
    The `code` type represents a code fragment, which can be single/multi-line
    string literal.

``bits<n>``
    A 'bits' type is an arbitrary, but fixed, size integer that is broken up
    into individual bits.  This type is useful because it can handle some bits
    being defined while others are undefined.

``list<ty>``
    This type represents a list whose elements are some other type.  The
    contained type is arbitrary: it can even be another list type.

Class type
    Specifying a class name in a type context means that the defined value must
    be a subclass of the specified class.  This is useful in conjunction with
    the ``list`` type, for example, to constrain the elements of the list to a
    common base class (e.g., a ``list<Register>`` can only contain definitions
    derived from the "``Register``" class).

``dag``
    This type represents a nestable directed graph of elements.

To date, these types have been sufficient for describing things that TableGen
has been used for, but it is straight-forward to extend this list if needed.

.. _TableGen expressions:

TableGen values and expressions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TableGen allows for a pretty reasonable number of different expression forms
when building up values.  These forms allow the TableGen file to be written in a
natural syntax and flavor for the application.  The current expression forms
supported include:

``?``
    uninitialized field

``0b1001011``
    binary integer value.
    Note that this is sized by the number of bits given and will not be
    silently extended/truncated.

``7``
    decimal integer value

``0x7F``
    hexadecimal integer value

``"foo"``
    a single-line string value, can be assigned to ``string`` or ``code`` variable.

``[{ ... }]``
    usually called a "code fragment", but is just a multiline string literal

``[ X, Y, Z ]<type>``
    list value.  <type> is the type of the list element and is usually optional.
    In rare cases, TableGen is unable to deduce the element type in which case
    the user must specify it explicitly.

``{ a, b, 0b10 }``
    initializer for a "bits<4>" value.
    1-bit from "a", 1-bit from "b", 2-bits from 0b10.

``value``
    value reference

``value{17}``
    access to one bit of a value

``value{15-17}``
    access to an ordered sequence of bits of a value, in particular ``value{15-17}``
    produces an order that is the reverse of ``value{17-15}``.

``DEF``
    reference to a record definition

``CLASS<val list>``
    reference to a new anonymous definition of CLASS with the specified template
    arguments.

``X.Y``
    reference to the subfield of a value

``list[4-7,17,2-3]``
    A slice of the 'list' list, including elements 4,5,6,7,17,2, and 3 from it.
    Elements may be included multiple times.

``foreach <var> = [ <list> ] in { <body> }``

``foreach <var> = [ <list> ] in <def>``
    Replicate <body> or <def>, replacing instances of <var> with each value
    in <list>.  <var> is scoped at the level of the ``foreach`` loop and must
    not conflict with any other object introduced in <body> or <def>.  Only
    ``def``\s and ``defm``\s are expanded within <body>.

``foreach <var> = 0-15 in ...``

``foreach <var> = {0-15,32-47} in ...``
    Loop over ranges of integers. The braces are required for multiple ranges.

``(DEF a, b)``
    a dag value.  The first element is required to be a record definition, the
    remaining elements in the list may be arbitrary other values, including
    nested ```dag``' values.

``!con(a, b, ...)``
    Concatenate two or more DAG nodes. Their operations must equal.

    Example: !con((op a1:$name1, a2:$name2), (op b1:$name3)) results in
    the DAG node (op a1:$name1, a2:$name2, b1:$name3).

``!dag(op, children, names)``
    Generate a DAG node programmatically. 'children' and 'names' must be lists
    of equal length or unset ('?'). 'names' must be a 'list<string>'.

    Due to limitations of the type system, 'children' must be a list of items
    of a common type. In practice, this means that they should either have the
    same type or be records with a common superclass. Mixing dag and non-dag
    items is not possible. However, '?' can be used.

    Example: !dag(op, [a1, a2, ?], ["name1", "name2", "name3"]) results in
    (op a1:$name1, a2:$name2, ?:$name3).

``!listconcat(a, b, ...)``
    A list value that is the result of concatenating the 'a' and 'b' lists.
    The lists must have the same element type.
    More than two arguments are accepted with the result being the concatenation
    of all the lists given.

``!strconcat(a, b, ...)``
    A string value that is the result of concatenating the 'a' and 'b' strings.
    More than two arguments are accepted with the result being the concatenation
    of all the strings given.

``str1#str2``
    "#" (paste) is a shorthand for !strconcat.  It may concatenate things that
    are not quoted strings, in which case an implicit !cast<string> is done on
    the operand of the paste.

``!cast<type>(a)``
    If 'a' is a string, a record of type *type* obtained by looking up the
    string 'a' in the list of all records defined by the time that all template
    arguments in 'a' are fully resolved.

    For example, if !cast<type>(a) appears in a multiclass definition, or in a
    class instantiated inside of a multiclass definition, and 'a' does not
    reference any template arguments of the multiclass, then a record of name
    'a' must be instantiated earlier in the source file. If 'a' does reference
    a template argument, then the lookup is delayed until defm statements
    instantiating the multiclass (or later, if the defm occurs in another
    multiclass and template arguments of the inner multiclass that are
    referenced by 'a' are substituted by values that themselves contain
    references to template arguments of the outer multiclass).

    If the type of 'a' does not match *type*, TableGen aborts with an error.

    Otherwise, perform a normal type cast e.g. between an int and a bit, or
    between record types. This allows casting a record to a subclass, though if
    the types do not match, constant folding will be inhibited. !cast<string>
    is a special case in that the argument can be an int or a record. In the
    latter case, the record's name is returned.

``!isa<type>(a)``
    Returns an integer: 1 if 'a' is dynamically of the given type, 0 otherwise.

``!subst(a, b, c)``
    If 'a' and 'b' are of string type or are symbol references, substitute 'b'
    for 'a' in 'c.'  This operation is analogous to $(subst) in GNU make.

``!foreach(a, b, c)``
    For each member of dag or list 'b' apply operator 'c'. 'a' is the name
    of a variable that will be substituted by members of 'b' in 'c'.
    This operation is analogous to $(foreach) in GNU make.

``!foldl(start, lst, a, b, expr)``
    Perform a left-fold over 'lst' with the given starting value. 'a' and 'b'
    are variable names which will be substituted in 'expr'. If you think of
    expr as a function f(a,b), the fold will compute
    'f(...f(f(start, lst[0]), lst[1]), ...), lst[n-1])' for a list of length n.
    As usual, 'a' will be of the type of 'start', and 'b' will be of the type
    of elements of 'lst'. These types need not be the same, but 'expr' must be
    of the same type as 'start'.

``!head(a)``
    The first element of list 'a.'

``!tail(a)``
    The 2nd-N elements of list 'a.'

``!empty(a)``
    An integer {0,1} indicating whether list 'a' is empty.

``!size(a)``
    An integer indicating the number of elements in list 'a'.

``!if(a,b,c)``
  'b' if the result of 'int' or 'bit' operator 'a' is nonzero, 'c' otherwise.

``!eq(a,b)``
    'bit 1' if string a is equal to string b, 0 otherwise.  This only operates
    on string, int and bit objects.  Use !cast<string> to compare other types of
    objects.

``!ne(a,b)``
    The negation of ``!eq(a,b)``.

``!le(a,b), !lt(a,b), !ge(a,b), !gt(a,b)``
    (Signed) comparison of integer values that returns bit 1 or 0 depending on
    the result of the comparison.

``!shl(a,b)`` ``!srl(a,b)`` ``!sra(a,b)``
    The usual shift operators. Operations are on 64-bit integers, the result
    is undefined for shift counts outside [0, 63].

``!add(a,b,...)`` ``!and(a,b,...)`` ``!or(a,b,...)``
    The usual arithmetic and binary operators.

Note that all of the values have rules specifying how they convert to values
for different types.  These rules allow you to assign a value like "``7``"
to a "``bits<4>``" value, for example.

Classes and definitions
-----------------------

As mentioned in the :doc:`introduction <index>`, classes and definitions (collectively known as
'records') in TableGen are the main high-level unit of information that TableGen
collects.  Records are defined with a ``def`` or ``class`` keyword, the record
name, and an optional list of "`template arguments`_".  If the record has
superclasses, they are specified as a comma separated list that starts with a
colon character ("``:``").  If `value definitions`_ or `let expressions`_ are
needed for the class, they are enclosed in curly braces ("``{}``"); otherwise,
the record ends with a semicolon.

Here is a simple TableGen file:

.. code-block:: text

  class C { bit V = 1; }
  def X : C;
  def Y : C {
    string Greeting = "hello";
  }

This example defines two definitions, ``X`` and ``Y``, both of which derive from
the ``C`` class.  Because of this, they both get the ``V`` bit value.  The ``Y``
definition also gets the Greeting member as well.

In general, classes are useful for collecting together the commonality between a
group of records and isolating it in a single place.  Also, classes permit the
specification of default values for their subclasses, allowing the subclasses to
override them as they wish.

.. _value definition:
.. _value definitions:

Value definitions
^^^^^^^^^^^^^^^^^

Value definitions define named entries in records.  A value must be defined
before it can be referred to as the operand for another value definition or
before the value is reset with a `let expression`_.  A value is defined by
specifying a `TableGen type`_ and a name.  If an initial value is available, it
may be specified after the type with an equal sign.  Value definitions require
terminating semicolons.

.. _let expression:
.. _let expressions:
.. _"let" expressions within a record:

'let' expressions
^^^^^^^^^^^^^^^^^

A record-level let expression is used to change the value of a value definition
in a record.  This is primarily useful when a superclass defines a value that a
derived class or definition wants to override.  Let expressions consist of the
'``let``' keyword followed by a value name, an equal sign ("``=``"), and a new
value.  For example, a new class could be added to the example above, redefining
the ``V`` field for all of its subclasses:

.. code-block:: text

  class D : C { let V = 0; }
  def Z : D;

In this case, the ``Z`` definition will have a zero value for its ``V`` value,
despite the fact that it derives (indirectly) from the ``C`` class, because the
``D`` class overrode its value.

References between variables in a record are substituted late, which gives
``let`` expressions unusual power. Consider this admittedly silly example:

.. code-block:: text

  class A<int x> {
    int Y = x;
    int Yplus1 = !add(Y, 1);
    int xplus1 = !add(x, 1);
  }
  def Z : A<5> {
    let Y = 10;
  }

The value of ``Z.xplus1`` will be 6, but the value of ``Z.Yplus1`` is 11. Use
this power wisely.

.. _template arguments:

Class template arguments
^^^^^^^^^^^^^^^^^^^^^^^^

TableGen permits the definition of parameterized classes as well as normal
concrete classes.  Parameterized TableGen classes specify a list of variable
bindings (which may optionally have defaults) that are bound when used.  Here is
a simple example:

.. code-block:: text

  class FPFormat<bits<3> val> {
    bits<3> Value = val;
  }
  def NotFP      : FPFormat<0>;
  def ZeroArgFP  : FPFormat<1>;
  def OneArgFP   : FPFormat<2>;
  def OneArgFPRW : FPFormat<3>;
  def TwoArgFP   : FPFormat<4>;
  def CompareFP  : FPFormat<5>;
  def CondMovFP  : FPFormat<6>;
  def SpecialFP  : FPFormat<7>;

In this case, template arguments are used as a space efficient way to specify a
list of "enumeration values", each with a "``Value``" field set to the specified
integer.

The more esoteric forms of `TableGen expressions`_ are useful in conjunction
with template arguments.  As an example:

.. code-block:: text

  class ModRefVal<bits<2> val> {
    bits<2> Value = val;
  }

  def None   : ModRefVal<0>;
  def Mod    : ModRefVal<1>;
  def Ref    : ModRefVal<2>;
  def ModRef : ModRefVal<3>;

  class Value<ModRefVal MR> {
    // Decode some information into a more convenient format, while providing
    // a nice interface to the user of the "Value" class.
    bit isMod = MR.Value{0};
    bit isRef = MR.Value{1};

    // other stuff...
  }

  // Example uses
  def bork : Value<Mod>;
  def zork : Value<Ref>;
  def hork : Value<ModRef>;

This is obviously a contrived example, but it shows how template arguments can
be used to decouple the interface provided to the user of the class from the
actual internal data representation expected by the class.  In this case,
running ``llvm-tblgen`` on the example prints the following definitions:

.. code-block:: text

  def bork {      // Value
    bit isMod = 1;
    bit isRef = 0;
  }
  def hork {      // Value
    bit isMod = 1;
    bit isRef = 1;
  }
  def zork {      // Value
    bit isMod = 0;
    bit isRef = 1;
  }

This shows that TableGen was able to dig into the argument and extract a piece
of information that was requested by the designer of the "Value" class.  For
more realistic examples, please see existing users of TableGen, such as the X86
backend.

Multiclass definitions and instances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While classes with template arguments are a good way to factor commonality
between two instances of a definition, multiclasses allow a convenient notation
for defining multiple definitions at once (instances of implicitly constructed
classes).  For example, consider an 3-address instruction set whose instructions
come in two forms: "``reg = reg op reg``" and "``reg = reg op imm``"
(e.g. SPARC). In this case, you'd like to specify in one place that this
commonality exists, then in a separate place indicate what all the ops are.

Here is an example TableGen fragment that shows this idea:

.. code-block:: text

  def ops;
  def GPR;
  def Imm;
  class inst<int opc, string asmstr, dag operandlist>;

  multiclass ri_inst<int opc, string asmstr> {
    def _rr : inst<opc, !strconcat(asmstr, " $dst, $src1, $src2"),
                   (ops GPR:$dst, GPR:$src1, GPR:$src2)>;
    def _ri : inst<opc, !strconcat(asmstr, " $dst, $src1, $src2"),
                   (ops GPR:$dst, GPR:$src1, Imm:$src2)>;
  }

  // Instantiations of the ri_inst multiclass.
  defm ADD : ri_inst<0b111, "add">;
  defm SUB : ri_inst<0b101, "sub">;
  defm MUL : ri_inst<0b100, "mul">;
  ...

The name of the resultant definitions has the multidef fragment names appended
to them, so this defines ``ADD_rr``, ``ADD_ri``, ``SUB_rr``, etc.  A defm may
inherit from multiple multiclasses, instantiating definitions from each
multiclass.  Using a multiclass this way is exactly equivalent to instantiating
the classes multiple times yourself, e.g. by writing:

.. code-block:: text

  def ops;
  def GPR;
  def Imm;
  class inst<int opc, string asmstr, dag operandlist>;

  class rrinst<int opc, string asmstr>
    : inst<opc, !strconcat(asmstr, " $dst, $src1, $src2"),
           (ops GPR:$dst, GPR:$src1, GPR:$src2)>;

  class riinst<int opc, string asmstr>
    : inst<opc, !strconcat(asmstr, " $dst, $src1, $src2"),
           (ops GPR:$dst, GPR:$src1, Imm:$src2)>;

  // Instantiations of the ri_inst multiclass.
  def ADD_rr : rrinst<0b111, "add">;
  def ADD_ri : riinst<0b111, "add">;
  def SUB_rr : rrinst<0b101, "sub">;
  def SUB_ri : riinst<0b101, "sub">;
  def MUL_rr : rrinst<0b100, "mul">;
  def MUL_ri : riinst<0b100, "mul">;
  ...

A ``defm`` can also be used inside a multiclass providing several levels of
multiclass instantiations.

.. code-block:: text

  class Instruction<bits<4> opc, string Name> {
    bits<4> opcode = opc;
    string name = Name;
  }

  multiclass basic_r<bits<4> opc> {
    def rr : Instruction<opc, "rr">;
    def rm : Instruction<opc, "rm">;
  }

  multiclass basic_s<bits<4> opc> {
    defm SS : basic_r<opc>;
    defm SD : basic_r<opc>;
    def X : Instruction<opc, "x">;
  }

  multiclass basic_p<bits<4> opc> {
    defm PS : basic_r<opc>;
    defm PD : basic_r<opc>;
    def Y : Instruction<opc, "y">;
  }

  defm ADD : basic_s<0xf>, basic_p<0xf>;
  ...

  // Results
  def ADDPDrm { ...
  def ADDPDrr { ...
  def ADDPSrm { ...
  def ADDPSrr { ...
  def ADDSDrm { ...
  def ADDSDrr { ...
  def ADDY { ...
  def ADDX { ...

``defm`` declarations can inherit from classes too, the rule to follow is that
the class list must start after the last multiclass, and there must be at least
one multiclass before them.

.. code-block:: text

  class XD { bits<4> Prefix = 11; }
  class XS { bits<4> Prefix = 12; }

  class I<bits<4> op> {
    bits<4> opcode = op;
  }

  multiclass R {
    def rr : I<4>;
    def rm : I<2>;
  }

  multiclass Y {
    defm SS : R, XD;
    defm SD : R, XS;
  }

  defm Instr : Y;

  // Results
  def InstrSDrm {
    bits<4> opcode = { 0, 0, 1, 0 };
    bits<4> Prefix = { 1, 1, 0, 0 };
  }
  ...
  def InstrSSrr {
    bits<4> opcode = { 0, 1, 0, 0 };
    bits<4> Prefix = { 1, 0, 1, 1 };
  }

File scope entities
-------------------

File inclusion
^^^^^^^^^^^^^^

TableGen supports the '``include``' token, which textually substitutes the
specified file in place of the include directive.  The filename should be
specified as a double quoted string immediately after the '``include``' keyword.
Example:

.. code-block:: text

  include "foo.td"

'let' expressions
^^^^^^^^^^^^^^^^^

"Let" expressions at file scope are similar to `"let" expressions within a
record`_, except they can specify a value binding for multiple records at a
time, and may be useful in certain other cases.  File-scope let expressions are
really just another way that TableGen allows the end-user to factor out
commonality from the records.

File-scope "let" expressions take a comma-separated list of bindings to apply,
and one or more records to bind the values in.  Here are some examples:

.. code-block:: text

  let isTerminator = 1, isReturn = 1, isBarrier = 1, hasCtrlDep = 1 in
    def RET : I<0xC3, RawFrm, (outs), (ins), "ret", [(X86retflag 0)]>;

  let isCall = 1 in
    // All calls clobber the non-callee saved registers...
    let Defs = [EAX, ECX, EDX, FP0, FP1, FP2, FP3, FP4, FP5, FP6, ST0,
                MM0, MM1, MM2, MM3, MM4, MM5, MM6, MM7,
                XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, EFLAGS] in {
      def CALLpcrel32 : Ii32<0xE8, RawFrm, (outs), (ins i32imm:$dst,variable_ops),
                             "call\t${dst:call}", []>;
      def CALL32r     : I<0xFF, MRM2r, (outs), (ins GR32:$dst, variable_ops),
                          "call\t{*}$dst", [(X86call GR32:$dst)]>;
      def CALL32m     : I<0xFF, MRM2m, (outs), (ins i32mem:$dst, variable_ops),
                          "call\t{*}$dst", []>;
    }

File-scope "let" expressions are often useful when a couple of definitions need
to be added to several records, and the records do not otherwise need to be
opened, as in the case with the ``CALL*`` instructions above.

It's also possible to use "let" expressions inside multiclasses, providing more
ways to factor out commonality from the records, specially if using several
levels of multiclass instantiations. This also avoids the need of using "let"
expressions within subsequent records inside a multiclass.

.. code-block:: text

  multiclass basic_r<bits<4> opc> {
    let Predicates = [HasSSE2] in {
      def rr : Instruction<opc, "rr">;
      def rm : Instruction<opc, "rm">;
    }
    let Predicates = [HasSSE3] in
      def rx : Instruction<opc, "rx">;
  }

  multiclass basic_ss<bits<4> opc> {
    let IsDouble = 0 in
      defm SS : basic_r<opc>;

    let IsDouble = 1 in
      defm SD : basic_r<opc>;
  }

  defm ADD : basic_ss<0xf>;

Looping
^^^^^^^

TableGen supports the '``foreach``' block, which textually replicates the loop
body, substituting iterator values for iterator references in the body.
Example:

.. code-block:: text

  foreach i = [0, 1, 2, 3] in {
    def R#i : Register<...>;
    def F#i : Register<...>;
  }

This will create objects ``R0``, ``R1``, ``R2`` and ``R3``.  ``foreach`` blocks
may be nested. If there is only one item in the body the braces may be
elided:

.. code-block:: text

  foreach i = [0, 1, 2, 3] in
    def R#i : Register<...>;

Code Generator backend info
===========================

Expressions used by code generator to describe instructions and isel patterns:

``(implicit a)``
    an implicitly defined physical register.  This tells the dag instruction
    selection emitter the input pattern's extra definitions matches implicit
    physical register definitions.

