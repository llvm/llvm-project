.. _tablegen:

=====================
TableGen Fundamentals
=====================

.. contents::
   :local:

Introduction
============

TableGen's purpose is to help a human develop and maintain records of
domain-specific information.  Because there may be a large number of these
records, it is specifically designed to allow writing flexible descriptions and
for common features of these records to be factored out.  This reduces the
amount of duplication in the description, reduces the chance of error, and makes
it easier to structure domain specific information.

The core part of TableGen `parses a file`_, instantiates the declarations, and
hands the result off to a domain-specific `TableGen backend`_ for processing.
The current major user of TableGen is the `LLVM code
generator <CodeGenerator.html>`_.

Note that if you work on TableGen much, and use emacs or vim, that you can find
an emacs "TableGen mode" and a vim language file in the ``llvm/utils/emacs`` and
``llvm/utils/vim`` directories of your LLVM distribution, respectively.

.. _intro:

Basic concepts
--------------

TableGen files consist of two key parts: 'classes' and 'definitions', both of
which are considered 'records'.

**TableGen records** have a unique name, a list of values, and a list of
superclasses.  The list of values is the main data that TableGen builds for each
record; it is this that holds the domain specific information for the
application.  The interpretation of this data is left to a specific `TableGen
backend`_, but the structure and format rules are taken care of and are fixed by
TableGen.

**TableGen definitions** are the concrete form of 'records'.  These generally do
not have any undefined values, and are marked with the '``def``' keyword.

**TableGen classes** are abstract records that are used to build and describe
other records.  These 'classes' allow the end-user to build abstractions for
either the domain they are targeting (such as "Register", "RegisterClass", and
"Instruction" in the LLVM code generator) or for the implementor to help factor
out common properties of records (such as "FPInst", which is used to represent
floating point instructions in the X86 backend).  TableGen keeps track of all of
the classes that are used to build up a definition, so the backend can find all
definitions of a particular class, such as "Instruction".

**TableGen multiclasses** are groups of abstract records that are instantiated
all at once.  Each instantiation can result in multiple TableGen definitions.
If a multiclass inherits from another multiclass, the definitions in the
sub-multiclass become part of the current multiclass, as if they were declared
in the current multiclass.

.. _described above:

An example record
-----------------

With no other arguments, TableGen parses the specified file and prints out all
of the classes, then all of the definitions.  This is a good way to see what the
various definitions expand to fully.  Running this on the ``X86.td`` file prints
this (at the time of this writing):

.. code-block:: llvm

  ...
  def ADD32rr {   // Instruction X86Inst I
    string Namespace = "X86";
    dag OutOperandList = (outs GR32:$dst);
    dag InOperandList = (ins GR32:$src1, GR32:$src2);
    string AsmString = "add{l}\t{$src2, $dst|$dst, $src2}";
    list<dag> Pattern = [(set GR32:$dst, (add GR32:$src1, GR32:$src2))];
    list<Register> Uses = [];
    list<Register> Defs = [EFLAGS];
    list<Predicate> Predicates = [];
    int CodeSize = 3;
    int AddedComplexity = 0;
    bit isReturn = 0;
    bit isBranch = 0;
    bit isIndirectBranch = 0;
    bit isBarrier = 0;
    bit isCall = 0;
    bit canFoldAsLoad = 0;
    bit mayLoad = 0;
    bit mayStore = 0;
    bit isImplicitDef = 0;
    bit isConvertibleToThreeAddress = 1;
    bit isCommutable = 1;
    bit isTerminator = 0;
    bit isReMaterializable = 0;
    bit isPredicable = 0;
    bit hasDelaySlot = 0;
    bit usesCustomInserter = 0;
    bit hasCtrlDep = 0;
    bit isNotDuplicable = 0;
    bit hasSideEffects = 0;
    bit neverHasSideEffects = 0;
    InstrItinClass Itinerary = NoItinerary;
    string Constraints = "";
    string DisableEncoding = "";
    bits<8> Opcode = { 0, 0, 0, 0, 0, 0, 0, 1 };
    Format Form = MRMDestReg;
    bits<6> FormBits = { 0, 0, 0, 0, 1, 1 };
    ImmType ImmT = NoImm;
    bits<3> ImmTypeBits = { 0, 0, 0 };
    bit hasOpSizePrefix = 0;
    bit hasAdSizePrefix = 0;
    bits<4> Prefix = { 0, 0, 0, 0 };
    bit hasREX_WPrefix = 0;
    FPFormat FPForm = ?;
    bits<3> FPFormBits = { 0, 0, 0 };
  }
  ...

This definition corresponds to a 32-bit register-register add instruction in the
X86.  The string after the '``def``' string indicates the name of the
record---"``ADD32rr``" in this case---and the comment at the end of the line
indicates the superclasses of the definition.  The body of the record contains
all of the data that TableGen assembled for the record, indicating that the
instruction is part of the "X86" namespace, the pattern indicating how the the
instruction should be emitted into the assembly file, that it is a two-address
instruction, has a particular encoding, etc.  The contents and semantics of the
information in the record is specific to the needs of the X86 backend, and is
only shown as an example.

As you can see, a lot of information is needed for every instruction supported
by the code generator, and specifying it all manually would be unmaintainable,
prone to bugs, and tiring to do in the first place.  Because we are using
TableGen, all of the information was derived from the following definition:

.. code-block:: llvm

  let Defs = [EFLAGS],
      isCommutable = 1,                  // X = ADD Y,Z --> X = ADD Z,Y
      isConvertibleToThreeAddress = 1 in // Can transform into LEA.
  def ADD32rr  : I<0x01, MRMDestReg, (outs GR32:$dst),
                                     (ins GR32:$src1, GR32:$src2),
                   "add{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (add GR32:$src1, GR32:$src2))]>;

This definition makes use of the custom class ``I`` (extended from the custom
class ``X86Inst``), which is defined in the X86-specific TableGen file, to
factor out the common features that instructions of its class share.  A key
feature of TableGen is that it allows the end-user to define the abstractions
they prefer to use when describing their information.

Each def record has a special entry called "``NAME``."  This is the name of the
def ("``ADD32rr``" above).  In the general case def names can be formed from
various kinds of string processing expressions and ``NAME`` resolves to the
final value obtained after resolving all of those expressions.  The user may
refer to ``NAME`` anywhere she desires to use the ultimate name of the def.
``NAME`` should not be defined anywhere else in user code to avoid conflict
problems.

Running TableGen
----------------

TableGen runs just like any other LLVM tool.  The first (optional) argument
specifies the file to read.  If a filename is not specified, ``llvm-tblgen``
reads from standard input.

To be useful, one of the `TableGen backends`_ must be used.  These backends are
selectable on the command line (type '``llvm-tblgen -help``' for a list).  For
example, to get a list of all of the definitions that subclass a particular type
(which can be useful for building up an enum list of these records), use the
``-print-enums`` option:

.. code-block:: bash

  $ llvm-tblgen X86.td -print-enums -class=Register
  AH, AL, AX, BH, BL, BP, BPL, BX, CH, CL, CX, DH, DI, DIL, DL, DX, EAX, EBP, EBX,
  ECX, EDI, EDX, EFLAGS, EIP, ESI, ESP, FP0, FP1, FP2, FP3, FP4, FP5, FP6, IP,
  MM0, MM1, MM2, MM3, MM4, MM5, MM6, MM7, R10, R10B, R10D, R10W, R11, R11B, R11D,
  R11W, R12, R12B, R12D, R12W, R13, R13B, R13D, R13W, R14, R14B, R14D, R14W, R15,
  R15B, R15D, R15W, R8, R8B, R8D, R8W, R9, R9B, R9D, R9W, RAX, RBP, RBX, RCX, RDI,
  RDX, RIP, RSI, RSP, SI, SIL, SP, SPL, ST0, ST1, ST2, ST3, ST4, ST5, ST6, ST7,
  XMM0, XMM1, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15, XMM2, XMM3, XMM4, XMM5,
  XMM6, XMM7, XMM8, XMM9,

  $ llvm-tblgen X86.td -print-enums -class=Instruction 
  ABS_F, ABS_Fp32, ABS_Fp64, ABS_Fp80, ADC32mi, ADC32mi8, ADC32mr, ADC32ri,
  ADC32ri8, ADC32rm, ADC32rr, ADC64mi32, ADC64mi8, ADC64mr, ADC64ri32, ADC64ri8,
  ADC64rm, ADC64rr, ADD16mi, ADD16mi8, ADD16mr, ADD16ri, ADD16ri8, ADD16rm,
  ADD16rr, ADD32mi, ADD32mi8, ADD32mr, ADD32ri, ADD32ri8, ADD32rm, ADD32rr,
  ADD64mi32, ADD64mi8, ADD64mr, ADD64ri32, ...

The default backend prints out all of the records, as `described above`_.

If you plan to use TableGen, you will most likely have to `write a backend`_
that extracts the information specific to what you need and formats it in the
appropriate way.

.. _parses a file:

TableGen syntax
===============

TableGen doesn't care about the meaning of data (that is up to the backend to
define), but it does care about syntax, and it enforces a simple type system.
This section describes the syntax and the constructs allowed in a TableGen file.

TableGen primitives
-------------------

TableGen comments
^^^^^^^^^^^^^^^^^

TableGen supports BCPL style "``//``" comments, which run to the end of the
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

``code``
    This represents a big hunk of text.  This is lexically distinct from string
    values because it doesn't require escaping double quotes and other common
    characters that occur in code.

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
    binary integer value

``07654321``
    octal integer value (indicated by a leading 0)

``7``
    decimal integer value

``0x7F``
    hexadecimal integer value

``"foo"``
    string value

``[{ ... }]``
    code fragment

``[ X, Y, Z ]<type>``
    list value.  <type> is the type of the list element and is usually optional.
    In rare cases, TableGen is unable to deduce the element type in which case
    the user must specify it explicitly.

``{ a, b, c }``
    initializer for a "bits<3>" value

``value``
    value reference

``value{17}``
    access to one bit of a value

``value{15-17}``
    access to multiple bits of a value

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
    not conflict with any other object introduced in <body> or <def>.  Currently
    only ``def``\s are expanded within <body>.

``foreach <var> = 0-15 in ...``

``foreach <var> = {0-15,32-47} in ...``
    Loop over ranges of integers. The braces are required for multiple ranges.

``(DEF a, b)``
    a dag value.  The first element is required to be a record definition, the
    remaining elements in the list may be arbitrary other values, including
    nested ```dag``' values.

``!strconcat(a, b)``
    A string value that is the result of concatenating the 'a' and 'b' strings.

``str1#str2``
    "#" (paste) is a shorthand for !strconcat.  It may concatenate things that
    are not quoted strings, in which case an implicit !cast<string> is done on
    the operand of the paste.

``!cast<type>(a)``
    A symbol of type *type* obtained by looking up the string 'a' in the symbol
    table.  If the type of 'a' does not match *type*, TableGen aborts with an
    error. !cast<string> is a special case in that the argument must be an
    object defined by a 'def' construct.

``!subst(a, b, c)``
    If 'a' and 'b' are of string type or are symbol references, substitute 'b'
    for 'a' in 'c.'  This operation is analogous to $(subst) in GNU make.

``!foreach(a, b, c)``
    For each member 'b' of dag or list 'a' apply operator 'c.'  'b' is a dummy
    variable that should be declared as a member variable of an instantiated
    class.  This operation is analogous to $(foreach) in GNU make.

``!head(a)``
    The first element of list 'a.'

``!tail(a)``
    The 2nd-N elements of list 'a.'

``!empty(a)``
    An integer {0,1} indicating whether list 'a' is empty.

``!if(a,b,c)``
  'b' if the result of 'int' or 'bit' operator 'a' is nonzero, 'c' otherwise.

``!eq(a,b)``
    'bit 1' if string a is equal to string b, 0 otherwise.  This only operates
    on string, int and bit objects.  Use !cast<string> to compare other types of
    objects.

Note that all of the values have rules specifying how they convert to values
for different types.  These rules allow you to assign a value like "``7``"
to a "``bits<4>``" value, for example.

Classes and definitions
-----------------------

As mentioned in the `intro`_, classes and definitions (collectively known as
'records') in TableGen are the main high-level unit of information that TableGen
collects.  Records are defined with a ``def`` or ``class`` keyword, the record
name, and an optional list of "`template arguments`_".  If the record has
superclasses, they are specified as a comma separated list that starts with a
colon character ("``:``").  If `value definitions`_ or `let expressions`_ are
needed for the class, they are enclosed in curly braces ("``{}``"); otherwise,
the record ends with a semicolon.

Here is a simple TableGen file:

.. code-block:: llvm

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

.. code-block:: llvm

  class D : C { let V = 0; }
  def Z : D;

In this case, the ``Z`` definition will have a zero value for its ``V`` value,
despite the fact that it derives (indirectly) from the ``C`` class, because the
``D`` class overrode its value.

.. _template arguments:

Class template arguments
^^^^^^^^^^^^^^^^^^^^^^^^

TableGen permits the definition of parameterized classes as well as normal
concrete classes.  Parameterized TableGen classes specify a list of variable
bindings (which may optionally have defaults) that are bound when used.  Here is
a simple example:

.. code-block:: llvm

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

.. code-block:: llvm

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

.. code-block:: llvm

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

.. code-block:: llvm

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

.. code-block:: llvm

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
multiclass instanciations.

.. code-block:: llvm

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

.. code-block:: llvm

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

.. code-block:: llvm

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

.. code-block:: llvm

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
levels of multiclass instanciations. This also avoids the need of using "let"
expressions within subsequent records inside a multiclass.

.. code-block:: llvm

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

.. code-block:: llvm

  foreach i = [0, 1, 2, 3] in {
    def R#i : Register<...>;
    def F#i : Register<...>;
  }

This will create objects ``R0``, ``R1``, ``R2`` and ``R3``.  ``foreach`` blocks
may be nested. If there is only one item in the body the braces may be
elided:

.. code-block:: llvm

  foreach i = [0, 1, 2, 3] in
    def R#i : Register<...>;

Code Generator backend info
===========================

Expressions used by code generator to describe instructions and isel patterns:

``(implicit a)``
    an implicitly defined physical register.  This tells the dag instruction
    selection emitter the input pattern's extra definitions matches implicit
    physical register definitions.

.. _TableGen backend:
.. _TableGen backends:
.. _write a backend:

TableGen backends
=================

TODO: How they work, how to write one.  This section should not contain details
about any particular backend, except maybe ``-print-enums`` as an example.  This
should highlight the APIs in ``TableGen/Record.h``.
