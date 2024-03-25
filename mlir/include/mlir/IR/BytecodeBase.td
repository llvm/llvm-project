//===-- BytecodeBase.td - Base bytecode R/W defs -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the base bytecode reader/writer definition file.
//
//===----------------------------------------------------------------------===//

#ifndef BYTECODE_BASE
#define BYTECODE_BASE

class Bytecode<string parse="", string build="", string print="", string get="", string t=""> {
  // Template for parsing.
  //   $_reader == dialect bytecode reader
  //   $_resultType == result type of parsed instance
  //   $_var == variable being parsed
  // If parser is not specified, then the parse of members is used.
  string cParser = parse;

  // Template for building from parsed.
  //   $_resultType == result type of parsed instance
  //   $_args == args/members comma separated
  string cBuilder = build;

  // Template for printing.
  //   $_writer == dialect bytecode writer
  //   $_name == parent attribute/type name
  //   $_getter == getter
  string cPrinter = print;

  // Template for getter from in memory form.
  //   $_attrType == attribute/type
  //   $_member == member instance
  //   $_getMember == get + UpperCamelFromSnake($_member)
  string cGetter = get;

  // Type built.
  // Note: if cType is empty, then name of def is used.
  string cType = t;

  // Predicate guarding parse method as an Attribute/Type could have multiple
  // parse methods, specify predicates to be orthogonal and cover entire
  // "print space" to avoid order dependence.
  // If empty then method is unconditional.
  //   $_val == predicate function to apply on value dyn_casted to cType.
  string printerPredicate = "";
}

class WithParser<string p="", Bytecode t=Bytecode<>> :
  Bytecode<p, t.cBuilder, t.cPrinter, t.cGetter, t.cType>;
class WithBuilder<string b="", Bytecode t=Bytecode<>> :
  Bytecode<t.cParser, b, t.cPrinter, t.cGetter, t.cType>;
class WithPrinter<string p="", Bytecode t=Bytecode<>> :
  Bytecode<t.cParser, t.cBuilder, p, t.cGetter, t.cType>;
class WithType<string ty="", Bytecode t=Bytecode<>> :
  Bytecode<t.cParser, t.cBuilder, t.cPrinter, t.cGetter, ty>;
class WithGetter<string g="", Bytecode t=Bytecode<>> :
  Bytecode<t.cParser, t.cBuilder, t.cPrinter, g, t.cType>;

// Representation of a bytecode element consisting of other bytecode atoms.
// E.g., it is effectively a struct of bytecode elements. Set the members by
// define a members dag: `dag members = (attr ...)`.
class CompositeBytecode<string t = ""> : WithType<t>;

class AttributeKind :
  WithParser <"succeeded($_reader.readAttribute($_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeAttribute($_getter)">>>;
def Attribute : AttributeKind;
class TypeKind :
  WithParser <"succeeded($_reader.readType($_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeType($_getter)">>>;
def Type : TypeKind;

def VarInt :
  WithParser <"succeeded($_reader.readVarInt($_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeVarInt($_getter)",
  WithType   <"uint64_t">>>>;
def SignedVarInt :
  WithParser <"succeeded($_reader.readSignedVarInt($_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeSignedVarInt($_getter)",
  WithType   <"int64_t">>>>;
def Blob :
  WithParser <"succeeded($_reader.readBlob($_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeOwnedBlob($_getter)",
  WithType   <"ArrayRef<char>">>>>;
def Bool :
  WithParser <"succeeded($_reader.readBool($_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeOwnedBool($_getter)",
  WithType   <"bool">>>>;
class KnownWidthAPInt<string s> :
  WithParser <"succeeded(readAPIntWithKnownWidth($_reader, " # s # ", $_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeAPIntWithKnownWidth($_getter)",
  WithType   <"FailureOr<APInt>">>>>;
class KnownSemanticsAPFloat<string s> :
  WithParser <"succeeded(readAPFloatWithKnownSemantics($_reader, " # s # ", $_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeAPFloatWithKnownSemantics($_getter)",
  WithType   <"FailureOr<APFloat>">>>>;
class ResourceHandle<string s> :
  WithParser <"succeeded(readResourceHandle<" # s # ">($_reader, $_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeResourceHandle($_getter)",
  WithType   <"FailureOr<" # s # ">">>>>;

// Helper to define variable that is defined later but not parsed nor printed.
class LocalVar<string t, string d> :
  WithParser <"(($_var = " # d # "), true)",
  WithBuilder<"$_args",
  WithPrinter<"",
  WithType   <t>>>>;

// Array instances.
class Array<Bytecode t> {
  Bytecode elemT = t;

  string cBuilder = "$_args";
}
// - Array elements currently needs a different bytecode type to accommodate
//   for the list print/parsing.
class List<Bytecode t> : WithGetter<"$_member", t>;
def SignedVarIntList : List<SignedVarInt>;
def BoolList : List<Bool>;

// Define dialect attribute or type.
class DialectAttrOrType<dag d> {
  // Any members starting with underscore is not fed to create function but
  // treated as purely local variable.
  dag members = d;

  // When needing to specify a custom return type.
  string cType = "";

  // Any post-processing that needs to be done.
  code postProcess = "";
}

class DialectAttribute<dag d> : DialectAttrOrType<d>, AttributeKind {
  let cParser = "succeeded($_reader.readAttribute<$_resultType>($_var))";
  let cBuilder = "get<$_resultType>(context, $_args)";
}
class DialectType<dag d> : DialectAttrOrType<d>, TypeKind {
  let cParser = "succeeded($_reader.readType<$_resultType>($_var))";
  let cBuilder = "get<$_resultType>(context, $_args)";
}

class DialectAttributes<string d> {
  string dialect = d;
  list<DialectAttrOrType> elems;
}

class DialectTypes<string d> {
  string dialect = d;
  list<DialectAttrOrType> elems;
}

def attr;
def type;

// Marker to indicate a skipped attribute or type in the enum. Could either be
// reserved for a future value or for marking a previously used value as dead.
def none;
def ReservedOrDead : DialectAttrOrType<(none)>;

#endif // BYTECODE_BASE

