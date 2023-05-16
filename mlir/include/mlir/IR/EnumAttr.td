//===-- EnumAttr.td - Enum attributes ----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENUMATTR_TD
#define ENUMATTR_TD

include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Enum attribute kinds

// Additional information for an enum attribute case.
class EnumAttrCaseInfo<string sym, int intVal, string strVal> {
  // The C++ enumerant symbol.
  string symbol = sym;

  // The C++ enumerant value.
  // If less than zero, there will be no explicit discriminator values assigned
  // to enumerators in the generated enum class.
  int value = intVal;

  // The string representation of the enumerant. May be the same as symbol.
  string str = strVal;
}

// An enum attribute case stored with IntegerAttr, which has an integer value,
// its representation as a string and a C++ symbol name which may be different.
class IntEnumAttrCaseBase<I intType, string sym, string strVal, int intVal> :
    EnumAttrCaseInfo<sym, intVal, strVal>,
    SignlessIntegerAttrBase<intType, "case " # strVal> {
  let predicate =
    CPred<"::llvm::cast<::mlir::IntegerAttr>($_self).getInt() == " # intVal>;
}

// Cases of integer enum attributes with a specific type. By default, the string
// representation is the same as the C++ symbol name.
class I32EnumAttrCase<string sym, int val, string str = sym>
    : IntEnumAttrCaseBase<I32, sym, str, val>;
class I64EnumAttrCase<string sym, int val, string str = sym>
    : IntEnumAttrCaseBase<I64, sym, str, val>;

// A bit enum case stored with an IntegerAttr. `val` here is *not* the ordinal
// number of a bit that is set. It is an integer value with bits set to match
// the case.
class BitEnumAttrCaseBase<I intType, string sym, int val, string str = sym> :
    EnumAttrCaseInfo<sym, val, str>,
    SignlessIntegerAttrBase<intType, "case " #str>;

class I8BitEnumAttrCase<string sym, int val, string str = sym>
    : BitEnumAttrCaseBase<I8, sym, val, str>;
class I16BitEnumAttrCase<string sym, int val, string str = sym>
    : BitEnumAttrCaseBase<I16, sym, val, str>;
class I32BitEnumAttrCase<string sym, int val, string str = sym>
    : BitEnumAttrCaseBase<I32, sym, val, str>;
class I64BitEnumAttrCase<string sym, int val, string str = sym>
    : BitEnumAttrCaseBase<I64, sym, val, str>;

// The special bit enum case with no bits set (i.e. value = 0).
class I8BitEnumAttrCaseNone<string sym, string str = sym>
    : I8BitEnumAttrCase<sym, 0, str>;
class I16BitEnumAttrCaseNone<string sym, string str = sym>
    : I16BitEnumAttrCase<sym, 0, str>;
class I32BitEnumAttrCaseNone<string sym, string str = sym>
    : I32BitEnumAttrCase<sym, 0, str>;
class I64BitEnumAttrCaseNone<string sym, string str = sym>
    : I64BitEnumAttrCase<sym, 0, str>;

// A bit enum case for a single bit, specified by a bit position.
// The pos argument refers to the index of the bit, and is limited
// to be in the range [0, bitwidth).
class BitEnumAttrCaseBit<I intType, string sym, int pos, string str = sym>
    : BitEnumAttrCaseBase<intType, sym, !shl(1, pos), str> {
  assert !and(!ge(pos, 0), !lt(pos, intType.bitwidth)),
      "bit position larger than underlying storage";
}

class I8BitEnumAttrCaseBit<string sym, int pos, string str = sym>
    : BitEnumAttrCaseBit<I8, sym, pos, str>;
class I16BitEnumAttrCaseBit<string sym, int pos, string str = sym>
    : BitEnumAttrCaseBit<I16, sym, pos, str>;
class I32BitEnumAttrCaseBit<string sym, int pos, string str = sym>
    : BitEnumAttrCaseBit<I32, sym, pos, str>;
class I64BitEnumAttrCaseBit<string sym, int pos, string str = sym>
    : BitEnumAttrCaseBit<I64, sym, pos, str>;

// A bit enum case for a group/list of previously declared cases, providing
// a convenient alias for that group.
class BitEnumAttrCaseGroup<I intType, string sym,
                           list<BitEnumAttrCaseBase> cases, string str = sym>
    : BitEnumAttrCaseBase<intType, sym,
          !foldl(0, cases, value, bitcase, !or(value, bitcase.value)),
          str>;

class I8BitEnumAttrCaseGroup<string sym, list<BitEnumAttrCaseBase> cases,
                              string str = sym>
    : BitEnumAttrCaseGroup<I8, sym, cases, str>;
class I16BitEnumAttrCaseGroup<string sym, list<BitEnumAttrCaseBase> cases,
                              string str = sym>
    : BitEnumAttrCaseGroup<I16, sym, cases, str>;
class I32BitEnumAttrCaseGroup<string sym, list<BitEnumAttrCaseBase> cases,
                              string str = sym>
    : BitEnumAttrCaseGroup<I32, sym, cases, str>;
class I64BitEnumAttrCaseGroup<string sym, list<BitEnumAttrCaseBase> cases,
                              string str = sym>
    : BitEnumAttrCaseGroup<I64, sym, cases, str>;

// Additional information for an enum attribute.
class EnumAttrInfo<
    string name, list<EnumAttrCaseInfo> cases, Attr baseClass> :
      Attr<baseClass.predicate, baseClass.summary> {
  // The C++ enum class name
  string className = name;

  // List of all accepted cases
  list<EnumAttrCaseInfo> enumerants = cases;

  // The following fields are only used by the EnumsGen backend to generate
  // an enum class definition and conversion utility functions.

  // The underlying type for the C++ enum class. An empty string mean the
  // underlying type is not explicitly specified.
  string underlyingType = "";

  // The name of the utility function that converts a value of the underlying
  // type to the corresponding symbol. It will have the following signature:
  //
  // ```c++
  // std::optional<<qualified-enum-class-name>> <fn-name>(<underlying-type>);
  // ```
  string underlyingToSymbolFnName = "symbolize" # name;

  // The name of the utility function that converts a string to the
  // corresponding symbol. It will have the following signature:
  //
  // ```c++
  // std::optional<<qualified-enum-class-name>> <fn-name>(llvm::StringRef);
  // ```
  string stringToSymbolFnName = "symbolize" # name;

  // The name of the utility function that converts a symbol to the
  // corresponding string. It will have the following signature:
  //
  // ```c++
  // <return-type> <fn-name>(<qualified-enum-class-name>);
  // ```
  string symbolToStringFnName = "stringify" # name;
  string symbolToStringFnRetType = "::llvm::StringRef";

  // The name of the utility function that returns the max enum value used
  // within the enum class. It will have the following signature:
  //
  // ```c++
  // static constexpr unsigned <fn-name>();
  // ```
  string maxEnumValFnName = "getMaxEnumValFor" # name;

  // Generate specialized Attribute class
  bit genSpecializedAttr = 1;
  // The underlying Attribute class, which holds the enum value
  Attr baseAttrClass = baseClass;
  // The name of specialized Enum Attribute class
  string specializedAttrClassName = name # Attr;

  // Override Attr class fields for specialized class
  let predicate = !if(genSpecializedAttr,
    CPred<"$_self.isa<" # cppNamespace # "::" # specializedAttrClassName # ">()">,
    baseAttrClass.predicate);
  let storageType = !if(genSpecializedAttr,
    cppNamespace # "::" # specializedAttrClassName,
    baseAttrClass.storageType);
  let returnType = !if(genSpecializedAttr,
    cppNamespace # "::" # className,
    baseAttrClass.returnType);
  let constBuilderCall = !if(genSpecializedAttr,
    cppNamespace # "::" # specializedAttrClassName # "::get($_builder.getContext(), $0)",
    baseAttrClass.constBuilderCall);
  let valueType = baseAttrClass.valueType;

  // C++ type wrapped by attribute
  string cppType = cppNamespace # "::" # className;

  // Parser and printer code used by the EnumParameter class, to be provided by
  // derived classes
  string parameterParser = ?;
  string parameterPrinter = ?;
}

// An enum attribute backed by IntegerAttr.
//
// Op attributes of this kind are stored as IntegerAttr. Extra verification will
// be generated on the integer though: only the values of the allowed cases are
// permitted as the integer value.
class IntEnumAttrBase<I intType, list<IntEnumAttrCaseBase> cases, string summary> :
    SignlessIntegerAttrBase<intType, summary> {
  let predicate = And<[
    SignlessIntegerAttrBase<intType, summary>.predicate,
    Or<!foreach(case, cases, case.predicate)>]>;
}

class IntEnumAttr<I intType, string name, string summary,
                  list<IntEnumAttrCaseBase> cases> :
  EnumAttrInfo<name, cases,
    IntEnumAttrBase<intType, cases,
      !if(!empty(summary), "allowed " # intType.summary # " cases: " #
          !interleave(!foreach(case, cases, case.value), ", "),
          summary)>> {
  // Parse a keyword and pass it to `stringToSymbol`. Emit an error if a the
  // symbol is not valid.
  let parameterParser = [{[&]() -> ::mlir::FailureOr<}] # cppType # [{> {
    auto loc = $_parser.getCurrentLocation();
    ::llvm::StringRef enumKeyword;
    if (::mlir::failed($_parser.parseKeyword(&enumKeyword)))
      return ::mlir::failure();
    auto maybeEnum = }] # cppNamespace # "::" #
                          stringToSymbolFnName # [{(enumKeyword);
    if (maybeEnum)
      return *maybeEnum;
    return {(::mlir::LogicalResult)($_parser.emitError(loc) << "expected " }] #
    [{<< "}] # cppType # [{" << " to be one of: " << }] #
    !interleave(!foreach(enum, enumerants, "\"" # enum.str # "\""),
                [{ << ", " << }]) # [{)};
  }()}];
  // Print the enum by calling `symbolToString`.
  let parameterPrinter = "$_printer << " # symbolToStringFnName # "($_self)";
}

class I32EnumAttr<string name, string summary, list<I32EnumAttrCase> cases> :
    IntEnumAttr<I32, name, summary, cases> {
  let underlyingType = "uint32_t";
}
class I64EnumAttr<string name, string summary, list<I64EnumAttrCase> cases> :
    IntEnumAttr<I64, name, summary, cases> {
  let underlyingType = "uint64_t";
}

// A bit enum stored with an IntegerAttr.
//
// Op attributes of this kind are stored as IntegerAttr. Extra verification will
// be generated on the integer to make sure only allowed bits are set. Besides,
// helper methods are generated to parse a string separated with a specified
// delimiter to a symbol and vice versa.
class BitEnumAttrBase<I intType, list<BitEnumAttrCaseBase> cases,
                      string summary>
    : SignlessIntegerAttrBase<intType, summary> {
  let predicate = And<[
    SignlessIntegerAttrBase<intType, summary>.predicate,
    // Make sure we don't have unknown bit set.
    CPred<"!(::llvm::cast<::mlir::IntegerAttr>($_self).getValue().getZExtValue() & (~("
          # !interleave(!foreach(case, cases, case.value # "u"), "|") #
          ")))">
  ]>;
}

class BitEnumAttr<I intType, string name, string summary,
                  list<BitEnumAttrCaseBase> cases>
    : EnumAttrInfo<name, cases, BitEnumAttrBase<intType, cases, summary>> {
  // Determine "valid" bits from enum cases for error checking
  int validBits = !foldl(0, cases, value, bitcase, !or(value, bitcase.value));

  // We need to return a string because we may concatenate symbols for multiple
  // bits together.
  let symbolToStringFnRetType = "std::string";

  // The delimiter used to separate bit enum cases in strings. Only "|" and
  // "," (along with optional spaces) are supported due to the use of the
  // parseSeparatorFn in parameterParser below.
  // Spaces in the separator string are used for printing, but will be optional
  // for parsing.
  string separator = "|";
  assert !or(!ge(!find(separator, "|"), 0), !ge(!find(separator, ","), 0)),
      "separator must contain '|' or ',' for parameter parsing";

  // Parsing function that corresponds to the enum separator. Only
  // "," and "|" are supported by this definition.
  string parseSeparatorFn = !if(!ge(!find(separator, "|"), 0),
                                "parseOptionalVerticalBar",
                                "parseOptionalComma");

  // Parse a keyword and pass it to `stringToSymbol`. Emit an error if a the
  // symbol is not valid.
  let parameterParser = [{[&]() -> ::mlir::FailureOr<}] # cppType # [{> {
    }] # cppType # [{ flags = {};
    auto loc = $_parser.getCurrentLocation();
    ::llvm::StringRef enumKeyword;
    do {
      if (::mlir::failed($_parser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = }] # cppNamespace # "::" #
                            stringToSymbolFnName # [{(enumKeyword);
      if (!maybeEnum) {
          return {(::mlir::LogicalResult)($_parser.emitError(loc) << }] #
              [{"expected " << "}] # cppType # [{" << " to be one of: " << }] #
              !interleave(!foreach(enum, enumerants, "\"" # enum.str # "\""),
                          [{ << ", " << }]) # [{)};
      }
      flags = flags | *maybeEnum;
    } while(::mlir::succeeded($_parser.}] # parseSeparatorFn # [{()));
    return flags;
  }()}];
  // Print the enum by calling `symbolToString`.
  let parameterPrinter = "$_printer << " # symbolToStringFnName # "($_self)";

  // Print the "primary group" only for bits that are members of case groups
  // that have all bits present. When the value is 0, printing will display both
  // both individual bit case names AND the names for all groups that the bit is
  // contained in. When the value is 1, for each bit that is set AND is a member
  // of a group with all bits set, only the "primary group" (i.e. the first
  // group with all bits set in reverse declaration order) will be printed (for
  // conciseness).
  bit printBitEnumPrimaryGroups = 0;
}

class I8BitEnumAttr<string name, string summary,
                     list<BitEnumAttrCaseBase> cases>
    : BitEnumAttr<I8, name, summary, cases> {
  let underlyingType = "uint8_t";
}

class I16BitEnumAttr<string name, string summary,
                     list<BitEnumAttrCaseBase> cases>
    : BitEnumAttr<I16, name, summary, cases> {
  let underlyingType = "uint16_t";
}

class I32BitEnumAttr<string name, string summary,
                     list<BitEnumAttrCaseBase> cases>
    : BitEnumAttr<I32, name, summary, cases> {
  let underlyingType = "uint32_t";
}

class I64BitEnumAttr<string name, string summary,
                     list<BitEnumAttrCaseBase> cases>
    : BitEnumAttr<I64, name, summary, cases> {
  let underlyingType = "uint64_t";
}

// A C++ enum as an attribute parameter. The parameter implements a parser and
// printer for the enum by dispatching calls to `stringToSymbol` and
// `symbolToString`.
class EnumParameter<EnumAttrInfo enumInfo>
    : AttrParameter<enumInfo.cppNamespace # "::" # enumInfo.className,
                    "an enum of type " # enumInfo.className> {
  let parser = enumInfo.parameterParser;
  let printer = enumInfo.parameterPrinter;
}

// An attribute backed by a C++ enum. The attribute contains a single
// parameter `value` whose type is the C++ enum class.
//
// Example:
//
// ```
// def MyEnum : I32EnumAttr<"MyEnum", "a simple enum", [
//                            I32EnumAttrCase<"First", 0, "first">,
//                            I32EnumAttrCase<"Second", 1, "second>]> {
//   let genSpecializedAttr = 0;
// }
//
// def MyEnumAttr : EnumAttr<MyDialect, MyEnum, "enum">;
// ```
//
// By default, the assembly format of the attribute works best with operation
// assembly formats. For example:
//
// ```
// def MyOp : Op<MyDialect, "my_op"> {
//   let arguments = (ins MyEnumAttr:$enum);
//   let assemblyFormat = "$enum attr-dict";
// }
// ```
//
// The op will appear in the IR as `my_dialect.my_op first`. However, the
// generic format of the attribute will be `#my_dialect<"enum first">`. Override
// the attribute's assembly format as required.
class EnumAttr<Dialect dialect, EnumAttrInfo enumInfo, string name = "",
               list <Trait> traits = []>
    : AttrDef<dialect, enumInfo.className, traits> {
  let summary = enumInfo.summary;

  // The backing enumeration.
  EnumAttrInfo enum = enumInfo;

  // Inherit the C++ namespace from the enum.
  let cppNamespace = enumInfo.cppNamespace;

  // Define a constant builder for the attribute to convert from C++ enums.
  let constBuilderCall = cppNamespace # "::" # cppClassName #
                         "::get($_builder.getContext(), $0)";

  // Op attribute getters should return the underlying C++ enum type.
  let returnType = enumInfo.cppNamespace # "::" # enumInfo.className;

  // Convert from attribute to the underlying C++ type in op getters.
  let convertFromStorage = "$_self.getValue()";

  // The enum attribute has one parameter: the C++ enum value.
  let parameters = (ins EnumParameter<enumInfo>:$value);

  // If a mnemonic was provided, use it to generate a custom assembly format.
  let mnemonic = name;

  // The default assembly format for enum attributes. Selected to best work with
  // operation assembly formats.
  let assemblyFormat = "$value";
}

#endif // ENUMATTR_TD
