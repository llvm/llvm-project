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
include "mlir/IR/Properties.td"

//===----------------------------------------------------------------------===//
// Enum attribute kinds
//===----------------------------------------------------------------------===//

// Additional information for an enum case.
class EnumCase<string sym, int intVal, string strVal, int widthVal> {
  // The C++ enumerant symbol.
  string symbol = sym;

  // The C++ enumerant value.
  // If less than zero, there will be no explicit discriminator values assigned
  // to enumerators in the generated enum class.
  int value = intVal;

  // The string representation of the enumerant. May be the same as symbol.
  string str = strVal;

  // The bitwidth of the enum.
  int width = widthVal;
}

// An enum attribute case stored with IntegerAttr, which has an integer value,
// its representation as a string and a C++ symbol name which may be different.
// Not needed when using the newer `EnumCase` form for defining enum cases.
class IntEnumAttrCaseBase<I intType, string sym, string strVal, int intVal> :
    EnumCase<sym, intVal, strVal, intType.bitwidth>,
    SignlessIntegerAttrBase<intType, "case " # strVal> {
  let predicate = CPred<[{
    ::llvm::cast<::mlir::IntegerAttr>($_self).getValue().eq(::llvm::APInt(}]
      # intType.bitwidth # ", "
      # intVal # 
    "))">;
}

// Cases of integer enums with a specific type. By default, the string
// representation is the same as the C++ symbol name.
class I32EnumCase<string sym, int val, string str = sym>
  : EnumCase<sym, val, str, 32>;
class I64EnumCase<string sym, int val, string str = sym>
  : EnumCase<sym, val, str, 64>;

// Cases of integer enum attributes with a specific type. By default, the string
// representation is the same as the C++ symbol name. These forms
// are not needed when using the newer `EnumCase` form.
class I32EnumAttrCase<string sym, int val, string str = sym>
    : IntEnumAttrCaseBase<I32, sym, str, val>;
class I64EnumAttrCase<string sym, int val, string str = sym>
    : IntEnumAttrCaseBase<I64, sym, str, val>;

// A bit enum case. `val` here is *not* the ordinal number of a bit
// that is set. It is an integer value with bits set to match the case.
class BitEnumCaseBase<string sym, int val, string str, int width> :
    EnumCase<sym, val, str, width>;
// Bit enum attr cases. The string representation is the same as the C++ symbol
// name unless otherwise specified.
class I8BitEnumCase<string sym, int val, string str = sym>
  : BitEnumCaseBase<sym, val, str, 8>;
class I16BitEnumCase<string sym, int val, string str = sym>
  : BitEnumCaseBase<sym, val, str, 16>;
class I32BitEnumCase<string sym, int val, string str = sym>
  : BitEnumCaseBase<sym, val, str, 32>;
class I64BitEnumCase<string sym, int val, string str = sym>
  : BitEnumCaseBase<sym, val, str, 64>;

// A form of `BitEnumCaseBase` that also inherits from `Attr` and encodes
// the width of the enum, which was defined when enums were always
// stored in attributes.
class BitEnumAttrCaseBase<I intType, string sym, int val, string str = sym> :
    BitEnumCaseBase<sym, val, str, intType.bitwidth>,
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
class BitEnumCaseNone<string sym, string str, int width>
    : BitEnumCaseBase<sym, 0, str, width>;

class I8BitEnumCaseNone<string sym, string str = sym>
  : BitEnumCaseNone<sym, str, 8>;
class I16BitEnumCaseNone<string sym, string str = sym>
  : BitEnumCaseNone<sym, str, 16>;
class I32BitEnumCaseNone<string sym, string str = sym>
  : BitEnumCaseNone<sym, str, 32>;
class I64BitEnumCaseNone<string sym, string str = sym>
  : BitEnumCaseNone<sym, str, 64>;

// Older forms, used when enums were necessarily attributes.
class I8BitEnumAttrCaseNone<string sym, string str = sym>
    : I8BitEnumAttrCase<sym, 0, str>;
class I16BitEnumAttrCaseNone<string sym, string str = sym>
    : I16BitEnumAttrCase<sym, 0, str>;
class I32BitEnumAttrCaseNone<string sym, string str = sym>
    : I32BitEnumAttrCase<sym, 0, str>;
class I64BitEnumAttrCaseNone<string sym, string str = sym>
    : I64BitEnumAttrCase<sym, 0, str>;

// A bit enum case for a single bit, specified by a bit position `pos`.
// The `pos` argument refers to the index of the bit, and is limited
// to be in the range [0, width).
class BitEnumCaseBit<string sym, int pos, string str, int width>
    : BitEnumCaseBase<sym, !shl(1, pos), str, width> {
  assert !and(!ge(pos, 0), !lt(pos, width)),
      "bit position larger than underlying storage";
}

class I8BitEnumCaseBit<string sym, int pos, string str = sym>
    : BitEnumCaseBit<sym, pos, str, 8>;
class I16BitEnumCaseBit<string sym, int pos, string str = sym>
    : BitEnumCaseBit<sym, pos, str, 16>;
class I32BitEnumCaseBit<string sym, int pos, string str = sym>
    : BitEnumCaseBit<sym, pos, str, 32>;
class I64BitEnumCaseBit<string sym, int pos, string str = sym>
    : BitEnumCaseBit<sym, pos, str, 64>;

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
class BitEnumCaseGroup<string sym, list<BitEnumCaseBase> cases, string str = sym>
    : BitEnumCaseBase<sym,
      !foldl(0, cases, value, bitcase, !or(value, bitcase.value)),
      str, !head(cases).width>;

// The attribute-only form of `BitEnumCaseGroup`.
class BitEnumAttrCaseGroup<I intType, string sym,
                           list<BitEnumCaseBase> cases, string str = sym>
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

// Information describing an enum and the functions that should be generated for it.
class EnumInfo<string name, string summaryValue, list<EnumCase> cases, int width> {
  string summary = summaryValue;
  // Generate a description of this enums members for the MLIR docs.
  string description =
        "Enum cases:\n" # !interleave(
          !foreach(case, cases,
              "* " # case.str  # " (`" # case.symbol # "`)"), "\n");

  // The C++ namespace for this enum
  string cppNamespace = "";

  // The C++ enum class name
  string className = name;

  // C++ type wrapped by attribute
  string cppType = cppNamespace # "::" # className;

  // List of all accepted cases
  list<EnumCase> enumerants = cases;

  // The following fields are only used by the EnumsGen backend to generate
  // an enum class definition and conversion utility functions.

  // The bitwidth underlying the class
  int bitwidth = width;

  // The underlying type for the C++ enum class. An empty string mean the
  // underlying type is not explicitly specified.
  string underlyingType = "uint" # width # "_t";

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
}

// A wrapper around `EnumInfo` that also makes the Enum an attribute
// if `genSeecializedAttr` is 1 (though `EnumAttr` is the preferred means
// to accomplish this) or declares that the enum will be stored in an attribute.
class EnumAttrInfo<
    string name, list<EnumCase> cases, SignlessIntegerAttrBase baseClass> :
      EnumInfo<name, baseClass.summary, cases, !cast<I>(baseClass.valueType).bitwidth>,
      Attr<baseClass.predicate, baseClass.summary> {

  // Generate specialized Attribute class
  bit genSpecializedAttr = 1;
  // The underlying Attribute class, which holds the enum value
  Attr baseAttrClass = baseClass;
  // The name of specialized Enum Attribute class
  string specializedAttrClassName = name # Attr;

  // Override Attr class fields for specialized class
  let predicate = !if(genSpecializedAttr,
    CPred<"::llvm::isa<" # cppNamespace # "::" # specializedAttrClassName # ">($_self)">,
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

  // Parser and printer code used by the EnumParameter class, to be provided by
  // derived classes
  string parameterParser = ?;
  string parameterPrinter = ?;
}

// An attribute holding a single integer value.
class IntEnum<string name, string summary, list<EnumCase> cases, int width>
    : EnumInfo<name,
      !if(!empty(summary), "allowed i" # width # " cases: " #
          !interleave(!foreach(case, cases, case.value), ", "),
          summary),
      cases, width>;

class I32Enum<string name, string summary, list<EnumCase> cases>
    : IntEnum<name, summary, cases, 32>;
class I64Enum<string name, string summary, list<EnumCase> cases>
    : IntEnum<name, summary, cases, 64>;

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
    std::string enumKeyword;
    if (::mlir::failed($_parser.parseKeywordOrString(&enumKeyword)))
      return ::mlir::failure();
    auto maybeEnum = }] # cppNamespace # "::" #
                          stringToSymbolFnName # [{(enumKeyword);
    if (maybeEnum)
      return *maybeEnum;
    return {(::llvm::LogicalResult)($_parser.emitError(loc) << "expected " }] #
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

// The base mixin for bit enums that are stored as an integer.
// This is used by both BitEnum and BitEnumAttr, which need to have a set of
// extra properties that bit enums have which normal enums don't. However,
// we can't just use BitEnum as a base class of BitEnumAttr, since BitEnumAttr
// also inherits from EnumAttrInfo, causing double inheritance of EnumInfo.
class BitEnumBase<list<BitEnumCaseBase> cases> {
  // Determine "valid" bits from enum cases for error checking
  int validBits = !foldl(0, cases, value, bitcase, !or(value, bitcase.value));

  // The delimiter used to separate bit enum cases in strings. Only "|" and
  // "," (along with optional spaces) are supported due to the use of the
  // parseSeparatorFn in parameterParser below.
  // Spaces in the separator string are used for printing, but will be optional
  // for parsing.
  string separator = "|";
  assert !or(!ge(!find(separator, "|"), 0), !ge(!find(separator, ","), 0)),
      "separator must contain '|' or ',' for parameter parsing";

  // Print the "primary group" only for bits that are members of case groups
  // that have all bits present. When the value is 0, printing will display both
  // both individual bit case names AND the names for all groups that the bit is
  // contained in. When the value is 1, for each bit that is set AND is a member
  // of a group with all bits set, only the "primary group" (i.e. the first
  // group with all bits set in reverse declaration order) will be printed (for
  // conciseness).
  bit printBitEnumPrimaryGroups = 0;

  // 1 if the operator<< for this enum should put quotes around values with
  // multiple entries. Off by default in the general case but on for BitEnumAttrs
  // since that was the original behavior.
  bit printBitEnumQuoted = 0;
}

// A bit enum stored as an integer.
//
// Enums of these kind are staored as an integer. Attributes or properties deriving
// from this enum will have additional verification generated on them to make sure
// only allowed bits are set. Helper methods are generated to parse a sring of enum
// values generated by the specified separator to a symbol and vice versa.
class BitEnum<string name, string summary, list<BitEnumCaseBase> cases, int width>
    : EnumInfo<name, summary, cases, width>, BitEnumBase<cases> {
  // We need to return a string because we may concatenate symbols for multiple
  // bits together.
  let symbolToStringFnRetType = "std::string";
}

class I8BitEnum<string name, string summary,
                     list<BitEnumCaseBase> cases>
    : BitEnum<name, summary, cases, 8>;
class I16BitEnum<string name, string summary,
                     list<BitEnumCaseBase> cases>
    : BitEnum<name, summary, cases, 16>;
class I32BitEnum<string name, string summary,
                     list<BitEnumCaseBase> cases>
    : BitEnum<name, summary, cases, 32>;

class I64BitEnum<string name, string summary,
                     list<BitEnumCaseBase> cases>
    : BitEnum<name, summary, cases, 64>;

// A bit enum stored with an IntegerAttr.
//
// Op attributes of this kind are stored as IntegerAttr. Extra verification will
// be generated on the integer to make sure only allowed bits are set. Besides,
// helper methods are generated to parse a string separated with a specified
// delimiter to a symbol and vice versa.
class BitEnumAttrBase<I intType, list<BitEnumCaseBase> cases,
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
                  list<BitEnumCaseBase> cases>
    : EnumAttrInfo<name, cases, BitEnumAttrBase<intType, cases, summary>>,
      BitEnumBase<cases> {
  // We need to return a string because we may concatenate symbols for multiple
  // bits together.
  let symbolToStringFnRetType = "std::string";

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
    std::string enumKeyword;
    do {
      if (::mlir::failed($_parser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = }] # cppNamespace # "::" #
                            stringToSymbolFnName # [{(enumKeyword);
      if (!maybeEnum) {
          return {(::llvm::LogicalResult)($_parser.emitError(loc) << }] #
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

  // Use old-style operator<< and FieldParser for compatibility
  let printBitEnumQuoted = 1;
}

class I8BitEnumAttr<string name, string summary,
                     list<BitEnumCaseBase> cases>
    : BitEnumAttr<I8, name, summary, cases> {
  let underlyingType = "uint8_t";
}

class I16BitEnumAttr<string name, string summary,
                     list<BitEnumCaseBase> cases>
    : BitEnumAttr<I16, name, summary, cases> {
  let underlyingType = "uint16_t";
}

class I32BitEnumAttr<string name, string summary,
                     list<BitEnumCaseBase> cases>
    : BitEnumAttr<I32, name, summary, cases> {
  let underlyingType = "uint32_t";
}

class I64BitEnumAttr<string name, string summary,
                     list<BitEnumCaseBase> cases>
    : BitEnumAttr<I64, name, summary, cases> {
  let underlyingType = "uint64_t";
}

// A C++ enum as an attribute parameter. The parameter implements a parser and
// printer for the enum by dispatching calls to `stringToSymbol` and
// `symbolToString`.
class EnumParameter<EnumInfo enumInfo>
    : AttrParameter<enumInfo.cppNamespace # "::" # enumInfo.className,
                    "an enum of type " # enumInfo.className> {
  let parser = !if(!isa<EnumAttrInfo>(enumInfo),
    !cast<EnumAttrInfo>(enumInfo).parameterParser, ?);
  let printer = !if(!isa<EnumAttrInfo>(enumInfo),
    !cast<EnumAttrInfo>(enumInfo).parameterPrinter, ?);
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
class EnumAttr<Dialect dialect, EnumInfo enumInfo, string name = "",
               list <Trait> traits = []>
    : AttrDef<dialect, enumInfo.className, traits> {
  let summary = enumInfo.summary;
  let description = enumInfo.description;

  // The backing enumeration.
  EnumInfo enum = enumInfo;

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

// A property wrapping by a C++ enum. This class will automatically create bytecode
// serialization logic for the given enum, as well as arranging for parser and
// printer calls.
class EnumProp<EnumInfo enumInfo> : Property<enumInfo.cppType, enumInfo.summary> {
  EnumInfo enum = enumInfo;

  let description = enum.description;
  let predicate = !if(
    !isa<BitEnumBase>(enum),
    CPred<"(static_cast<" # enum.underlyingType # ">($_self) & ~" # !cast<BitEnumBase>(enum).validBits # ") == 0">,
    Or<!foreach(case, enum.enumerants, CPred<"$_self == " # enum.cppType # "::" # case.symbol>)>);

  let convertFromAttribute = [{
    auto intAttr = ::mlir::dyn_cast_if_present<::mlir::IntegerAttr>($_attr);
    if (!intAttr) {
      return $_diag() << "expected IntegerAttr storage for }] #
        enum.cppType # [{";
    }
    $_storage = static_cast<}] # enum.cppType # [{>(intAttr.getValue().getZExtValue());
    return ::mlir::success();
  }];

  let convertToAttribute = [{
    return ::mlir::IntegerAttr::get(::mlir::IntegerType::get($_ctxt, }] # enum.bitwidth
      # [{), static_cast<}] # enum.underlyingType #[{>($_storage));
  }];

  let writeToMlirBytecode = [{
    $_writer.writeVarInt(static_cast<uint64_t>($_storage));
  }];

  let readFromMlirBytecode = [{
    uint64_t rawValue;
    if (::mlir::failed($_reader.readVarInt(rawValue)))
      return ::mlir::failure();
    if (rawValue > std::numeric_limits<}] # enum.underlyingType # [{>::max())
      return ::mlir::failure();
    $_storage = static_cast<}] # enum.cppType # [{>(rawValue);
  }];

  let optionalParser = [{
    auto value = ::mlir::FieldParser<std::optional<}] # enum.cppType # [{>>::parse($_parser);
    if (::mlir::failed(value))
      return ::mlir::failure();
    if (!(value->has_value()))
      return std::nullopt;
    $_storage = std::move(**value);
  }];
}

// Enum property that can have been (or, if `storeInCustomAttribute` is true, will also
// be stored as) an attribute, in addition to being stored as an integer attribute.
class EnumPropWithAttrForm<EnumInfo enumInfo, Attr attributeForm>
    : EnumProp<enumInfo> {
  Attr attrForm = attributeForm;
  bit storeInCustomAttribute = 0;

  let convertFromAttribute = [{
    auto customAttr = ::mlir::dyn_cast_if_present<}]
    # attrForm.storageType # [{>($_attr);
    if (customAttr) {
      $_storage = customAttr.getValue();
      return ::mlir::success();
    }
    auto intAttr = ::mlir::dyn_cast_if_present<::mlir::IntegerAttr>($_attr);
    if (!intAttr) {
      return $_diag() << "expected }] # attrForm.storageType
      # [{ or IntegerAttr storage for }] # enum.cppType # [{";
    }
    $_storage = static_cast<}] # enum.cppType # [{>(intAttr.getValue().getZExtValue());
    return ::mlir::success();
  }];

  let convertToAttribute = !if(storeInCustomAttribute, [{
    return }] # attrForm.storageType # [{::get($_ctxt, $_storage);
  }], [{
    return ::mlir::IntegerAttr::get(::mlir::IntegerType::get($_ctxt, }] # enumInfo.bitwidth
      # [{), static_cast<}] # enum.underlyingType #[{>($_storage));
  }]);
}

class _namedEnumPropFields<string cppType, string mnemonic> {
  code parser = [{
    if ($_parser.parseKeyword("}] # mnemonic # [{")
        || $_parser.parseLess()) {
      return ::mlir::failure();
    }
    auto parseRes = ::mlir::FieldParser<}] # cppType # [{>::parse($_parser);
    if (::mlir::failed(parseRes) ||
        ::mlir::failed($_parser.parseGreater())) {
      return ::mlir::failure();
    }
    $_storage = *parseRes;
  }];

  code optionalParser = [{
    if ($_parser.parseOptionalKeyword("}] # mnemonic # [{")) {
      return std::nullopt;
    }
    if ($_parser.parseLess()) {
      return ::mlir::failure();
    }
    auto parseRes = ::mlir::FieldParser<}] # cppType # [{>::parse($_parser);
    if (::mlir::failed(parseRes) ||
        ::mlir::failed($_parser.parseGreater())) {
      return ::mlir::failure();
    }
    $_storage = *parseRes;
  }];

  code printer = [{
    $_printer << "}] # mnemonic # [{<" << $_storage << ">";
  }];
}

// An EnumProp which, when printed, is surrounded by mnemonic<>.
// For example, if the enum can be a, b, or c, and the mnemonic is foo,
// the format of this property will be "foo<a>", "foo<b>", or "foo<c>".
class NamedEnumProp<EnumInfo enumInfo, string name>
    : EnumProp<enumInfo> {
  string mnemonic = name;
  let parser = _namedEnumPropFields<enum.cppType, mnemonic>.parser;
  let optionalParser = _namedEnumPropFields<enum.cppType, mnemonic>.optionalParser;
  let printer = _namedEnumPropFields<enum.cppType, mnemonic>.printer;
}

// A `NamedEnumProp` with an attribute form as in `EnumPropWithAttrForm`.
class NamedEnumPropWithAttrForm<EnumInfo enumInfo, string name, Attr attributeForm>
    : EnumPropWithAttrForm<enumInfo, attributeForm> {
  string mnemonic = name;
  let parser = _namedEnumPropFields<enum.cppType, mnemonic>.parser;
  let optionalParser = _namedEnumPropFields<enum.cppType, mnemonic>.optionalParser;
  let printer = _namedEnumPropFields<enumInfo.cppType, mnemonic>.printer;
}

class _symbolToValue<EnumInfo enumInfo, string case> {
  defvar cases =
    !filter(iter, enumInfo.enumerants, !eq(iter.str, case));

  assert !not(!empty(cases)), "failed to find enum-case '" # case # "'";

  // `!empty` check to not cause an error if the cases are empty.
  // The assertion catches the issue later and emits a proper error message.
  string value = enumInfo.cppType # "::"
    # !if(!empty(cases), "", !head(cases).symbol);
}

class _bitSymbolsToValue<EnumInfo bitEnum, string case> {
  assert !isa<BitEnumBase>(bitEnum), "_bitSymbolsToValue not given a bit enum";
  defvar pos = !find(case, "|");

  // Recursive instantiation looking up the symbol before the `|` in
  // enum cases.
  string value = !if(
    !eq(pos, -1), /*baseCase=*/_symbolToValue<bitEnum, case>.value,
    /*rec=*/_symbolToValue<bitEnum, !substr(case, 0, pos)>.value # "|"
    # _bitSymbolsToValue<bitEnum, !substr(case, !add(pos, 1))>.value
  );
}

class ConstantEnumCaseBase<Attr attribute,
    EnumInfo enumInfo, string case>
  : ConstantAttr<attribute,
  !if(!isa<BitEnumBase>(enumInfo),
    _bitSymbolsToValue<enumInfo, case>.value,
    _symbolToValue<enumInfo, case>.value
  )
>;

/// Attribute constraint matching a constant enum case. `attribute` should be
/// one of `EnumInfo` or `EnumAttr` and `symbol` the string representation
/// of an enum case. Multiple enum values of a bit-enum can be combined using
/// `|` as a separator. Note that there mustn't be any whitespace around the
/// separator.
/// This attribute constraint is additionally buildable, making it possible to
/// use it in result patterns.
///
/// Examples:
/// * ConstantEnumCase<Arith_IntegerOverflowAttr, "nsw|nuw">
/// * ConstantEnumCase<Arith_CmpIPredicateAttr, "slt">
class ConstantEnumCase<Attr attribute, string case>
  : ConstantEnumCaseBase<attribute,
    !if(!isa<EnumInfo>(attribute), !cast<EnumInfo>(attribute),
          !cast<EnumAttr>(attribute).enum), case> {
  assert !or(!isa<EnumAttr>(attribute), !isa<EnumInfo>(attribute)),
    "attribute must be one of 'EnumAttr' or 'EnumInfo'";
}

#endif // ENUMATTR_TD
