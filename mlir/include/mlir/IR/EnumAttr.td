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

// A C++ enum as an attribute parameter. The parameter implements a parser and
// printer for the enum by dispatching calls to `stringToSymbol` and
// `symbolToString`.
class EnumParameter<EnumAttrInfo enumInfo>
    : AttrParameter<enumInfo.cppNamespace # "::" # enumInfo.className,
                    "an enum of type " # enumInfo.className> {
  // Parse a keyword and pass it to `stringToSymbol`. Emit an error if a the
  // symbol is not valid.
  let parser = [{[&]() -> ::mlir::FailureOr<}] # cppType # [{> {
    auto loc = $_parser.getCurrentLocation();
    ::llvm::StringRef enumKeyword;
    if (::mlir::failed($_parser.parseKeyword(&enumKeyword)))
      return ::mlir::failure();
    auto maybeEnum = }] # enumInfo.cppNamespace # "::" #
                          enumInfo.stringToSymbolFnName # [{(enumKeyword);
    if (maybeEnum)
      return *maybeEnum;
    return {(::mlir::LogicalResult)$_parser.emitError(loc, "expected }] #
    cppType # [{ to be one of: }] #
    !interleave(!foreach(enum, enumInfo.enumerants, enum.str), ", ") # [{")};
  }()}];
  // Print the enum by calling `symbolToString`.
  let printer = "$_printer << " # enumInfo.symbolToStringFnName # "($_self)";
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
