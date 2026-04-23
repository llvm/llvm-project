//===- AttrOrTypeFormatGen.h - Attr/type format generator -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_GENERATORS_ATTRORTYPEFORMATGEN_H
#define MLIR_TABLEGEN_GENERATORS_ATTRORTYPEFORMATGEN_H

#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Class.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/Generators/FormatGen.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SourceMgr.h"
#include <vector>

namespace mlir {
namespace tblgen {

//===----------------------------------------------------------------------===//
// ParameterElement
//===----------------------------------------------------------------------===//

/// Represents a variable element referring to an attribute or type parameter.
class ParameterElement
    : public VariableElementBase<VariableElement::Parameter> {
public:
  ParameterElement(AttrOrTypeParameter param) : param(param) {}

  /// Get the parameter in the element.
  const AttrOrTypeParameter &getParam() const { return param; }

  /// Indicate if this variable is printed "qualified" (that is it is
  /// prefixed with the `#dialect.mnemonic`).
  bool shouldBeQualified() { return shouldBeQualifiedFlag; }
  void setShouldBeQualified(bool qualified = true) {
    shouldBeQualifiedFlag = qualified;
  }

  /// Returns true if the element contains an optional parameter.
  bool isOptional() const { return param.isOptional(); }

  /// Returns the name of the parameter.
  llvm::StringRef getName() const { return param.getName(); }

  /// Return the code to check whether the parameter is present.
  auto genIsPresent(FmtContext &ctx, const llvm::Twine &self) const {
    assert(isOptional() && "cannot guard on a mandatory parameter");
    std::string valueStr = tgfmt(*param.getDefaultValue(), &ctx).str();
    ctx.addSubst("_lhs", self).addSubst("_rhs", valueStr);
    return tgfmt(getParam().getComparator(), &ctx);
  }

  /// Generate the code to check whether the parameter should be printed.
  MethodBody &genPrintGuard(FmtContext &ctx, MethodBody &os) const;

private:
  bool shouldBeQualifiedFlag = false;
  AttrOrTypeParameter param;
};

//===----------------------------------------------------------------------===//
// ParamsDirective
//===----------------------------------------------------------------------===//

/// Represents a `params` directive that refers to all parameters of an
/// attribute or type.
class ParamsDirective
    : public VectorDirectiveBase<DirectiveElement::Params, ParameterElement *> {
public:
  using Base::Base;

  /// Returns true if there are optional parameters present.
  bool hasOptionalElements() const;
};

//===----------------------------------------------------------------------===//
// StructDirective
//===----------------------------------------------------------------------===//

/// Represents a `struct` directive that generates a struct format.
class StructDirective
    : public VectorDirectiveBase<DirectiveElement::Struct, FormatElement *> {
public:
  using Base::Base;

  /// Returns true if there are optional format elements present.
  bool hasOptionalElements() const;
};

//===----------------------------------------------------------------------===//
// AttrTypeDefFormat
//===----------------------------------------------------------------------===//

/// Holds the parsed assembly format for an attribute or type and generates
/// parser and printer code.
class AttrTypeDefFormat {
public:
  AttrTypeDefFormat(const AttrOrTypeDef &def,
                    std::vector<FormatElement *> &&elements)
      : def(def), elements(std::move(elements)) {}

  virtual ~AttrTypeDefFormat() = default;

  /// Generate the attribute or type parser.
  virtual void genParser(MethodBody &os);
  /// Generate the attribute or type printer.
  virtual void genPrinter(MethodBody &os);

protected:
  /// Generate the parser code for a specific format element.
  void genElementParser(FormatElement *el, FmtContext &ctx, MethodBody &os);
  /// Generate the parser code for a literal.
  void genLiteralParser(llvm::StringRef value, FmtContext &ctx, MethodBody &os,
                        bool isOptional = false);
  /// Generate the parser code for a variable.
  void genVariableParser(ParameterElement *el, FmtContext &ctx, MethodBody &os);
  /// Generate the parser code for a `params` directive.
  virtual void genParamsParser(ParamsDirective *el, FmtContext &ctx,
                               MethodBody &os);
  /// Generate the parser code for a `struct` directive.
  virtual void genStructParser(StructDirective *el, FmtContext &ctx,
                               MethodBody &os);
  /// Generate the parser code for a `custom` directive.
  virtual void genCustomParser(CustomDirective *el, FmtContext &ctx,
                               MethodBody &os, bool isOptional = false);
  /// Generate the parser code for an optional group.
  void genOptionalGroupParser(OptionalElement *el, FmtContext &ctx,
                              MethodBody &os);

  /// Generate the printer code for a specific format element.
  void genElementPrinter(FormatElement *el, FmtContext &ctx, MethodBody &os);
  /// Generate the printer code for a literal.
  void genLiteralPrinter(llvm::StringRef value, FmtContext &ctx,
                         MethodBody &os);
  /// Generate the printer code for a variable.
  void genVariablePrinter(ParameterElement *el, FmtContext &ctx, MethodBody &os,
                          bool skipGuard = false);
  /// Generate a printer for comma-separated format elements.
  void genCommaSeparatedPrinter(
      llvm::ArrayRef<FormatElement *> params, FmtContext &ctx, MethodBody &os,
      llvm::function_ref<void(FormatElement *)> extra,
      llvm::function_ref<void(FormatElement *)> extraPost = nullptr);
  /// Generate the printer code for a `params` directive.
  virtual void genParamsPrinter(ParamsDirective *el, FmtContext &ctx,
                                MethodBody &os);
  /// Generate the printer code for a `struct` directive.
  virtual void genStructPrinter(StructDirective *el, FmtContext &ctx,
                                MethodBody &os);
  /// Generate the printer code for a `custom` directive.
  virtual void genCustomPrinter(CustomDirective *el, FmtContext &ctx,
                                MethodBody &os);
  /// Generate the printer code for an optional group.
  void genOptionalGroupPrinter(OptionalElement *el, FmtContext &ctx,
                               MethodBody &os);
  /// Generate a printer (or space eraser) for a whitespace element.
  void genWhitespacePrinter(WhitespaceElement *el, FmtContext &ctx,
                            MethodBody &os);

  /// The ODS definition of the attribute or type whose format is being used to
  /// generate a parser and printer.
  const AttrOrTypeDef &def;
  /// The list of top-level format elements returned by the assembly format
  /// parser.
  std::vector<FormatElement *> elements;

  /// Flags for printing spaces.
  bool shouldEmitSpace = false;
  bool lastWasPunctuation = false;
};

//===----------------------------------------------------------------------===//
// AttrTypeDefFormatParser
//===----------------------------------------------------------------------===//

/// Parser for attribute and type assembly formats.
class AttrTypeDefFormatParser : public FormatParser {
public:
  AttrTypeDefFormatParser(llvm::SourceMgr &mgr, const AttrOrTypeDef &def)
      : FormatParser(mgr, def.getLoc()[0]), def(def),
        seenParams(def.getNumParameters()) {}

  /// Parse the attribute or type format and create the format elements.
  FailureOr<AttrTypeDefFormat> parse();

protected:
  /// Verify the parsed elements.
  LogicalResult verify(SMLoc loc,
                       llvm::ArrayRef<FormatElement *> elements) override;
  /// Verify the elements of a custom directive.
  LogicalResult verifyCustomDirectiveArguments(
      SMLoc loc, llvm::ArrayRef<FormatElement *> arguments) override;
  /// Verify the elements of an optional group.
  LogicalResult
  verifyOptionalGroupElements(SMLoc loc,
                              llvm::ArrayRef<FormatElement *> elements,
                              FormatElement *anchor) override;
  /// Verify the arguments to a struct directive.
  LogicalResult
  verifyStructArguments(SMLoc loc, llvm::ArrayRef<FormatElement *> arguments);

  LogicalResult markQualified(SMLoc loc, FormatElement *element) override;

  /// Parse an attribute or type variable.
  FailureOr<FormatElement *> parseVariableImpl(SMLoc loc, llvm::StringRef name,
                                               Context ctx) override;
  /// Parse an attribute or type format directive.
  FailureOr<FormatElement *>
  parseDirectiveImpl(SMLoc loc, FormatToken::Kind kind, Context ctx) override;

private:
  /// Parse a `params` directive.
  FailureOr<FormatElement *> parseParamsDirective(SMLoc loc, Context ctx);
  /// Parse a `struct` directive.
  FailureOr<FormatElement *> parseStructDirective(SMLoc loc, Context ctx);

  /// Attribute or type tablegen def.
  const AttrOrTypeDef &def;

  /// Seen attribute or type parameters.
  llvm::BitVector seenParams;
};

//===----------------------------------------------------------------------===//
// Interface
//===----------------------------------------------------------------------===//

/// Generate a parser and printer based on a custom assembly format for an
/// attribute or type. If fatalOnError is true, a parse failure is a fatal
/// error; otherwise it is silently ignored.
void generateAttrOrTypeFormat(const AttrOrTypeDef &def, MethodBody &parser,
                              MethodBody &printer, bool fatalOnError = true);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_GENERATORS_ATTRORTYPEFORMATGEN_H
