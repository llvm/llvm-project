//===- AsmPrinter.cpp - MLIR Assembly Printer Implementation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MLIR AsmPrinter class, which is used to implement
// the various print() methods on the core IR objects.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <tuple>

using namespace mlir;
using namespace mlir::detail;

#define DEBUG_TYPE "mlir-asm-printer"

void OperationName::print(raw_ostream &os) const { os << getStringRef(); }

void OperationName::dump() const { print(llvm::errs()); }

//===--------------------------------------------------------------------===//
// AsmParser
//===--------------------------------------------------------------------===//

AsmParser::~AsmParser() = default;
DialectAsmParser::~DialectAsmParser() = default;
OpAsmParser::~OpAsmParser() = default;

MLIRContext *AsmParser::getContext() const { return getBuilder().getContext(); }

//===----------------------------------------------------------------------===//
// DialectAsmPrinter
//===----------------------------------------------------------------------===//

DialectAsmPrinter::~DialectAsmPrinter() = default;

//===----------------------------------------------------------------------===//
// OpAsmPrinter
//===----------------------------------------------------------------------===//

OpAsmPrinter::~OpAsmPrinter() = default;

void OpAsmPrinter::printFunctionalType(Operation *op) {
  auto &os = getStream();
  os << '(';
  llvm::interleaveComma(op->getOperands(), os, [&](Value operand) {
    // Print the types of null values as <<NULL TYPE>>.
    *this << (operand ? operand.getType() : Type());
  });
  os << ") -> ";

  // Print the result list.  We don't parenthesize single result types unless
  // it is a function (avoiding a grammar ambiguity).
  bool wrapped = op->getNumResults() != 1;
  if (!wrapped && op->getResult(0).getType() &&
      llvm::isa<FunctionType>(op->getResult(0).getType()))
    wrapped = true;

  if (wrapped)
    os << '(';

  llvm::interleaveComma(op->getResults(), os, [&](const OpResult &result) {
    // Print the types of null values as <<NULL TYPE>>.
    *this << (result ? result.getType() : Type());
  });

  if (wrapped)
    os << ')';
}

//===----------------------------------------------------------------------===//
// Operation OpAsm interface.
//===----------------------------------------------------------------------===//

/// The OpAsmOpInterface, see OpAsmInterface.td for more details.
#include "mlir/IR/OpAsmInterface.cpp.inc"

LogicalResult
OpAsmDialectInterface::parseResource(AsmParsedResourceEntry &entry) const {
  return entry.emitError() << "unknown 'resource' key '" << entry.getKey()
                           << "' for dialect '" << getDialect()->getNamespace()
                           << "'";
}

//===----------------------------------------------------------------------===//
// OpPrintingFlags
//===----------------------------------------------------------------------===//

namespace {
/// This struct contains command line options that can be used to initialize
/// various bits of the AsmPrinter. This uses a struct wrapper to avoid the need
/// for global command line options.
struct AsmPrinterOptions {
  llvm::cl::opt<int64_t> printElementsAttrWithHexIfLarger{
      "mlir-print-elementsattrs-with-hex-if-larger",
      llvm::cl::desc(
          "Print DenseElementsAttrs with a hex string that have "
          "more elements than the given upper limit (use -1 to disable)")};

  llvm::cl::opt<unsigned> elideElementsAttrIfLarger{
      "mlir-elide-elementsattrs-if-larger",
      llvm::cl::desc("Elide ElementsAttrs with \"...\" that have "
                     "more elements than the given upper limit")};

  llvm::cl::opt<unsigned> elideResourceStringsIfLarger{
      "mlir-elide-resource-strings-if-larger",
      llvm::cl::desc(
          "Elide printing value of resources if string is too long in chars.")};

  llvm::cl::opt<bool> printDebugInfoOpt{
      "mlir-print-debuginfo", llvm::cl::init(false),
      llvm::cl::desc("Print debug info in MLIR output")};

  llvm::cl::opt<bool> printPrettyDebugInfoOpt{
      "mlir-pretty-debuginfo", llvm::cl::init(false),
      llvm::cl::desc("Print pretty debug info in MLIR output")};

  // Use the generic op output form in the operation printer even if the custom
  // form is defined.
  llvm::cl::opt<bool> printGenericOpFormOpt{
      "mlir-print-op-generic", llvm::cl::init(false),
      llvm::cl::desc("Print the generic op form"), llvm::cl::Hidden};

  llvm::cl::opt<bool> assumeVerifiedOpt{
      "mlir-print-assume-verified", llvm::cl::init(false),
      llvm::cl::desc("Skip op verification when using custom printers"),
      llvm::cl::Hidden};

  llvm::cl::opt<bool> printLocalScopeOpt{
      "mlir-print-local-scope", llvm::cl::init(false),
      llvm::cl::desc("Print with local scope and inline information (eliding "
                     "aliases for attributes, types, and locations")};

  llvm::cl::opt<bool> printValueUsers{
      "mlir-print-value-users", llvm::cl::init(false),
      llvm::cl::desc(
          "Print users of operation results and block arguments as a comment")};
};
} // namespace

static llvm::ManagedStatic<AsmPrinterOptions> clOptions;

/// Register a set of useful command-line options that can be used to configure
/// various flags within the AsmPrinter.
void mlir::registerAsmPrinterCLOptions() {
  // Make sure that the options struct has been initialized.
  *clOptions;
}

/// Initialize the printing flags with default supplied by the cl::opts above.
OpPrintingFlags::OpPrintingFlags()
    : printDebugInfoFlag(false), printDebugInfoPrettyFormFlag(false),
      printGenericOpFormFlag(false), skipRegionsFlag(false),
      assumeVerifiedFlag(false), printLocalScope(false),
      printValueUsersFlag(false) {
  // Initialize based upon command line options, if they are available.
  if (!clOptions.isConstructed())
    return;
  if (clOptions->elideElementsAttrIfLarger.getNumOccurrences())
    elementsAttrElementLimit = clOptions->elideElementsAttrIfLarger;
  if (clOptions->elideResourceStringsIfLarger.getNumOccurrences())
    resourceStringCharLimit = clOptions->elideResourceStringsIfLarger;
  printDebugInfoFlag = clOptions->printDebugInfoOpt;
  printDebugInfoPrettyFormFlag = clOptions->printPrettyDebugInfoOpt;
  printGenericOpFormFlag = clOptions->printGenericOpFormOpt;
  assumeVerifiedFlag = clOptions->assumeVerifiedOpt;
  printLocalScope = clOptions->printLocalScopeOpt;
  printValueUsersFlag = clOptions->printValueUsers;
}

/// Enable the elision of large elements attributes, by printing a '...'
/// instead of the element data, when the number of elements is greater than
/// `largeElementLimit`. Note: The IR generated with this option is not
/// parsable.
OpPrintingFlags &
OpPrintingFlags::elideLargeElementsAttrs(int64_t largeElementLimit) {
  elementsAttrElementLimit = largeElementLimit;
  return *this;
}

/// Enable printing of debug information. If 'prettyForm' is set to true,
/// debug information is printed in a more readable 'pretty' form.
OpPrintingFlags &OpPrintingFlags::enableDebugInfo(bool enable,
                                                  bool prettyForm) {
  printDebugInfoFlag = enable;
  printDebugInfoPrettyFormFlag = prettyForm;
  return *this;
}

/// Always print operations in the generic form.
OpPrintingFlags &OpPrintingFlags::printGenericOpForm(bool enable) {
  printGenericOpFormFlag = enable;
  return *this;
}

/// Always skip Regions.
OpPrintingFlags &OpPrintingFlags::skipRegions(bool skip) {
  skipRegionsFlag = skip;
  return *this;
}

/// Do not verify the operation when using custom operation printers.
OpPrintingFlags &OpPrintingFlags::assumeVerified() {
  assumeVerifiedFlag = true;
  return *this;
}

/// Use local scope when printing the operation. This allows for using the
/// printer in a more localized and thread-safe setting, but may not necessarily
/// be identical of what the IR will look like when dumping the full module.
OpPrintingFlags &OpPrintingFlags::useLocalScope() {
  printLocalScope = true;
  return *this;
}

/// Print users of values as comments.
OpPrintingFlags &OpPrintingFlags::printValueUsers() {
  printValueUsersFlag = true;
  return *this;
}

/// Return if the given ElementsAttr should be elided.
bool OpPrintingFlags::shouldElideElementsAttr(ElementsAttr attr) const {
  return elementsAttrElementLimit &&
         *elementsAttrElementLimit < int64_t(attr.getNumElements()) &&
         !llvm::isa<SplatElementsAttr>(attr);
}

/// Return the size limit for printing large ElementsAttr.
std::optional<int64_t> OpPrintingFlags::getLargeElementsAttrLimit() const {
  return elementsAttrElementLimit;
}

/// Return the size limit for printing large ElementsAttr.
std::optional<uint64_t> OpPrintingFlags::getLargeResourceStringLimit() const {
  return resourceStringCharLimit;
}

/// Return if debug information should be printed.
bool OpPrintingFlags::shouldPrintDebugInfo() const {
  return printDebugInfoFlag;
}

/// Return if debug information should be printed in the pretty form.
bool OpPrintingFlags::shouldPrintDebugInfoPrettyForm() const {
  return printDebugInfoPrettyFormFlag;
}

/// Return if operations should be printed in the generic form.
bool OpPrintingFlags::shouldPrintGenericOpForm() const {
  return printGenericOpFormFlag;
}

/// Return if Region should be skipped.
bool OpPrintingFlags::shouldSkipRegions() const { return skipRegionsFlag; }

/// Return if operation verification should be skipped.
bool OpPrintingFlags::shouldAssumeVerified() const {
  return assumeVerifiedFlag;
}

/// Return if the printer should use local scope when dumping the IR.
bool OpPrintingFlags::shouldUseLocalScope() const { return printLocalScope; }

/// Return if the printer should print users of values.
bool OpPrintingFlags::shouldPrintValueUsers() const {
  return printValueUsersFlag;
}

/// Returns true if an ElementsAttr with the given number of elements should be
/// printed with hex.
static bool shouldPrintElementsAttrWithHex(int64_t numElements) {
  // Check to see if a command line option was provided for the limit.
  if (clOptions.isConstructed()) {
    if (clOptions->printElementsAttrWithHexIfLarger.getNumOccurrences()) {
      // -1 is used to disable hex printing.
      if (clOptions->printElementsAttrWithHexIfLarger == -1)
        return false;
      return numElements > clOptions->printElementsAttrWithHexIfLarger;
    }
  }

  // Otherwise, default to printing with hex if the number of elements is >100.
  return numElements > 100;
}

//===----------------------------------------------------------------------===//
// NewLineCounter
//===----------------------------------------------------------------------===//

namespace {
/// This class is a simple formatter that emits a new line when inputted into a
/// stream, that enables counting the number of newlines emitted. This class
/// should be used whenever emitting newlines in the printer.
struct NewLineCounter {
  unsigned curLine = 1;
};

static raw_ostream &operator<<(raw_ostream &os, NewLineCounter &newLine) {
  ++newLine.curLine;
  return os << '\n';
}
} // namespace

//===----------------------------------------------------------------------===//
// AsmPrinter::Impl
//===----------------------------------------------------------------------===//

namespace mlir {
class AsmPrinter::Impl {
public:
  Impl(raw_ostream &os, AsmStateImpl &state);
  explicit Impl(Impl &other) : Impl(other.os, other.state) {}

  /// Returns the output stream of the printer.
  raw_ostream &getStream() { return os; }

  template <typename Container, typename UnaryFunctor>
  inline void interleaveComma(const Container &c, UnaryFunctor eachFn) const {
    llvm::interleaveComma(c, os, eachFn);
  }

  /// This enum describes the different kinds of elision for the type of an
  /// attribute when printing it.
  enum class AttrTypeElision {
    /// The type must not be elided,
    Never,
    /// The type may be elided when it matches the default used in the parser
    /// (for example i64 is the default for integer attributes).
    May,
    /// The type must be elided.
    Must
  };

  /// Print the given attribute or an alias.
  void printAttribute(Attribute attr,
                      AttrTypeElision typeElision = AttrTypeElision::Never);
  /// Print the given attribute without considering an alias.
  void printAttributeImpl(Attribute attr,
                          AttrTypeElision typeElision = AttrTypeElision::Never);

  /// Print the alias for the given attribute, return failure if no alias could
  /// be printed.
  LogicalResult printAlias(Attribute attr);

  /// Print the given type or an alias.
  void printType(Type type);
  /// Print the given type.
  void printTypeImpl(Type type);

  /// Print the alias for the given type, return failure if no alias could
  /// be printed.
  LogicalResult printAlias(Type type);

  /// Print the given location to the stream. If `allowAlias` is true, this
  /// allows for the internal location to use an attribute alias.
  void printLocation(LocationAttr loc, bool allowAlias = false);

  /// Print a reference to the given resource that is owned by the given
  /// dialect.
  void printResourceHandle(const AsmDialectResourceHandle &resource);

  void printAffineMap(AffineMap map);
  void
  printAffineExpr(AffineExpr expr,
                  function_ref<void(unsigned, bool)> printValueName = nullptr);
  void printAffineConstraint(AffineExpr expr, bool isEq);
  void printIntegerSet(IntegerSet set);

  LogicalResult pushCyclicPrinting(const void *opaquePointer);

  void popCyclicPrinting();

protected:
  void printOptionalAttrDict(ArrayRef<NamedAttribute> attrs,
                             ArrayRef<StringRef> elidedAttrs = {},
                             bool withKeyword = false);
  void printNamedAttribute(NamedAttribute attr);
  void printTrailingLocation(Location loc, bool allowAlias = true);
  void printLocationInternal(LocationAttr loc, bool pretty = false,
                             bool isTopLevel = false);

  /// Print a dense elements attribute. If 'allowHex' is true, a hex string is
  /// used instead of individual elements when the elements attr is large.
  void printDenseElementsAttr(DenseElementsAttr attr, bool allowHex);

  /// Print a dense string elements attribute.
  void printDenseStringElementsAttr(DenseStringElementsAttr attr);

  /// Print a dense elements attribute. If 'allowHex' is true, a hex string is
  /// used instead of individual elements when the elements attr is large.
  void printDenseIntOrFPElementsAttr(DenseIntOrFPElementsAttr attr,
                                     bool allowHex);

  /// Print a dense array attribute.
  void printDenseArrayAttr(DenseArrayAttr attr);

  void printDialectAttribute(Attribute attr);
  void printDialectType(Type type);

  /// Print an escaped string, wrapped with "".
  void printEscapedString(StringRef str);

  /// Print a hex string, wrapped with "".
  void printHexString(StringRef str);
  void printHexString(ArrayRef<char> data);

  /// This enum is used to represent the binding strength of the enclosing
  /// context that an AffineExprStorage is being printed in, so we can
  /// intelligently produce parens.
  enum class BindingStrength {
    Weak,   // + and -
    Strong, // All other binary operators.
  };
  void printAffineExprInternal(
      AffineExpr expr, BindingStrength enclosingTightness,
      function_ref<void(unsigned, bool)> printValueName = nullptr);

  /// The output stream for the printer.
  raw_ostream &os;

  /// An underlying assembly printer state.
  AsmStateImpl &state;

  /// A set of flags to control the printer's behavior.
  OpPrintingFlags printerFlags;

  /// A tracker for the number of new lines emitted during printing.
  NewLineCounter newLine;
};
} // namespace mlir

//===----------------------------------------------------------------------===//
// AliasInitializer
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a specific instance of a symbol Alias.
class SymbolAlias {
public:
  SymbolAlias(StringRef name, uint32_t suffixIndex, bool isType,
              bool isDeferrable)
      : name(name), suffixIndex(suffixIndex), isType(isType),
        isDeferrable(isDeferrable) {}

  /// Print this alias to the given stream.
  void print(raw_ostream &os) const {
    os << (isType ? "!" : "#") << name;
    if (suffixIndex)
      os << suffixIndex;
  }

  /// Returns true if this is a type alias.
  bool isTypeAlias() const { return isType; }

  /// Returns true if this alias supports deferred resolution when parsing.
  bool canBeDeferred() const { return isDeferrable; }

private:
  /// The main name of the alias.
  StringRef name;
  /// The suffix index of the alias.
  uint32_t suffixIndex : 30;
  /// A flag indicating whether this alias is for a type.
  bool isType : 1;
  /// A flag indicating whether this alias may be deferred or not.
  bool isDeferrable : 1;
};

/// This class represents a utility that initializes the set of attribute and
/// type aliases, without the need to store the extra information within the
/// main AliasState class or pass it around via function arguments.
class AliasInitializer {
public:
  AliasInitializer(
      DialectInterfaceCollection<OpAsmDialectInterface> &interfaces,
      llvm::BumpPtrAllocator &aliasAllocator)
      : interfaces(interfaces), aliasAllocator(aliasAllocator),
        aliasOS(aliasBuffer) {}

  void initialize(Operation *op, const OpPrintingFlags &printerFlags,
                  llvm::MapVector<const void *, SymbolAlias> &attrTypeToAlias);

  /// Visit the given attribute to see if it has an alias. `canBeDeferred` is
  /// set to true if the originator of this attribute can resolve the alias
  /// after parsing has completed (e.g. in the case of operation locations).
  /// `elideType` indicates if the type of the attribute should be skipped when
  /// looking for nested aliases. Returns the maximum alias depth of the
  /// attribute, and the alias index of this attribute.
  std::pair<size_t, size_t> visit(Attribute attr, bool canBeDeferred = false,
                                  bool elideType = false) {
    return visitImpl(attr, aliases, canBeDeferred, elideType);
  }

  /// Visit the given type to see if it has an alias. `canBeDeferred` is
  /// set to true if the originator of this attribute can resolve the alias
  /// after parsing has completed. Returns the maximum alias depth of the type,
  /// and the alias index of this type.
  std::pair<size_t, size_t> visit(Type type, bool canBeDeferred = false) {
    return visitImpl(type, aliases, canBeDeferred);
  }

private:
  struct InProgressAliasInfo {
    InProgressAliasInfo()
        : aliasDepth(0), isType(false), canBeDeferred(false) {}
    InProgressAliasInfo(StringRef alias, bool isType, bool canBeDeferred)
        : alias(alias), aliasDepth(1), isType(isType),
          canBeDeferred(canBeDeferred) {}

    bool operator<(const InProgressAliasInfo &rhs) const {
      // Order first by depth, then by attr/type kind, and then by name.
      if (aliasDepth != rhs.aliasDepth)
        return aliasDepth < rhs.aliasDepth;
      if (isType != rhs.isType)
        return isType;
      return alias < rhs.alias;
    }

    /// The alias for the attribute or type, or std::nullopt if the value has no
    /// alias.
    std::optional<StringRef> alias;
    /// The alias depth of this attribute or type, i.e. an indication of the
    /// relative ordering of when to print this alias.
    unsigned aliasDepth : 30;
    /// If this alias represents a type or an attribute.
    bool isType : 1;
    /// If this alias can be deferred or not.
    bool canBeDeferred : 1;
    /// Indices for child aliases.
    SmallVector<size_t> childIndices;
  };

  /// Visit the given attribute or type to see if it has an alias.
  /// `canBeDeferred` is set to true if the originator of this value can resolve
  /// the alias after parsing has completed (e.g. in the case of operation
  /// locations). Returns the maximum alias depth of the value, and its alias
  /// index.
  template <typename T, typename... PrintArgs>
  std::pair<size_t, size_t>
  visitImpl(T value,
            llvm::MapVector<const void *, InProgressAliasInfo> &aliases,
            bool canBeDeferred, PrintArgs &&...printArgs);

  /// Mark the given alias as non-deferrable.
  void markAliasNonDeferrable(size_t aliasIndex);

  /// Try to generate an alias for the provided symbol. If an alias is
  /// generated, the provided alias mapping and reverse mapping are updated.
  template <typename T>
  void generateAlias(T symbol, InProgressAliasInfo &alias, bool canBeDeferred);

  /// Given a collection of aliases and symbols, initialize a mapping from a
  /// symbol to a given alias.
  static void initializeAliases(
      llvm::MapVector<const void *, InProgressAliasInfo> &visitedSymbols,
      llvm::MapVector<const void *, SymbolAlias> &symbolToAlias);

  /// The set of asm interfaces within the context.
  DialectInterfaceCollection<OpAsmDialectInterface> &interfaces;

  /// An allocator used for alias names.
  llvm::BumpPtrAllocator &aliasAllocator;

  /// The set of built aliases.
  llvm::MapVector<const void *, InProgressAliasInfo> aliases;

  /// Storage and stream used when generating an alias.
  SmallString<32> aliasBuffer;
  llvm::raw_svector_ostream aliasOS;
};

/// This class implements a dummy OpAsmPrinter that doesn't print any output,
/// and merely collects the attributes and types that *would* be printed in a
/// normal print invocation so that we can generate proper aliases. This allows
/// for us to generate aliases only for the attributes and types that would be
/// in the output, and trims down unnecessary output.
class DummyAliasOperationPrinter : private OpAsmPrinter {
public:
  explicit DummyAliasOperationPrinter(const OpPrintingFlags &printerFlags,
                                      AliasInitializer &initializer)
      : printerFlags(printerFlags), initializer(initializer) {}

  /// Prints the entire operation with the custom assembly form, if available,
  /// or the generic assembly form, otherwise.
  void printCustomOrGenericOp(Operation *op) override {
    // Visit the operation location.
    if (printerFlags.shouldPrintDebugInfo())
      initializer.visit(op->getLoc(), /*canBeDeferred=*/true);

    // If requested, always print the generic form.
    if (!printerFlags.shouldPrintGenericOpForm()) {
      op->getName().printAssembly(op, *this, /*defaultDialect=*/"");
      return;
    }

    // Otherwise print with the generic assembly form.
    printGenericOp(op);
  }

private:
  /// Print the given operation in the generic form.
  void printGenericOp(Operation *op, bool printOpName = true) override {
    // Consider nested operations for aliases.
    if (!printerFlags.shouldSkipRegions()) {
      for (Region &region : op->getRegions())
        printRegion(region, /*printEntryBlockArgs=*/true,
                    /*printBlockTerminators=*/true);
    }

    // Visit all the types used in the operation.
    for (Type type : op->getOperandTypes())
      printType(type);
    for (Type type : op->getResultTypes())
      printType(type);

    // Consider the attributes of the operation for aliases.
    for (const NamedAttribute &attr : op->getAttrs())
      printAttribute(attr.getValue());
  }

  /// Print the given block. If 'printBlockArgs' is false, the arguments of the
  /// block are not printed. If 'printBlockTerminator' is false, the terminator
  /// operation of the block is not printed.
  void print(Block *block, bool printBlockArgs = true,
             bool printBlockTerminator = true) {
    // Consider the types of the block arguments for aliases if 'printBlockArgs'
    // is set to true.
    if (printBlockArgs) {
      for (BlockArgument arg : block->getArguments()) {
        printType(arg.getType());

        // Visit the argument location.
        if (printerFlags.shouldPrintDebugInfo())
          // TODO: Allow deferring argument locations.
          initializer.visit(arg.getLoc(), /*canBeDeferred=*/false);
      }
    }

    // Consider the operations within this block, ignoring the terminator if
    // requested.
    bool hasTerminator =
        !block->empty() && block->back().hasTrait<OpTrait::IsTerminator>();
    auto range = llvm::make_range(
        block->begin(),
        std::prev(block->end(),
                  (!hasTerminator || printBlockTerminator) ? 0 : 1));
    for (Operation &op : range)
      printCustomOrGenericOp(&op);
  }

  /// Print the given region.
  void printRegion(Region &region, bool printEntryBlockArgs,
                   bool printBlockTerminators,
                   bool printEmptyBlock = false) override {
    if (region.empty())
      return;
    if (printerFlags.shouldSkipRegions()) {
      os << "{...}";
      return;
    }

    auto *entryBlock = &region.front();
    print(entryBlock, printEntryBlockArgs, printBlockTerminators);
    for (Block &b : llvm::drop_begin(region, 1))
      print(&b);
  }

  void printRegionArgument(BlockArgument arg, ArrayRef<NamedAttribute> argAttrs,
                           bool omitType) override {
    printType(arg.getType());
    // Visit the argument location.
    if (printerFlags.shouldPrintDebugInfo())
      // TODO: Allow deferring argument locations.
      initializer.visit(arg.getLoc(), /*canBeDeferred=*/false);
  }

  /// Consider the given type to be printed for an alias.
  void printType(Type type) override { initializer.visit(type); }

  /// Consider the given attribute to be printed for an alias.
  void printAttribute(Attribute attr) override { initializer.visit(attr); }
  void printAttributeWithoutType(Attribute attr) override {
    printAttribute(attr);
  }
  LogicalResult printAlias(Attribute attr) override {
    initializer.visit(attr);
    return success();
  }
  LogicalResult printAlias(Type type) override {
    initializer.visit(type);
    return success();
  }

  /// Consider the given location to be printed for an alias.
  void printOptionalLocationSpecifier(Location loc) override {
    printAttribute(loc);
  }

  /// Print the given set of attributes with names not included within
  /// 'elidedAttrs'.
  void printOptionalAttrDict(ArrayRef<NamedAttribute> attrs,
                             ArrayRef<StringRef> elidedAttrs = {}) override {
    if (attrs.empty())
      return;
    if (elidedAttrs.empty()) {
      for (const NamedAttribute &attr : attrs)
        printAttribute(attr.getValue());
      return;
    }
    llvm::SmallDenseSet<StringRef> elidedAttrsSet(elidedAttrs.begin(),
                                                  elidedAttrs.end());
    for (const NamedAttribute &attr : attrs)
      if (!elidedAttrsSet.contains(attr.getName().strref()))
        printAttribute(attr.getValue());
  }
  void printOptionalAttrDictWithKeyword(
      ArrayRef<NamedAttribute> attrs,
      ArrayRef<StringRef> elidedAttrs = {}) override {
    printOptionalAttrDict(attrs, elidedAttrs);
  }

  /// Return a null stream as the output stream, this will ignore any data fed
  /// to it.
  raw_ostream &getStream() const override { return os; }

  /// The following are hooks of `OpAsmPrinter` that are not necessary for
  /// determining potential aliases.
  void printFloat(const APFloat &) override {}
  void printAffineMapOfSSAIds(AffineMapAttr, ValueRange) override {}
  void printAffineExprOfSSAIds(AffineExpr, ValueRange, ValueRange) override {}
  void printNewline() override {}
  void increaseIndent() override {}
  void decreaseIndent() override {}
  void printOperand(Value) override {}
  void printOperand(Value, raw_ostream &os) override {
    // Users expect the output string to have at least the prefixed % to signal
    // a value name. To maintain this invariant, emit a name even if it is
    // guaranteed to go unused.
    os << "%";
  }
  void printKeywordOrString(StringRef) override {}
  void printString(StringRef) override {}
  void printResourceHandle(const AsmDialectResourceHandle &) override {}
  void printSymbolName(StringRef) override {}
  void printSuccessor(Block *) override {}
  void printSuccessorAndUseList(Block *, ValueRange) override {}
  void shadowRegionArgs(Region &, ValueRange) override {}

  /// The printer flags to use when determining potential aliases.
  const OpPrintingFlags &printerFlags;

  /// The initializer to use when identifying aliases.
  AliasInitializer &initializer;

  /// A dummy output stream.
  mutable llvm::raw_null_ostream os;
};

class DummyAliasDialectAsmPrinter : public DialectAsmPrinter {
public:
  explicit DummyAliasDialectAsmPrinter(AliasInitializer &initializer,
                                       bool canBeDeferred,
                                       SmallVectorImpl<size_t> &childIndices)
      : initializer(initializer), canBeDeferred(canBeDeferred),
        childIndices(childIndices) {}

  /// Print the given attribute/type, visiting any nested aliases that would be
  /// generated as part of printing. Returns the maximum alias depth found while
  /// printing the given value.
  template <typename T, typename... PrintArgs>
  size_t printAndVisitNestedAliases(T value, PrintArgs &&...printArgs) {
    printAndVisitNestedAliasesImpl(value, printArgs...);
    return maxAliasDepth;
  }

private:
  /// Print the given attribute/type, visiting any nested aliases that would be
  /// generated as part of printing.
  void printAndVisitNestedAliasesImpl(Attribute attr, bool elideType) {
    if (!isa<BuiltinDialect>(attr.getDialect())) {
      attr.getDialect().printAttribute(attr, *this);

      // Process the builtin attributes.
    } else if (llvm::isa<AffineMapAttr, DenseArrayAttr, FloatAttr, IntegerAttr,
                         IntegerSetAttr, UnitAttr>(attr)) {
      return;
    } else if (auto distinctAttr = dyn_cast<DistinctAttr>(attr)) {
      printAttribute(distinctAttr.getReferencedAttr());
    } else if (auto dictAttr = dyn_cast<DictionaryAttr>(attr)) {
      for (const NamedAttribute &nestedAttr : dictAttr.getValue()) {
        printAttribute(nestedAttr.getName());
        printAttribute(nestedAttr.getValue());
      }
    } else if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
      for (Attribute nestedAttr : arrayAttr.getValue())
        printAttribute(nestedAttr);
    } else if (auto typeAttr = dyn_cast<TypeAttr>(attr)) {
      printType(typeAttr.getValue());
    } else if (auto locAttr = dyn_cast<OpaqueLoc>(attr)) {
      printAttribute(locAttr.getFallbackLocation());
    } else if (auto locAttr = dyn_cast<NameLoc>(attr)) {
      if (!isa<UnknownLoc>(locAttr.getChildLoc()))
        printAttribute(locAttr.getChildLoc());
    } else if (auto locAttr = dyn_cast<CallSiteLoc>(attr)) {
      printAttribute(locAttr.getCallee());
      printAttribute(locAttr.getCaller());
    } else if (auto locAttr = dyn_cast<FusedLoc>(attr)) {
      if (Attribute metadata = locAttr.getMetadata())
        printAttribute(metadata);
      for (Location nestedLoc : locAttr.getLocations())
        printAttribute(nestedLoc);
    }

    // Don't print the type if we must elide it, or if it is a None type.
    if (!elideType) {
      if (auto typedAttr = llvm::dyn_cast<TypedAttr>(attr)) {
        Type attrType = typedAttr.getType();
        if (!llvm::isa<NoneType>(attrType))
          printType(attrType);
      }
    }
  }
  void printAndVisitNestedAliasesImpl(Type type) {
    if (!isa<BuiltinDialect>(type.getDialect()))
      return type.getDialect().printType(type, *this);

    // Only visit the layout of memref if it isn't the identity.
    if (auto memrefTy = llvm::dyn_cast<MemRefType>(type)) {
      printType(memrefTy.getElementType());
      MemRefLayoutAttrInterface layout = memrefTy.getLayout();
      if (!llvm::isa<AffineMapAttr>(layout) || !layout.isIdentity())
        printAttribute(memrefTy.getLayout());
      if (memrefTy.getMemorySpace())
        printAttribute(memrefTy.getMemorySpace());
      return;
    }

    // For most builtin types, we can simply walk the sub elements.
    auto visitFn = [&](auto element) {
      if (element)
        (void)printAlias(element);
    };
    type.walkImmediateSubElements(visitFn, visitFn);
  }

  /// Consider the given type to be printed for an alias.
  void printType(Type type) override {
    recordAliasResult(initializer.visit(type, canBeDeferred));
  }

  /// Consider the given attribute to be printed for an alias.
  void printAttribute(Attribute attr) override {
    recordAliasResult(initializer.visit(attr, canBeDeferred));
  }
  void printAttributeWithoutType(Attribute attr) override {
    recordAliasResult(
        initializer.visit(attr, canBeDeferred, /*elideType=*/true));
  }
  LogicalResult printAlias(Attribute attr) override {
    printAttribute(attr);
    return success();
  }
  LogicalResult printAlias(Type type) override {
    printType(type);
    return success();
  }

  /// Record the alias result of a child element.
  void recordAliasResult(std::pair<size_t, size_t> aliasDepthAndIndex) {
    childIndices.push_back(aliasDepthAndIndex.second);
    if (aliasDepthAndIndex.first > maxAliasDepth)
      maxAliasDepth = aliasDepthAndIndex.first;
  }

  /// Return a null stream as the output stream, this will ignore any data fed
  /// to it.
  raw_ostream &getStream() const override { return os; }

  /// The following are hooks of `DialectAsmPrinter` that are not necessary for
  /// determining potential aliases.
  void printFloat(const APFloat &) override {}
  void printKeywordOrString(StringRef) override {}
  void printString(StringRef) override {}
  void printSymbolName(StringRef) override {}
  void printResourceHandle(const AsmDialectResourceHandle &) override {}

  LogicalResult pushCyclicPrinting(const void *opaquePointer) override {
    return success(cyclicPrintingStack.insert(opaquePointer));
  }

  void popCyclicPrinting() override { cyclicPrintingStack.pop_back(); }

  /// Stack of potentially cyclic mutable attributes or type currently being
  /// printed.
  SetVector<const void *> cyclicPrintingStack;

  /// The initializer to use when identifying aliases.
  AliasInitializer &initializer;

  /// If the aliases visited by this printer can be deferred.
  bool canBeDeferred;

  /// The indices of child aliases.
  SmallVectorImpl<size_t> &childIndices;

  /// The maximum alias depth found by the printer.
  size_t maxAliasDepth = 0;

  /// A dummy output stream.
  mutable llvm::raw_null_ostream os;
};
} // namespace

/// Sanitize the given name such that it can be used as a valid identifier. If
/// the string needs to be modified in any way, the provided buffer is used to
/// store the new copy,
static StringRef sanitizeIdentifier(StringRef name, SmallString<16> &buffer,
                                    StringRef allowedPunctChars = "$._-",
                                    bool allowTrailingDigit = true) {
  assert(!name.empty() && "Shouldn't have an empty name here");

  auto copyNameToBuffer = [&] {
    for (char ch : name) {
      if (llvm::isAlnum(ch) || allowedPunctChars.contains(ch))
        buffer.push_back(ch);
      else if (ch == ' ')
        buffer.push_back('_');
      else
        buffer.append(llvm::utohexstr((unsigned char)ch));
    }
  };

  // Check to see if this name is valid. If it starts with a digit, then it
  // could conflict with the autogenerated numeric ID's, so add an underscore
  // prefix to avoid problems.
  if (isdigit(name[0])) {
    buffer.push_back('_');
    copyNameToBuffer();
    return buffer;
  }

  // If the name ends with a trailing digit, add a '_' to avoid potential
  // conflicts with autogenerated ID's.
  if (!allowTrailingDigit && isdigit(name.back())) {
    copyNameToBuffer();
    buffer.push_back('_');
    return buffer;
  }

  // Check to see that the name consists of only valid identifier characters.
  for (char ch : name) {
    if (!llvm::isAlnum(ch) && !allowedPunctChars.contains(ch)) {
      copyNameToBuffer();
      return buffer;
    }
  }

  // If there are no invalid characters, return the original name.
  return name;
}

/// Given a collection of aliases and symbols, initialize a mapping from a
/// symbol to a given alias.
void AliasInitializer::initializeAliases(
    llvm::MapVector<const void *, InProgressAliasInfo> &visitedSymbols,
    llvm::MapVector<const void *, SymbolAlias> &symbolToAlias) {
  SmallVector<std::pair<const void *, InProgressAliasInfo>, 0>
      unprocessedAliases = visitedSymbols.takeVector();
  llvm::stable_sort(unprocessedAliases, [](const auto &lhs, const auto &rhs) {
    return lhs.second < rhs.second;
  });

  llvm::StringMap<unsigned> nameCounts;
  for (auto &[symbol, aliasInfo] : unprocessedAliases) {
    if (!aliasInfo.alias)
      continue;
    StringRef alias = *aliasInfo.alias;
    unsigned nameIndex = nameCounts[alias]++;
    symbolToAlias.insert(
        {symbol, SymbolAlias(alias, nameIndex, aliasInfo.isType,
                             aliasInfo.canBeDeferred)});
  }
}

void AliasInitializer::initialize(
    Operation *op, const OpPrintingFlags &printerFlags,
    llvm::MapVector<const void *, SymbolAlias> &attrTypeToAlias) {
  // Use a dummy printer when walking the IR so that we can collect the
  // attributes/types that will actually be used during printing when
  // considering aliases.
  DummyAliasOperationPrinter aliasPrinter(printerFlags, *this);
  aliasPrinter.printCustomOrGenericOp(op);

  // Initialize the aliases.
  initializeAliases(aliases, attrTypeToAlias);
}

template <typename T, typename... PrintArgs>
std::pair<size_t, size_t> AliasInitializer::visitImpl(
    T value, llvm::MapVector<const void *, InProgressAliasInfo> &aliases,
    bool canBeDeferred, PrintArgs &&...printArgs) {
  auto [it, inserted] =
      aliases.insert({value.getAsOpaquePointer(), InProgressAliasInfo()});
  size_t aliasIndex = std::distance(aliases.begin(), it);
  if (!inserted) {
    // Make sure that the alias isn't deferred if we don't permit it.
    if (!canBeDeferred)
      markAliasNonDeferrable(aliasIndex);
    return {static_cast<size_t>(it->second.aliasDepth), aliasIndex};
  }

  // Try to generate an alias for this value.
  generateAlias(value, it->second, canBeDeferred);

  // Print the value, capturing any nested elements that require aliases.
  SmallVector<size_t> childAliases;
  DummyAliasDialectAsmPrinter printer(*this, canBeDeferred, childAliases);
  size_t maxAliasDepth =
      printer.printAndVisitNestedAliases(value, printArgs...);

  // Make sure to recompute `it` in case the map was reallocated.
  it = std::next(aliases.begin(), aliasIndex);

  // If we had sub elements, update to account for the depth.
  it->second.childIndices = std::move(childAliases);
  if (maxAliasDepth)
    it->second.aliasDepth = maxAliasDepth + 1;

  // Propagate the alias depth of the value.
  return {(size_t)it->second.aliasDepth, aliasIndex};
}

void AliasInitializer::markAliasNonDeferrable(size_t aliasIndex) {
  auto it = std::next(aliases.begin(), aliasIndex);

  // If already marked non-deferrable stop the recursion.
  // All children should already be marked non-deferrable as well.
  if (!it->second.canBeDeferred)
    return;

  it->second.canBeDeferred = false;

  // Propagate the non-deferrable flag to any child aliases.
  for (size_t childIndex : it->second.childIndices)
    markAliasNonDeferrable(childIndex);
}

template <typename T>
void AliasInitializer::generateAlias(T symbol, InProgressAliasInfo &alias,
                                     bool canBeDeferred) {
  SmallString<32> nameBuffer;
  for (const auto &interface : interfaces) {
    OpAsmDialectInterface::AliasResult result =
        interface.getAlias(symbol, aliasOS);
    if (result == OpAsmDialectInterface::AliasResult::NoAlias)
      continue;
    nameBuffer = std::move(aliasBuffer);
    assert(!nameBuffer.empty() && "expected valid alias name");
    if (result == OpAsmDialectInterface::AliasResult::FinalAlias)
      break;
  }

  if (nameBuffer.empty())
    return;

  SmallString<16> tempBuffer;
  StringRef name =
      sanitizeIdentifier(nameBuffer, tempBuffer, /*allowedPunctChars=*/"$_-",
                         /*allowTrailingDigit=*/false);
  name = name.copy(aliasAllocator);
  alias = InProgressAliasInfo(name, /*isType=*/std::is_base_of_v<Type, T>,
                              canBeDeferred);
}

//===----------------------------------------------------------------------===//
// AliasState
//===----------------------------------------------------------------------===//

namespace {
/// This class manages the state for type and attribute aliases.
class AliasState {
public:
  // Initialize the internal aliases.
  void
  initialize(Operation *op, const OpPrintingFlags &printerFlags,
             DialectInterfaceCollection<OpAsmDialectInterface> &interfaces);

  /// Get an alias for the given attribute if it has one and print it in `os`.
  /// Returns success if an alias was printed, failure otherwise.
  LogicalResult getAlias(Attribute attr, raw_ostream &os) const;

  /// Get an alias for the given type if it has one and print it in `os`.
  /// Returns success if an alias was printed, failure otherwise.
  LogicalResult getAlias(Type ty, raw_ostream &os) const;

  /// Print all of the referenced aliases that can not be resolved in a deferred
  /// manner.
  void printNonDeferredAliases(AsmPrinter::Impl &p, NewLineCounter &newLine) {
    printAliases(p, newLine, /*isDeferred=*/false);
  }

  /// Print all of the referenced aliases that support deferred resolution.
  void printDeferredAliases(AsmPrinter::Impl &p, NewLineCounter &newLine) {
    printAliases(p, newLine, /*isDeferred=*/true);
  }

private:
  /// Print all of the referenced aliases that support the provided resolution
  /// behavior.
  void printAliases(AsmPrinter::Impl &p, NewLineCounter &newLine,
                    bool isDeferred);

  /// Mapping between attribute/type and alias.
  llvm::MapVector<const void *, SymbolAlias> attrTypeToAlias;

  /// An allocator used for alias names.
  llvm::BumpPtrAllocator aliasAllocator;
};
} // namespace

void AliasState::initialize(
    Operation *op, const OpPrintingFlags &printerFlags,
    DialectInterfaceCollection<OpAsmDialectInterface> &interfaces) {
  AliasInitializer initializer(interfaces, aliasAllocator);
  initializer.initialize(op, printerFlags, attrTypeToAlias);
}

LogicalResult AliasState::getAlias(Attribute attr, raw_ostream &os) const {
  auto it = attrTypeToAlias.find(attr.getAsOpaquePointer());
  if (it == attrTypeToAlias.end())
    return failure();
  it->second.print(os);
  return success();
}

LogicalResult AliasState::getAlias(Type ty, raw_ostream &os) const {
  auto it = attrTypeToAlias.find(ty.getAsOpaquePointer());
  if (it == attrTypeToAlias.end())
    return failure();

  it->second.print(os);
  return success();
}

void AliasState::printAliases(AsmPrinter::Impl &p, NewLineCounter &newLine,
                              bool isDeferred) {
  auto filterFn = [=](const auto &aliasIt) {
    return aliasIt.second.canBeDeferred() == isDeferred;
  };
  for (auto &[opaqueSymbol, alias] :
       llvm::make_filter_range(attrTypeToAlias, filterFn)) {
    alias.print(p.getStream());
    p.getStream() << " = ";

    if (alias.isTypeAlias()) {
      // TODO: Support nested aliases in mutable types.
      Type type = Type::getFromOpaquePointer(opaqueSymbol);
      if (type.hasTrait<TypeTrait::IsMutable>())
        p.getStream() << type;
      else
        p.printTypeImpl(type);
    } else {
      // TODO: Support nested aliases in mutable attributes.
      Attribute attr = Attribute::getFromOpaquePointer(opaqueSymbol);
      if (attr.hasTrait<AttributeTrait::IsMutable>())
        p.getStream() << attr;
      else
        p.printAttributeImpl(attr);
    }

    p.getStream() << newLine;
  }
}

//===----------------------------------------------------------------------===//
// SSANameState
//===----------------------------------------------------------------------===//

namespace {
/// Info about block printing: a number which is its position in the visitation
/// order, and a name that is used to print reference to it, e.g. ^bb42.
struct BlockInfo {
  int ordering;
  StringRef name;
};

/// This class manages the state of SSA value names.
class SSANameState {
public:
  /// A sentinel value used for values with names set.
  enum : unsigned { NameSentinel = ~0U };

  SSANameState(Operation *op, const OpPrintingFlags &printerFlags);
  SSANameState() = default;

  /// Print the SSA identifier for the given value to 'stream'. If
  /// 'printResultNo' is true, it also presents the result number ('#' number)
  /// of this value.
  void printValueID(Value value, bool printResultNo, raw_ostream &stream) const;

  /// Print the operation identifier.
  void printOperationID(Operation *op, raw_ostream &stream) const;

  /// Return the result indices for each of the result groups registered by this
  /// operation, or empty if none exist.
  ArrayRef<int> getOpResultGroups(Operation *op);

  /// Get the info for the given block.
  BlockInfo getBlockInfo(Block *block);

  /// Renumber the arguments for the specified region to the same names as the
  /// SSA values in namesToUse. See OperationPrinter::shadowRegionArgs for
  /// details.
  void shadowRegionArgs(Region &region, ValueRange namesToUse);

private:
  /// Number the SSA values within the given IR unit.
  void numberValuesInRegion(Region &region);
  void numberValuesInBlock(Block &block);
  void numberValuesInOp(Operation &op);

  /// Given a result of an operation 'result', find the result group head
  /// 'lookupValue' and the result of 'result' within that group in
  /// 'lookupResultNo'. 'lookupResultNo' is only filled in if the result group
  /// has more than 1 result.
  void getResultIDAndNumber(OpResult result, Value &lookupValue,
                            std::optional<int> &lookupResultNo) const;

  /// Set a special value name for the given value.
  void setValueName(Value value, StringRef name);

  /// Uniques the given value name within the printer. If the given name
  /// conflicts, it is automatically renamed.
  StringRef uniqueValueName(StringRef name);

  /// This is the value ID for each SSA value. If this returns NameSentinel,
  /// then the valueID has an entry in valueNames.
  DenseMap<Value, unsigned> valueIDs;
  DenseMap<Value, StringRef> valueNames;

  /// When printing users of values, an operation without a result might
  /// be the user. This map holds ids for such operations.
  DenseMap<Operation *, unsigned> operationIDs;

  /// This is a map of operations that contain multiple named result groups,
  /// i.e. there may be multiple names for the results of the operation. The
  /// value of this map are the result numbers that start a result group.
  DenseMap<Operation *, SmallVector<int, 1>> opResultGroups;

  /// This maps blocks to there visitation number in the current region as well
  /// as the string representing their name.
  DenseMap<Block *, BlockInfo> blockNames;

  /// This keeps track of all of the non-numeric names that are in flight,
  /// allowing us to check for duplicates.
  /// Note: the value of the map is unused.
  llvm::ScopedHashTable<StringRef, char> usedNames;
  llvm::BumpPtrAllocator usedNameAllocator;

  /// This is the next value ID to assign in numbering.
  unsigned nextValueID = 0;
  /// This is the next ID to assign to a region entry block argument.
  unsigned nextArgumentID = 0;
  /// This is the next ID to assign when a name conflict is detected.
  unsigned nextConflictID = 0;

  /// These are the printing flags.  They control, eg., whether to print in
  /// generic form.
  OpPrintingFlags printerFlags;
};
} // namespace

SSANameState::SSANameState(Operation *op, const OpPrintingFlags &printerFlags)
    : printerFlags(printerFlags) {
  llvm::SaveAndRestore valueIDSaver(nextValueID);
  llvm::SaveAndRestore argumentIDSaver(nextArgumentID);
  llvm::SaveAndRestore conflictIDSaver(nextConflictID);

  // The naming context includes `nextValueID`, `nextArgumentID`,
  // `nextConflictID` and `usedNames` scoped HashTable. This information is
  // carried from the parent region.
  using UsedNamesScopeTy = llvm::ScopedHashTable<StringRef, char>::ScopeTy;
  using NamingContext =
      std::tuple<Region *, unsigned, unsigned, unsigned, UsedNamesScopeTy *>;

  // Allocator for UsedNamesScopeTy
  llvm::BumpPtrAllocator allocator;

  // Add a scope for the top level operation.
  auto *topLevelNamesScope =
      new (allocator.Allocate<UsedNamesScopeTy>()) UsedNamesScopeTy(usedNames);

  SmallVector<NamingContext, 8> nameContext;
  for (Region &region : op->getRegions())
    nameContext.push_back(std::make_tuple(&region, nextValueID, nextArgumentID,
                                          nextConflictID, topLevelNamesScope));

  numberValuesInOp(*op);

  while (!nameContext.empty()) {
    Region *region;
    UsedNamesScopeTy *parentScope;
    std::tie(region, nextValueID, nextArgumentID, nextConflictID, parentScope) =
        nameContext.pop_back_val();

    // When we switch from one subtree to another, pop the scopes(needless)
    // until the parent scope.
    while (usedNames.getCurScope() != parentScope) {
      usedNames.getCurScope()->~UsedNamesScopeTy();
      assert((usedNames.getCurScope() != nullptr || parentScope == nullptr) &&
             "top level parentScope must be a nullptr");
    }

    // Add a scope for the current region.
    auto *curNamesScope = new (allocator.Allocate<UsedNamesScopeTy>())
        UsedNamesScopeTy(usedNames);

    numberValuesInRegion(*region);

    for (Operation &op : region->getOps())
      for (Region &region : op.getRegions())
        nameContext.push_back(std::make_tuple(&region, nextValueID,
                                              nextArgumentID, nextConflictID,
                                              curNamesScope));
  }

  // Manually remove all the scopes.
  while (usedNames.getCurScope() != nullptr)
    usedNames.getCurScope()->~UsedNamesScopeTy();
}

void SSANameState::printValueID(Value value, bool printResultNo,
                                raw_ostream &stream) const {
  if (!value) {
    stream << "<<NULL VALUE>>";
    return;
  }

  std::optional<int> resultNo;
  auto lookupValue = value;

  // If this is an operation result, collect the head lookup value of the result
  // group and the result number of 'result' within that group.
  if (OpResult result = dyn_cast<OpResult>(value))
    getResultIDAndNumber(result, lookupValue, resultNo);

  auto it = valueIDs.find(lookupValue);
  if (it == valueIDs.end()) {
    stream << "<<UNKNOWN SSA VALUE>>";
    return;
  }

  stream << '%';
  if (it->second != NameSentinel) {
    stream << it->second;
  } else {
    auto nameIt = valueNames.find(lookupValue);
    assert(nameIt != valueNames.end() && "Didn't have a name entry?");
    stream << nameIt->second;
  }

  if (resultNo && printResultNo)
    stream << '#' << *resultNo;
}

void SSANameState::printOperationID(Operation *op, raw_ostream &stream) const {
  auto it = operationIDs.find(op);
  if (it == operationIDs.end()) {
    stream << "<<UNKNOWN OPERATION>>";
  } else {
    stream << '%' << it->second;
  }
}

ArrayRef<int> SSANameState::getOpResultGroups(Operation *op) {
  auto it = opResultGroups.find(op);
  return it == opResultGroups.end() ? ArrayRef<int>() : it->second;
}

BlockInfo SSANameState::getBlockInfo(Block *block) {
  auto it = blockNames.find(block);
  BlockInfo invalidBlock{-1, "INVALIDBLOCK"};
  return it != blockNames.end() ? it->second : invalidBlock;
}

void SSANameState::shadowRegionArgs(Region &region, ValueRange namesToUse) {
  assert(!region.empty() && "cannot shadow arguments of an empty region");
  assert(region.getNumArguments() == namesToUse.size() &&
         "incorrect number of names passed in");
  assert(region.getParentOp()->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
         "only KnownIsolatedFromAbove ops can shadow names");

  SmallVector<char, 16> nameStr;
  for (unsigned i = 0, e = namesToUse.size(); i != e; ++i) {
    auto nameToUse = namesToUse[i];
    if (nameToUse == nullptr)
      continue;
    auto nameToReplace = region.getArgument(i);

    nameStr.clear();
    llvm::raw_svector_ostream nameStream(nameStr);
    printValueID(nameToUse, /*printResultNo=*/true, nameStream);

    // Entry block arguments should already have a pretty "arg" name.
    assert(valueIDs[nameToReplace] == NameSentinel);

    // Use the name without the leading %.
    auto name = StringRef(nameStream.str()).drop_front();

    // Overwrite the name.
    valueNames[nameToReplace] = name.copy(usedNameAllocator);
  }
}

void SSANameState::numberValuesInRegion(Region &region) {
  auto setBlockArgNameFn = [&](Value arg, StringRef name) {
    assert(!valueIDs.count(arg) && "arg numbered multiple times");
    assert(llvm::cast<BlockArgument>(arg).getOwner()->getParent() == &region &&
           "arg not defined in current region");
    setValueName(arg, name);
  };

  if (!printerFlags.shouldPrintGenericOpForm()) {
    if (Operation *op = region.getParentOp()) {
      if (auto asmInterface = dyn_cast<OpAsmOpInterface>(op))
        asmInterface.getAsmBlockArgumentNames(region, setBlockArgNameFn);
    }
  }

  // Number the values within this region in a breadth-first order.
  unsigned nextBlockID = 0;
  for (auto &block : region) {
    // Each block gets a unique ID, and all of the operations within it get
    // numbered as well.
    auto blockInfoIt = blockNames.insert({&block, {-1, ""}});
    if (blockInfoIt.second) {
      // This block hasn't been named through `getAsmBlockArgumentNames`, use
      // default `^bbNNN` format.
      std::string name;
      llvm::raw_string_ostream(name) << "^bb" << nextBlockID;
      blockInfoIt.first->second.name = StringRef(name).copy(usedNameAllocator);
    }
    blockInfoIt.first->second.ordering = nextBlockID++;

    numberValuesInBlock(block);
  }
}

void SSANameState::numberValuesInBlock(Block &block) {
  // Number the block arguments. We give entry block arguments a special name
  // 'arg'.
  bool isEntryBlock = block.isEntryBlock();
  SmallString<32> specialNameBuffer(isEntryBlock ? "arg" : "");
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  for (auto arg : block.getArguments()) {
    if (valueIDs.count(arg))
      continue;
    if (isEntryBlock) {
      specialNameBuffer.resize(strlen("arg"));
      specialName << nextArgumentID++;
    }
    setValueName(arg, specialName.str());
  }

  // Number the operations in this block.
  for (auto &op : block)
    numberValuesInOp(op);
}

void SSANameState::numberValuesInOp(Operation &op) {
  // Function used to set the special result names for the operation.
  SmallVector<int, 2> resultGroups(/*Size=*/1, /*Value=*/0);
  auto setResultNameFn = [&](Value result, StringRef name) {
    assert(!valueIDs.count(result) && "result numbered multiple times");
    assert(result.getDefiningOp() == &op && "result not defined by 'op'");
    setValueName(result, name);

    // Record the result number for groups not anchored at 0.
    if (int resultNo = llvm::cast<OpResult>(result).getResultNumber())
      resultGroups.push_back(resultNo);
  };
  // Operations can customize the printing of block names in OpAsmOpInterface.
  auto setBlockNameFn = [&](Block *block, StringRef name) {
    assert(block->getParentOp() == &op &&
           "getAsmBlockArgumentNames callback invoked on a block not directly "
           "nested under the current operation");
    assert(!blockNames.count(block) && "block numbered multiple times");
    SmallString<16> tmpBuffer{"^"};
    name = sanitizeIdentifier(name, tmpBuffer);
    if (name.data() != tmpBuffer.data()) {
      tmpBuffer.append(name);
      name = tmpBuffer.str();
    }
    name = name.copy(usedNameAllocator);
    blockNames[block] = {-1, name};
  };

  if (!printerFlags.shouldPrintGenericOpForm()) {
    if (OpAsmOpInterface asmInterface = dyn_cast<OpAsmOpInterface>(&op)) {
      asmInterface.getAsmBlockNames(setBlockNameFn);
      asmInterface.getAsmResultNames(setResultNameFn);
    }
  }

  unsigned numResults = op.getNumResults();
  if (numResults == 0) {
    // If value users should be printed, operations with no result need an id.
    if (printerFlags.shouldPrintValueUsers()) {
      if (operationIDs.try_emplace(&op, nextValueID).second)
        ++nextValueID;
    }
    return;
  }
  Value resultBegin = op.getResult(0);

  // If the first result wasn't numbered, give it a default number.
  if (valueIDs.try_emplace(resultBegin, nextValueID).second)
    ++nextValueID;

  // If this operation has multiple result groups, mark it.
  if (resultGroups.size() != 1) {
    llvm::array_pod_sort(resultGroups.begin(), resultGroups.end());
    opResultGroups.try_emplace(&op, std::move(resultGroups));
  }
}

void SSANameState::getResultIDAndNumber(
    OpResult result, Value &lookupValue,
    std::optional<int> &lookupResultNo) const {
  Operation *owner = result.getOwner();
  if (owner->getNumResults() == 1)
    return;
  int resultNo = result.getResultNumber();

  // If this operation has multiple result groups, we will need to find the
  // one corresponding to this result.
  auto resultGroupIt = opResultGroups.find(owner);
  if (resultGroupIt == opResultGroups.end()) {
    // If not, just use the first result.
    lookupResultNo = resultNo;
    lookupValue = owner->getResult(0);
    return;
  }

  // Find the correct index using a binary search, as the groups are ordered.
  ArrayRef<int> resultGroups = resultGroupIt->second;
  const auto *it = llvm::upper_bound(resultGroups, resultNo);
  int groupResultNo = 0, groupSize = 0;

  // If there are no smaller elements, the last result group is the lookup.
  if (it == resultGroups.end()) {
    groupResultNo = resultGroups.back();
    groupSize = static_cast<int>(owner->getNumResults()) - resultGroups.back();
  } else {
    // Otherwise, the previous element is the lookup.
    groupResultNo = *std::prev(it);
    groupSize = *it - groupResultNo;
  }

  // We only record the result number for a group of size greater than 1.
  if (groupSize != 1)
    lookupResultNo = resultNo - groupResultNo;
  lookupValue = owner->getResult(groupResultNo);
}

void SSANameState::setValueName(Value value, StringRef name) {
  // If the name is empty, the value uses the default numbering.
  if (name.empty()) {
    valueIDs[value] = nextValueID++;
    return;
  }

  valueIDs[value] = NameSentinel;
  valueNames[value] = uniqueValueName(name);
}

StringRef SSANameState::uniqueValueName(StringRef name) {
  SmallString<16> tmpBuffer;
  name = sanitizeIdentifier(name, tmpBuffer);

  // Check to see if this name is already unique.
  if (!usedNames.count(name)) {
    name = name.copy(usedNameAllocator);
  } else {
    // Otherwise, we had a conflict - probe until we find a unique name. This
    // is guaranteed to terminate (and usually in a single iteration) because it
    // generates new names by incrementing nextConflictID.
    SmallString<64> probeName(name);
    probeName.push_back('_');
    while (true) {
      probeName += llvm::utostr(nextConflictID++);
      if (!usedNames.count(probeName)) {
        name = probeName.str().copy(usedNameAllocator);
        break;
      }
      probeName.resize(name.size() + 1);
    }
  }

  usedNames.insert(name, char());
  return name;
}

//===----------------------------------------------------------------------===//
// DistinctState
//===----------------------------------------------------------------------===//

namespace {
/// This class manages the state for distinct attributes.
class DistinctState {
public:
  /// Returns a unique identifier for the given distinct attribute.
  uint64_t getId(DistinctAttr distinctAttr);

private:
  uint64_t distinctCounter = 0;
  DenseMap<DistinctAttr, uint64_t> distinctAttrMap;
};
} // namespace

uint64_t DistinctState::getId(DistinctAttr distinctAttr) {
  auto [it, inserted] =
      distinctAttrMap.try_emplace(distinctAttr, distinctCounter);
  if (inserted)
    distinctCounter++;
  return it->getSecond();
}

//===----------------------------------------------------------------------===//
// Resources
//===----------------------------------------------------------------------===//

AsmParsedResourceEntry::~AsmParsedResourceEntry() = default;
AsmResourceBuilder::~AsmResourceBuilder() = default;
AsmResourceParser::~AsmResourceParser() = default;
AsmResourcePrinter::~AsmResourcePrinter() = default;

StringRef mlir::toString(AsmResourceEntryKind kind) {
  switch (kind) {
  case AsmResourceEntryKind::Blob:
    return "blob";
  case AsmResourceEntryKind::Bool:
    return "bool";
  case AsmResourceEntryKind::String:
    return "string";
  }
  llvm_unreachable("unknown AsmResourceEntryKind");
}

AsmResourceParser &FallbackAsmResourceMap::getParserFor(StringRef key) {
  std::unique_ptr<ResourceCollection> &collection = keyToResources[key.str()];
  if (!collection)
    collection = std::make_unique<ResourceCollection>(key);
  return *collection;
}

std::vector<std::unique_ptr<AsmResourcePrinter>>
FallbackAsmResourceMap::getPrinters() {
  std::vector<std::unique_ptr<AsmResourcePrinter>> printers;
  for (auto &it : keyToResources) {
    ResourceCollection *collection = it.second.get();
    auto buildValues = [=](Operation *op, AsmResourceBuilder &builder) {
      return collection->buildResources(op, builder);
    };
    printers.emplace_back(
        AsmResourcePrinter::fromCallable(collection->getName(), buildValues));
  }
  return printers;
}

LogicalResult FallbackAsmResourceMap::ResourceCollection::parseResource(
    AsmParsedResourceEntry &entry) {
  switch (entry.getKind()) {
  case AsmResourceEntryKind::Blob: {
    FailureOr<AsmResourceBlob> blob = entry.parseAsBlob();
    if (failed(blob))
      return failure();
    resources.emplace_back(entry.getKey(), std::move(*blob));
    return success();
  }
  case AsmResourceEntryKind::Bool: {
    FailureOr<bool> value = entry.parseAsBool();
    if (failed(value))
      return failure();
    resources.emplace_back(entry.getKey(), *value);
    break;
  }
  case AsmResourceEntryKind::String: {
    FailureOr<std::string> str = entry.parseAsString();
    if (failed(str))
      return failure();
    resources.emplace_back(entry.getKey(), std::move(*str));
    break;
  }
  }
  return success();
}

void FallbackAsmResourceMap::ResourceCollection::buildResources(
    Operation *op, AsmResourceBuilder &builder) const {
  for (const auto &entry : resources) {
    if (const auto *value = std::get_if<AsmResourceBlob>(&entry.value))
      builder.buildBlob(entry.key, *value);
    else if (const auto *value = std::get_if<bool>(&entry.value))
      builder.buildBool(entry.key, *value);
    else if (const auto *value = std::get_if<std::string>(&entry.value))
      builder.buildString(entry.key, *value);
    else
      llvm_unreachable("unknown AsmResourceEntryKind");
  }
}

//===----------------------------------------------------------------------===//
// AsmState
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
class AsmStateImpl {
public:
  explicit AsmStateImpl(Operation *op, const OpPrintingFlags &printerFlags,
                        AsmState::LocationMap *locationMap)
      : interfaces(op->getContext()), nameState(op, printerFlags),
        printerFlags(printerFlags), locationMap(locationMap) {}
  explicit AsmStateImpl(MLIRContext *ctx, const OpPrintingFlags &printerFlags,
                        AsmState::LocationMap *locationMap)
      : interfaces(ctx), printerFlags(printerFlags), locationMap(locationMap) {}

  /// Initialize the alias state to enable the printing of aliases.
  void initializeAliases(Operation *op) {
    aliasState.initialize(op, printerFlags, interfaces);
  }

  /// Get the state used for aliases.
  AliasState &getAliasState() { return aliasState; }

  /// Get the state used for SSA names.
  SSANameState &getSSANameState() { return nameState; }

  /// Get the state used for distinct attribute identifiers.
  DistinctState &getDistinctState() { return distinctState; }

  /// Return the dialects within the context that implement
  /// OpAsmDialectInterface.
  DialectInterfaceCollection<OpAsmDialectInterface> &getDialectInterfaces() {
    return interfaces;
  }

  /// Return the non-dialect resource printers.
  auto getResourcePrinters() {
    return llvm::make_pointee_range(externalResourcePrinters);
  }

  /// Get the printer flags.
  const OpPrintingFlags &getPrinterFlags() const { return printerFlags; }

  /// Register the location, line and column, within the buffer that the given
  /// operation was printed at.
  void registerOperationLocation(Operation *op, unsigned line, unsigned col) {
    if (locationMap)
      (*locationMap)[op] = std::make_pair(line, col);
  }

  /// Return the referenced dialect resources within the printer.
  DenseMap<Dialect *, SetVector<AsmDialectResourceHandle>> &
  getDialectResources() {
    return dialectResources;
  }

  LogicalResult pushCyclicPrinting(const void *opaquePointer) {
    return success(cyclicPrintingStack.insert(opaquePointer));
  }

  void popCyclicPrinting() { cyclicPrintingStack.pop_back(); }

private:
  /// Collection of OpAsm interfaces implemented in the context.
  DialectInterfaceCollection<OpAsmDialectInterface> interfaces;

  /// A collection of non-dialect resource printers.
  SmallVector<std::unique_ptr<AsmResourcePrinter>> externalResourcePrinters;

  /// A set of dialect resources that were referenced during printing.
  DenseMap<Dialect *, SetVector<AsmDialectResourceHandle>> dialectResources;

  /// The state used for attribute and type aliases.
  AliasState aliasState;

  /// The state used for SSA value names.
  SSANameState nameState;

  /// The state used for distinct attribute identifiers.
  DistinctState distinctState;

  /// Flags that control op output.
  OpPrintingFlags printerFlags;

  /// An optional location map to be populated.
  AsmState::LocationMap *locationMap;

  /// Stack of potentially cyclic mutable attributes or type currently being
  /// printed.
  SetVector<const void *> cyclicPrintingStack;

  // Allow direct access to the impl fields.
  friend AsmState;
};
} // namespace detail
} // namespace mlir

/// Verifies the operation and switches to generic op printing if verification
/// fails. We need to do this because custom print functions may fail for
/// invalid ops.
static OpPrintingFlags verifyOpAndAdjustFlags(Operation *op,
                                              OpPrintingFlags printerFlags) {
  if (printerFlags.shouldPrintGenericOpForm() ||
      printerFlags.shouldAssumeVerified())
    return printerFlags;

  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << ": Verifying operation: "
                          << op->getName() << "\n");

  // Ignore errors emitted by the verifier. We check the thread id to avoid
  // consuming other threads' errors.
  auto parentThreadId = llvm::get_threadid();
  ScopedDiagnosticHandler diagHandler(op->getContext(), [&](Diagnostic &diag) {
    if (parentThreadId == llvm::get_threadid()) {
      LLVM_DEBUG({
        diag.print(llvm::dbgs());
        llvm::dbgs() << "\n";
      });
      return success();
    }
    return failure();
  });
  if (failed(verify(op))) {
    LLVM_DEBUG(llvm::dbgs()
               << DEBUG_TYPE << ": '" << op->getName()
               << "' failed to verify and will be printed in generic form\n");
    printerFlags.printGenericOpForm();
  }

  return printerFlags;
}

AsmState::AsmState(Operation *op, const OpPrintingFlags &printerFlags,
                   LocationMap *locationMap, FallbackAsmResourceMap *map)
    : impl(std::make_unique<AsmStateImpl>(
          op, verifyOpAndAdjustFlags(op, printerFlags), locationMap)) {
  if (map)
    attachFallbackResourcePrinter(*map);
}
AsmState::AsmState(MLIRContext *ctx, const OpPrintingFlags &printerFlags,
                   LocationMap *locationMap, FallbackAsmResourceMap *map)
    : impl(std::make_unique<AsmStateImpl>(ctx, printerFlags, locationMap)) {
  if (map)
    attachFallbackResourcePrinter(*map);
}
AsmState::~AsmState() = default;

const OpPrintingFlags &AsmState::getPrinterFlags() const {
  return impl->getPrinterFlags();
}

void AsmState::attachResourcePrinter(
    std::unique_ptr<AsmResourcePrinter> printer) {
  impl->externalResourcePrinters.emplace_back(std::move(printer));
}

DenseMap<Dialect *, SetVector<AsmDialectResourceHandle>> &
AsmState::getDialectResources() const {
  return impl->getDialectResources();
}

//===----------------------------------------------------------------------===//
// AsmPrinter::Impl
//===----------------------------------------------------------------------===//

AsmPrinter::Impl::Impl(raw_ostream &os, AsmStateImpl &state)
    : os(os), state(state), printerFlags(state.getPrinterFlags()) {}

void AsmPrinter::Impl::printTrailingLocation(Location loc, bool allowAlias) {
  // Check to see if we are printing debug information.
  if (!printerFlags.shouldPrintDebugInfo())
    return;

  os << " ";
  printLocation(loc, /*allowAlias=*/allowAlias);
}

void AsmPrinter::Impl::printLocationInternal(LocationAttr loc, bool pretty,
                                             bool isTopLevel) {
  // If this isn't a top-level location, check for an alias.
  if (!isTopLevel && succeeded(state.getAliasState().getAlias(loc, os)))
    return;

  TypeSwitch<LocationAttr>(loc)
      .Case<OpaqueLoc>([&](OpaqueLoc loc) {
        printLocationInternal(loc.getFallbackLocation(), pretty);
      })
      .Case<UnknownLoc>([&](UnknownLoc loc) {
        if (pretty)
          os << "[unknown]";
        else
          os << "unknown";
      })
      .Case<FileLineColLoc>([&](FileLineColLoc loc) {
        if (pretty)
          os << loc.getFilename().getValue();
        else
          printEscapedString(loc.getFilename());
        os << ':' << loc.getLine() << ':' << loc.getColumn();
      })
      .Case<NameLoc>([&](NameLoc loc) {
        printEscapedString(loc.getName());

        // Print the child if it isn't unknown.
        auto childLoc = loc.getChildLoc();
        if (!llvm::isa<UnknownLoc>(childLoc)) {
          os << '(';
          printLocationInternal(childLoc, pretty);
          os << ')';
        }
      })
      .Case<CallSiteLoc>([&](CallSiteLoc loc) {
        Location caller = loc.getCaller();
        Location callee = loc.getCallee();
        if (!pretty)
          os << "callsite(";
        printLocationInternal(callee, pretty);
        if (pretty) {
          if (llvm::isa<NameLoc>(callee)) {
            if (llvm::isa<FileLineColLoc>(caller)) {
              os << " at ";
            } else {
              os << newLine << " at ";
            }
          } else {
            os << newLine << " at ";
          }
        } else {
          os << " at ";
        }
        printLocationInternal(caller, pretty);
        if (!pretty)
          os << ")";
      })
      .Case<FusedLoc>([&](FusedLoc loc) {
        if (!pretty)
          os << "fused";
        if (Attribute metadata = loc.getMetadata()) {
          os << '<';
          printAttribute(metadata);
          os << '>';
        }
        os << '[';
        interleave(
            loc.getLocations(),
            [&](Location loc) { printLocationInternal(loc, pretty); },
            [&]() { os << ", "; });
        os << ']';
      });
}

/// Print a floating point value in a way that the parser will be able to
/// round-trip losslessly.
static void printFloatValue(const APFloat &apValue, raw_ostream &os) {
  // We would like to output the FP constant value in exponential notation,
  // but we cannot do this if doing so will lose precision.  Check here to
  // make sure that we only output it in exponential format if we can parse
  // the value back and get the same value.
  bool isInf = apValue.isInfinity();
  bool isNaN = apValue.isNaN();
  if (!isInf && !isNaN) {
    SmallString<128> strValue;
    apValue.toString(strValue, /*FormatPrecision=*/6, /*FormatMaxPadding=*/0,
                     /*TruncateZero=*/false);

    // Check to make sure that the stringized number is not some string like
    // "Inf" or NaN, that atof will accept, but the lexer will not.  Check
    // that the string matches the "[-+]?[0-9]" regex.
    assert(((strValue[0] >= '0' && strValue[0] <= '9') ||
            ((strValue[0] == '-' || strValue[0] == '+') &&
             (strValue[1] >= '0' && strValue[1] <= '9'))) &&
           "[-+]?[0-9] regex does not match!");

    // Parse back the stringized version and check that the value is equal
    // (i.e., there is no precision loss).
    if (APFloat(apValue.getSemantics(), strValue).bitwiseIsEqual(apValue)) {
      os << strValue;
      return;
    }

    // If it is not, use the default format of APFloat instead of the
    // exponential notation.
    strValue.clear();
    apValue.toString(strValue);

    // Make sure that we can parse the default form as a float.
    if (strValue.str().contains('.')) {
      os << strValue;
      return;
    }
  }

  // Print special values in hexadecimal format. The sign bit should be included
  // in the literal.
  SmallVector<char, 16> str;
  APInt apInt = apValue.bitcastToAPInt();
  apInt.toString(str, /*Radix=*/16, /*Signed=*/false,
                 /*formatAsCLiteral=*/true);
  os << str;
}

void AsmPrinter::Impl::printLocation(LocationAttr loc, bool allowAlias) {
  if (printerFlags.shouldPrintDebugInfoPrettyForm())
    return printLocationInternal(loc, /*pretty=*/true, /*isTopLevel=*/true);

  os << "loc(";
  if (!allowAlias || failed(printAlias(loc)))
    printLocationInternal(loc, /*pretty=*/false, /*isTopLevel=*/true);
  os << ')';
}

void AsmPrinter::Impl::printResourceHandle(
    const AsmDialectResourceHandle &resource) {
  auto *interface = cast<OpAsmDialectInterface>(resource.getDialect());
  os << interface->getResourceKey(resource);
  state.getDialectResources()[resource.getDialect()].insert(resource);
}

/// Returns true if the given dialect symbol data is simple enough to print in
/// the pretty form. This is essentially when the symbol takes the form:
///   identifier (`<` body `>`)?
static bool isDialectSymbolSimpleEnoughForPrettyForm(StringRef symName) {
  // The name must start with an identifier.
  if (symName.empty() || !isalpha(symName.front()))
    return false;

  // Ignore all the characters that are valid in an identifier in the symbol
  // name.
  symName = symName.drop_while(
      [](char c) { return llvm::isAlnum(c) || c == '.' || c == '_'; });
  if (symName.empty())
    return true;

  // If we got to an unexpected character, then it must be a <>. Check that the
  // rest of the symbol is wrapped within <>.
  return symName.front() == '<' && symName.back() == '>';
}

/// Print the given dialect symbol to the stream.
static void printDialectSymbol(raw_ostream &os, StringRef symPrefix,
                               StringRef dialectName, StringRef symString) {
  os << symPrefix << dialectName;

  // If this symbol name is simple enough, print it directly in pretty form,
  // otherwise, we print it as an escaped string.
  if (isDialectSymbolSimpleEnoughForPrettyForm(symString)) {
    os << '.' << symString;
    return;
  }

  os << '<' << symString << '>';
}

/// Returns true if the given string can be represented as a bare identifier.
static bool isBareIdentifier(StringRef name) {
  // By making this unsigned, the value passed in to isalnum will always be
  // in the range 0-255. This is important when building with MSVC because
  // its implementation will assert. This situation can arise when dealing
  // with UTF-8 multibyte characters.
  if (name.empty() || (!isalpha(name[0]) && name[0] != '_'))
    return false;
  return llvm::all_of(name.drop_front(), [](unsigned char c) {
    return isalnum(c) || c == '_' || c == '$' || c == '.';
  });
}

/// Print the given string as a keyword, or a quoted and escaped string if it
/// has any special or non-printable characters in it.
static void printKeywordOrString(StringRef keyword, raw_ostream &os) {
  // If it can be represented as a bare identifier, write it directly.
  if (isBareIdentifier(keyword)) {
    os << keyword;
    return;
  }

  // Otherwise, output the keyword wrapped in quotes with proper escaping.
  os << "\"";
  printEscapedString(keyword, os);
  os << '"';
}

/// Print the given string as a symbol reference. A symbol reference is
/// represented as a string prefixed with '@'. The reference is surrounded with
/// ""'s and escaped if it has any special or non-printable characters in it.
static void printSymbolReference(StringRef symbolRef, raw_ostream &os) {
  if (symbolRef.empty()) {
    os << "@<<INVALID EMPTY SYMBOL>>";
    return;
  }
  os << '@';
  printKeywordOrString(symbolRef, os);
}

// Print out a valid ElementsAttr that is succinct and can represent any
// potential shape/type, for use when eliding a large ElementsAttr.
//
// We choose to use a dense resource ElementsAttr literal with conspicuous
// content to hopefully alert readers to the fact that this has been elided.
static void printElidedElementsAttr(raw_ostream &os) {
  os << R"(dense_resource<__elided__>)";
}

LogicalResult AsmPrinter::Impl::printAlias(Attribute attr) {
  return state.getAliasState().getAlias(attr, os);
}

LogicalResult AsmPrinter::Impl::printAlias(Type type) {
  return state.getAliasState().getAlias(type, os);
}

void AsmPrinter::Impl::printAttribute(Attribute attr,
                                      AttrTypeElision typeElision) {
  if (!attr) {
    os << "<<NULL ATTRIBUTE>>";
    return;
  }

  // Try to print an alias for this attribute.
  if (succeeded(printAlias(attr)))
    return;
  return printAttributeImpl(attr, typeElision);
}

void AsmPrinter::Impl::printAttributeImpl(Attribute attr,
                                          AttrTypeElision typeElision) {
  if (!isa<BuiltinDialect>(attr.getDialect())) {
    printDialectAttribute(attr);
  } else if (auto opaqueAttr = llvm::dyn_cast<OpaqueAttr>(attr)) {
    printDialectSymbol(os, "#", opaqueAttr.getDialectNamespace(),
                       opaqueAttr.getAttrData());
  } else if (llvm::isa<UnitAttr>(attr)) {
    os << "unit";
    return;
  } else if (auto distinctAttr = llvm::dyn_cast<DistinctAttr>(attr)) {
    os << "distinct[" << state.getDistinctState().getId(distinctAttr) << "]<";
    if (!llvm::isa<UnitAttr>(distinctAttr.getReferencedAttr())) {
      printAttribute(distinctAttr.getReferencedAttr());
    }
    os << '>';
    return;
  } else if (auto dictAttr = llvm::dyn_cast<DictionaryAttr>(attr)) {
    os << '{';
    interleaveComma(dictAttr.getValue(),
                    [&](NamedAttribute attr) { printNamedAttribute(attr); });
    os << '}';

  } else if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr)) {
    Type intType = intAttr.getType();
    if (intType.isSignlessInteger(1)) {
      os << (intAttr.getValue().getBoolValue() ? "true" : "false");

      // Boolean integer attributes always elides the type.
      return;
    }

    // Only print attributes as unsigned if they are explicitly unsigned or are
    // signless 1-bit values.  Indexes, signed values, and multi-bit signless
    // values print as signed.
    bool isUnsigned =
        intType.isUnsignedInteger() || intType.isSignlessInteger(1);
    intAttr.getValue().print(os, !isUnsigned);

    // IntegerAttr elides the type if I64.
    if (typeElision == AttrTypeElision::May && intType.isSignlessInteger(64))
      return;

  } else if (auto floatAttr = llvm::dyn_cast<FloatAttr>(attr)) {
    printFloatValue(floatAttr.getValue(), os);

    // FloatAttr elides the type if F64.
    if (typeElision == AttrTypeElision::May && floatAttr.getType().isF64())
      return;

  } else if (auto strAttr = llvm::dyn_cast<StringAttr>(attr)) {
    printEscapedString(strAttr.getValue());

  } else if (auto arrayAttr = llvm::dyn_cast<ArrayAttr>(attr)) {
    os << '[';
    interleaveComma(arrayAttr.getValue(), [&](Attribute attr) {
      printAttribute(attr, AttrTypeElision::May);
    });
    os << ']';

  } else if (auto affineMapAttr = llvm::dyn_cast<AffineMapAttr>(attr)) {
    os << "affine_map<";
    affineMapAttr.getValue().print(os);
    os << '>';

    // AffineMap always elides the type.
    return;

  } else if (auto integerSetAttr = llvm::dyn_cast<IntegerSetAttr>(attr)) {
    os << "affine_set<";
    integerSetAttr.getValue().print(os);
    os << '>';

    // IntegerSet always elides the type.
    return;

  } else if (auto typeAttr = llvm::dyn_cast<TypeAttr>(attr)) {
    printType(typeAttr.getValue());

  } else if (auto refAttr = llvm::dyn_cast<SymbolRefAttr>(attr)) {
    printSymbolReference(refAttr.getRootReference().getValue(), os);
    for (FlatSymbolRefAttr nestedRef : refAttr.getNestedReferences()) {
      os << "::";
      printSymbolReference(nestedRef.getValue(), os);
    }

  } else if (auto intOrFpEltAttr =
                 llvm::dyn_cast<DenseIntOrFPElementsAttr>(attr)) {
    if (printerFlags.shouldElideElementsAttr(intOrFpEltAttr)) {
      printElidedElementsAttr(os);
    } else {
      os << "dense<";
      printDenseIntOrFPElementsAttr(intOrFpEltAttr, /*allowHex=*/true);
      os << '>';
    }

  } else if (auto strEltAttr = llvm::dyn_cast<DenseStringElementsAttr>(attr)) {
    if (printerFlags.shouldElideElementsAttr(strEltAttr)) {
      printElidedElementsAttr(os);
    } else {
      os << "dense<";
      printDenseStringElementsAttr(strEltAttr);
      os << '>';
    }

  } else if (auto sparseEltAttr = llvm::dyn_cast<SparseElementsAttr>(attr)) {
    if (printerFlags.shouldElideElementsAttr(sparseEltAttr.getIndices()) ||
        printerFlags.shouldElideElementsAttr(sparseEltAttr.getValues())) {
      printElidedElementsAttr(os);
    } else {
      os << "sparse<";
      DenseIntElementsAttr indices = sparseEltAttr.getIndices();
      if (indices.getNumElements() != 0) {
        printDenseIntOrFPElementsAttr(indices, /*allowHex=*/false);
        os << ", ";
        printDenseElementsAttr(sparseEltAttr.getValues(), /*allowHex=*/true);
      }
      os << '>';
    }
  } else if (auto stridedLayoutAttr = llvm::dyn_cast<StridedLayoutAttr>(attr)) {
    stridedLayoutAttr.print(os);
  } else if (auto denseArrayAttr = llvm::dyn_cast<DenseArrayAttr>(attr)) {
    os << "array<";
    printType(denseArrayAttr.getElementType());
    if (!denseArrayAttr.empty()) {
      os << ": ";
      printDenseArrayAttr(denseArrayAttr);
    }
    os << ">";
    return;
  } else if (auto resourceAttr =
                 llvm::dyn_cast<DenseResourceElementsAttr>(attr)) {
    os << "dense_resource<";
    printResourceHandle(resourceAttr.getRawHandle());
    os << ">";
  } else if (auto locAttr = llvm::dyn_cast<LocationAttr>(attr)) {
    printLocation(locAttr);
  } else {
    llvm::report_fatal_error("Unknown builtin attribute");
  }
  // Don't print the type if we must elide it, or if it is a None type.
  if (typeElision != AttrTypeElision::Must) {
    if (auto typedAttr = llvm::dyn_cast<TypedAttr>(attr)) {
      Type attrType = typedAttr.getType();
      if (!llvm::isa<NoneType>(attrType)) {
        os << " : ";
        printType(attrType);
      }
    }
  }
}

/// Print the integer element of a DenseElementsAttr.
static void printDenseIntElement(const APInt &value, raw_ostream &os,
                                 Type type) {
  if (type.isInteger(1))
    os << (value.getBoolValue() ? "true" : "false");
  else
    value.print(os, !type.isUnsignedInteger());
}

static void
printDenseElementsAttrImpl(bool isSplat, ShapedType type, raw_ostream &os,
                           function_ref<void(unsigned)> printEltFn) {
  // Special case for 0-d and splat tensors.
  if (isSplat)
    return printEltFn(0);

  // Special case for degenerate tensors.
  auto numElements = type.getNumElements();
  if (numElements == 0)
    return;

  // We use a mixed-radix counter to iterate through the shape. When we bump a
  // non-least-significant digit, we emit a close bracket. When we next emit an
  // element we re-open all closed brackets.

  // The mixed-radix counter, with radices in 'shape'.
  int64_t rank = type.getRank();
  SmallVector<unsigned, 4> counter(rank, 0);
  // The number of brackets that have been opened and not closed.
  unsigned openBrackets = 0;

  auto shape = type.getShape();
  auto bumpCounter = [&] {
    // Bump the least significant digit.
    ++counter[rank - 1];
    // Iterate backwards bubbling back the increment.
    for (unsigned i = rank - 1; i > 0; --i)
      if (counter[i] >= shape[i]) {
        // Index 'i' is rolled over. Bump (i-1) and close a bracket.
        counter[i] = 0;
        ++counter[i - 1];
        --openBrackets;
        os << ']';
      }
  };

  for (unsigned idx = 0, e = numElements; idx != e; ++idx) {
    if (idx != 0)
      os << ", ";
    while (openBrackets++ < rank)
      os << '[';
    openBrackets = rank;
    printEltFn(idx);
    bumpCounter();
  }
  while (openBrackets-- > 0)
    os << ']';
}

void AsmPrinter::Impl::printDenseElementsAttr(DenseElementsAttr attr,
                                              bool allowHex) {
  if (auto stringAttr = llvm::dyn_cast<DenseStringElementsAttr>(attr))
    return printDenseStringElementsAttr(stringAttr);

  printDenseIntOrFPElementsAttr(llvm::cast<DenseIntOrFPElementsAttr>(attr),
                                allowHex);
}

void AsmPrinter::Impl::printDenseIntOrFPElementsAttr(
    DenseIntOrFPElementsAttr attr, bool allowHex) {
  auto type = attr.getType();
  auto elementType = type.getElementType();

  // Check to see if we should format this attribute as a hex string.
  auto numElements = type.getNumElements();
  if (!attr.isSplat() && allowHex &&
      shouldPrintElementsAttrWithHex(numElements)) {
    ArrayRef<char> rawData = attr.getRawData();
    if (llvm::support::endian::system_endianness() ==
        llvm::support::endianness::big) {
      // Convert endianess in big-endian(BE) machines. `rawData` is BE in BE
      // machines. It is converted here to print in LE format.
      SmallVector<char, 64> outDataVec(rawData.size());
      MutableArrayRef<char> convRawData(outDataVec);
      DenseIntOrFPElementsAttr::convertEndianOfArrayRefForBEmachine(
          rawData, convRawData, type);
      printHexString(convRawData);
    } else {
      printHexString(rawData);
    }

    return;
  }

  if (ComplexType complexTy = llvm::dyn_cast<ComplexType>(elementType)) {
    Type complexElementType = complexTy.getElementType();
    // Note: The if and else below had a common lambda function which invoked
    // printDenseElementsAttrImpl. This lambda was hitting a bug in gcc 9.1,9.2
    // and hence was replaced.
    if (llvm::isa<IntegerType>(complexElementType)) {
      auto valueIt = attr.value_begin<std::complex<APInt>>();
      printDenseElementsAttrImpl(attr.isSplat(), type, os, [&](unsigned index) {
        auto complexValue = *(valueIt + index);
        os << "(";
        printDenseIntElement(complexValue.real(), os, complexElementType);
        os << ",";
        printDenseIntElement(complexValue.imag(), os, complexElementType);
        os << ")";
      });
    } else {
      auto valueIt = attr.value_begin<std::complex<APFloat>>();
      printDenseElementsAttrImpl(attr.isSplat(), type, os, [&](unsigned index) {
        auto complexValue = *(valueIt + index);
        os << "(";
        printFloatValue(complexValue.real(), os);
        os << ",";
        printFloatValue(complexValue.imag(), os);
        os << ")";
      });
    }
  } else if (elementType.isIntOrIndex()) {
    auto valueIt = attr.value_begin<APInt>();
    printDenseElementsAttrImpl(attr.isSplat(), type, os, [&](unsigned index) {
      printDenseIntElement(*(valueIt + index), os, elementType);
    });
  } else {
    assert(llvm::isa<FloatType>(elementType) && "unexpected element type");
    auto valueIt = attr.value_begin<APFloat>();
    printDenseElementsAttrImpl(attr.isSplat(), type, os, [&](unsigned index) {
      printFloatValue(*(valueIt + index), os);
    });
  }
}

void AsmPrinter::Impl::printDenseStringElementsAttr(
    DenseStringElementsAttr attr) {
  ArrayRef<StringRef> data = attr.getRawStringData();
  auto printFn = [&](unsigned index) { printEscapedString(data[index]); };
  printDenseElementsAttrImpl(attr.isSplat(), attr.getType(), os, printFn);
}

void AsmPrinter::Impl::printDenseArrayAttr(DenseArrayAttr attr) {
  Type type = attr.getElementType();
  unsigned bitwidth = type.isInteger(1) ? 8 : type.getIntOrFloatBitWidth();
  unsigned byteSize = bitwidth / 8;
  ArrayRef<char> data = attr.getRawData();

  auto printElementAt = [&](unsigned i) {
    APInt value(bitwidth, 0);
    if (bitwidth) {
      llvm::LoadIntFromMemory(
          value, reinterpret_cast<const uint8_t *>(data.begin() + byteSize * i),
          byteSize);
    }
    // Print the data as-is or as a float.
    if (type.isIntOrIndex()) {
      printDenseIntElement(value, getStream(), type);
    } else {
      APFloat fltVal(llvm::cast<FloatType>(type).getFloatSemantics(), value);
      printFloatValue(fltVal, getStream());
    }
  };
  llvm::interleaveComma(llvm::seq<unsigned>(0, attr.size()), getStream(),
                        printElementAt);
}

void AsmPrinter::Impl::printType(Type type) {
  if (!type) {
    os << "<<NULL TYPE>>";
    return;
  }

  // Try to print an alias for this type.
  if (succeeded(printAlias(type)))
    return;
  return printTypeImpl(type);
}

void AsmPrinter::Impl::printTypeImpl(Type type) {
  TypeSwitch<Type>(type)
      .Case<OpaqueType>([&](OpaqueType opaqueTy) {
        printDialectSymbol(os, "!", opaqueTy.getDialectNamespace(),
                           opaqueTy.getTypeData());
      })
      .Case<IndexType>([&](Type) { os << "index"; })
      .Case<Float8E5M2Type>([&](Type) { os << "f8E5M2"; })
      .Case<Float8E4M3FNType>([&](Type) { os << "f8E4M3FN"; })
      .Case<Float8E5M2FNUZType>([&](Type) { os << "f8E5M2FNUZ"; })
      .Case<Float8E4M3FNUZType>([&](Type) { os << "f8E4M3FNUZ"; })
      .Case<Float8E4M3B11FNUZType>([&](Type) { os << "f8E4M3B11FNUZ"; })
      .Case<BFloat16Type>([&](Type) { os << "bf16"; })
      .Case<Float16Type>([&](Type) { os << "f16"; })
      .Case<FloatTF32Type>([&](Type) { os << "tf32"; })
      .Case<Float32Type>([&](Type) { os << "f32"; })
      .Case<Float64Type>([&](Type) { os << "f64"; })
      .Case<Float80Type>([&](Type) { os << "f80"; })
      .Case<Float128Type>([&](Type) { os << "f128"; })
      .Case<IntegerType>([&](IntegerType integerTy) {
        if (integerTy.isSigned())
          os << 's';
        else if (integerTy.isUnsigned())
          os << 'u';
        os << 'i' << integerTy.getWidth();
      })
      .Case<FunctionType>([&](FunctionType funcTy) {
        os << '(';
        interleaveComma(funcTy.getInputs(), [&](Type ty) { printType(ty); });
        os << ") -> ";
        ArrayRef<Type> results = funcTy.getResults();
        if (results.size() == 1 && !llvm::isa<FunctionType>(results[0])) {
          printType(results[0]);
        } else {
          os << '(';
          interleaveComma(results, [&](Type ty) { printType(ty); });
          os << ')';
        }
      })
      .Case<VectorType>([&](VectorType vectorTy) {
        auto scalableDims = vectorTy.getScalableDims();
        os << "vector<";
        auto vShape = vectorTy.getShape();
        unsigned lastDim = vShape.size();
        unsigned dimIdx = 0;
        for (dimIdx = 0; dimIdx < lastDim; dimIdx++) {
          if (!scalableDims.empty() && scalableDims[dimIdx])
            os << '[';
          os << vShape[dimIdx];
          if (!scalableDims.empty() && scalableDims[dimIdx])
            os << ']';
          os << 'x';
        }
        printType(vectorTy.getElementType());
        os << '>';
      })
      .Case<RankedTensorType>([&](RankedTensorType tensorTy) {
        os << "tensor<";
        for (int64_t dim : tensorTy.getShape()) {
          if (ShapedType::isDynamic(dim))
            os << '?';
          else
            os << dim;
          os << 'x';
        }
        printType(tensorTy.getElementType());
        // Only print the encoding attribute value if set.
        if (tensorTy.getEncoding()) {
          os << ", ";
          printAttribute(tensorTy.getEncoding());
        }
        os << '>';
      })
      .Case<UnrankedTensorType>([&](UnrankedTensorType tensorTy) {
        os << "tensor<*x";
        printType(tensorTy.getElementType());
        os << '>';
      })
      .Case<MemRefType>([&](MemRefType memrefTy) {
        os << "memref<";
        for (int64_t dim : memrefTy.getShape()) {
          if (ShapedType::isDynamic(dim))
            os << '?';
          else
            os << dim;
          os << 'x';
        }
        printType(memrefTy.getElementType());
        MemRefLayoutAttrInterface layout = memrefTy.getLayout();
        if (!llvm::isa<AffineMapAttr>(layout) || !layout.isIdentity()) {
          os << ", ";
          printAttribute(memrefTy.getLayout(), AttrTypeElision::May);
        }
        // Only print the memory space if it is the non-default one.
        if (memrefTy.getMemorySpace()) {
          os << ", ";
          printAttribute(memrefTy.getMemorySpace(), AttrTypeElision::May);
        }
        os << '>';
      })
      .Case<UnrankedMemRefType>([&](UnrankedMemRefType memrefTy) {
        os << "memref<*x";
        printType(memrefTy.getElementType());
        // Only print the memory space if it is the non-default one.
        if (memrefTy.getMemorySpace()) {
          os << ", ";
          printAttribute(memrefTy.getMemorySpace(), AttrTypeElision::May);
        }
        os << '>';
      })
      .Case<ComplexType>([&](ComplexType complexTy) {
        os << "complex<";
        printType(complexTy.getElementType());
        os << '>';
      })
      .Case<TupleType>([&](TupleType tupleTy) {
        os << "tuple<";
        interleaveComma(tupleTy.getTypes(),
                        [&](Type type) { printType(type); });
        os << '>';
      })
      .Case<NoneType>([&](Type) { os << "none"; })
      .Default([&](Type type) { return printDialectType(type); });
}

void AsmPrinter::Impl::printOptionalAttrDict(ArrayRef<NamedAttribute> attrs,
                                             ArrayRef<StringRef> elidedAttrs,
                                             bool withKeyword) {
  // If there are no attributes, then there is nothing to be done.
  if (attrs.empty())
    return;

  // Functor used to print a filtered attribute list.
  auto printFilteredAttributesFn = [&](auto filteredAttrs) {
    // Print the 'attributes' keyword if necessary.
    if (withKeyword)
      os << " attributes";

    // Otherwise, print them all out in braces.
    os << " {";
    interleaveComma(filteredAttrs,
                    [&](NamedAttribute attr) { printNamedAttribute(attr); });
    os << '}';
  };

  // If no attributes are elided, we can directly print with no filtering.
  if (elidedAttrs.empty())
    return printFilteredAttributesFn(attrs);

  // Otherwise, filter out any attributes that shouldn't be included.
  llvm::SmallDenseSet<StringRef> elidedAttrsSet(elidedAttrs.begin(),
                                                elidedAttrs.end());
  auto filteredAttrs = llvm::make_filter_range(attrs, [&](NamedAttribute attr) {
    return !elidedAttrsSet.contains(attr.getName().strref());
  });
  if (!filteredAttrs.empty())
    printFilteredAttributesFn(filteredAttrs);
}
void AsmPrinter::Impl::printNamedAttribute(NamedAttribute attr) {
  // Print the name without quotes if possible.
  ::printKeywordOrString(attr.getName().strref(), os);

  // Pretty printing elides the attribute value for unit attributes.
  if (llvm::isa<UnitAttr>(attr.getValue()))
    return;

  os << " = ";
  printAttribute(attr.getValue());
}

void AsmPrinter::Impl::printDialectAttribute(Attribute attr) {
  auto &dialect = attr.getDialect();

  // Ask the dialect to serialize the attribute to a string.
  std::string attrName;
  {
    llvm::raw_string_ostream attrNameStr(attrName);
    Impl subPrinter(attrNameStr, state);
    DialectAsmPrinter printer(subPrinter);
    dialect.printAttribute(attr, printer);
  }
  printDialectSymbol(os, "#", dialect.getNamespace(), attrName);
}

void AsmPrinter::Impl::printDialectType(Type type) {
  auto &dialect = type.getDialect();

  // Ask the dialect to serialize the type to a string.
  std::string typeName;
  {
    llvm::raw_string_ostream typeNameStr(typeName);
    Impl subPrinter(typeNameStr, state);
    DialectAsmPrinter printer(subPrinter);
    dialect.printType(type, printer);
  }
  printDialectSymbol(os, "!", dialect.getNamespace(), typeName);
}

void AsmPrinter::Impl::printEscapedString(StringRef str) {
  os << "\"";
  llvm::printEscapedString(str, os);
  os << "\"";
}

void AsmPrinter::Impl::printHexString(StringRef str) {
  os << "\"0x" << llvm::toHex(str) << "\"";
}
void AsmPrinter::Impl::printHexString(ArrayRef<char> data) {
  printHexString(StringRef(data.data(), data.size()));
}

LogicalResult AsmPrinter::Impl::pushCyclicPrinting(const void *opaquePointer) {
  return state.pushCyclicPrinting(opaquePointer);
}

void AsmPrinter::Impl::popCyclicPrinting() { state.popCyclicPrinting(); }

//===--------------------------------------------------------------------===//
// AsmPrinter
//===--------------------------------------------------------------------===//

AsmPrinter::~AsmPrinter() = default;

raw_ostream &AsmPrinter::getStream() const {
  assert(impl && "expected AsmPrinter::getStream to be overriden");
  return impl->getStream();
}

/// Print the given floating point value in a stablized form.
void AsmPrinter::printFloat(const APFloat &value) {
  assert(impl && "expected AsmPrinter::printFloat to be overriden");
  printFloatValue(value, impl->getStream());
}

void AsmPrinter::printType(Type type) {
  assert(impl && "expected AsmPrinter::printType to be overriden");
  impl->printType(type);
}

void AsmPrinter::printAttribute(Attribute attr) {
  assert(impl && "expected AsmPrinter::printAttribute to be overriden");
  impl->printAttribute(attr);
}

LogicalResult AsmPrinter::printAlias(Attribute attr) {
  assert(impl && "expected AsmPrinter::printAlias to be overriden");
  return impl->printAlias(attr);
}

LogicalResult AsmPrinter::printAlias(Type type) {
  assert(impl && "expected AsmPrinter::printAlias to be overriden");
  return impl->printAlias(type);
}

void AsmPrinter::printAttributeWithoutType(Attribute attr) {
  assert(impl &&
         "expected AsmPrinter::printAttributeWithoutType to be overriden");
  impl->printAttribute(attr, Impl::AttrTypeElision::Must);
}

void AsmPrinter::printKeywordOrString(StringRef keyword) {
  assert(impl && "expected AsmPrinter::printKeywordOrString to be overriden");
  ::printKeywordOrString(keyword, impl->getStream());
}

void AsmPrinter::printString(StringRef keyword) {
  assert(impl && "expected AsmPrinter::printString to be overriden");
  *this << '"';
  printEscapedString(keyword, getStream());
  *this << '"';
}

void AsmPrinter::printSymbolName(StringRef symbolRef) {
  assert(impl && "expected AsmPrinter::printSymbolName to be overriden");
  ::printSymbolReference(symbolRef, impl->getStream());
}

void AsmPrinter::printResourceHandle(const AsmDialectResourceHandle &resource) {
  assert(impl && "expected AsmPrinter::printResourceHandle to be overriden");
  impl->printResourceHandle(resource);
}

LogicalResult AsmPrinter::pushCyclicPrinting(const void *opaquePointer) {
  return impl->pushCyclicPrinting(opaquePointer);
}

void AsmPrinter::popCyclicPrinting() { impl->popCyclicPrinting(); }

//===----------------------------------------------------------------------===//
// Affine expressions and maps
//===----------------------------------------------------------------------===//

void AsmPrinter::Impl::printAffineExpr(
    AffineExpr expr, function_ref<void(unsigned, bool)> printValueName) {
  printAffineExprInternal(expr, BindingStrength::Weak, printValueName);
}

void AsmPrinter::Impl::printAffineExprInternal(
    AffineExpr expr, BindingStrength enclosingTightness,
    function_ref<void(unsigned, bool)> printValueName) {
  const char *binopSpelling = nullptr;
  switch (expr.getKind()) {
  case AffineExprKind::SymbolId: {
    unsigned pos = expr.cast<AffineSymbolExpr>().getPosition();
    if (printValueName)
      printValueName(pos, /*isSymbol=*/true);
    else
      os << 's' << pos;
    return;
  }
  case AffineExprKind::DimId: {
    unsigned pos = expr.cast<AffineDimExpr>().getPosition();
    if (printValueName)
      printValueName(pos, /*isSymbol=*/false);
    else
      os << 'd' << pos;
    return;
  }
  case AffineExprKind::Constant:
    os << expr.cast<AffineConstantExpr>().getValue();
    return;
  case AffineExprKind::Add:
    binopSpelling = " + ";
    break;
  case AffineExprKind::Mul:
    binopSpelling = " * ";
    break;
  case AffineExprKind::FloorDiv:
    binopSpelling = " floordiv ";
    break;
  case AffineExprKind::CeilDiv:
    binopSpelling = " ceildiv ";
    break;
  case AffineExprKind::Mod:
    binopSpelling = " mod ";
    break;
  }

  auto binOp = expr.cast<AffineBinaryOpExpr>();
  AffineExpr lhsExpr = binOp.getLHS();
  AffineExpr rhsExpr = binOp.getRHS();

  // Handle tightly binding binary operators.
  if (binOp.getKind() != AffineExprKind::Add) {
    if (enclosingTightness == BindingStrength::Strong)
      os << '(';

    // Pretty print multiplication with -1.
    auto rhsConst = rhsExpr.dyn_cast<AffineConstantExpr>();
    if (rhsConst && binOp.getKind() == AffineExprKind::Mul &&
        rhsConst.getValue() == -1) {
      os << "-";
      printAffineExprInternal(lhsExpr, BindingStrength::Strong, printValueName);
      if (enclosingTightness == BindingStrength::Strong)
        os << ')';
      return;
    }

    printAffineExprInternal(lhsExpr, BindingStrength::Strong, printValueName);

    os << binopSpelling;
    printAffineExprInternal(rhsExpr, BindingStrength::Strong, printValueName);

    if (enclosingTightness == BindingStrength::Strong)
      os << ')';
    return;
  }

  // Print out special "pretty" forms for add.
  if (enclosingTightness == BindingStrength::Strong)
    os << '(';

  // Pretty print addition to a product that has a negative operand as a
  // subtraction.
  if (auto rhs = rhsExpr.dyn_cast<AffineBinaryOpExpr>()) {
    if (rhs.getKind() == AffineExprKind::Mul) {
      AffineExpr rrhsExpr = rhs.getRHS();
      if (auto rrhs = rrhsExpr.dyn_cast<AffineConstantExpr>()) {
        if (rrhs.getValue() == -1) {
          printAffineExprInternal(lhsExpr, BindingStrength::Weak,
                                  printValueName);
          os << " - ";
          if (rhs.getLHS().getKind() == AffineExprKind::Add) {
            printAffineExprInternal(rhs.getLHS(), BindingStrength::Strong,
                                    printValueName);
          } else {
            printAffineExprInternal(rhs.getLHS(), BindingStrength::Weak,
                                    printValueName);
          }

          if (enclosingTightness == BindingStrength::Strong)
            os << ')';
          return;
        }

        if (rrhs.getValue() < -1) {
          printAffineExprInternal(lhsExpr, BindingStrength::Weak,
                                  printValueName);
          os << " - ";
          printAffineExprInternal(rhs.getLHS(), BindingStrength::Strong,
                                  printValueName);
          os << " * " << -rrhs.getValue();
          if (enclosingTightness == BindingStrength::Strong)
            os << ')';
          return;
        }
      }
    }
  }

  // Pretty print addition to a negative number as a subtraction.
  if (auto rhsConst = rhsExpr.dyn_cast<AffineConstantExpr>()) {
    if (rhsConst.getValue() < 0) {
      printAffineExprInternal(lhsExpr, BindingStrength::Weak, printValueName);
      os << " - " << -rhsConst.getValue();
      if (enclosingTightness == BindingStrength::Strong)
        os << ')';
      return;
    }
  }

  printAffineExprInternal(lhsExpr, BindingStrength::Weak, printValueName);

  os << " + ";
  printAffineExprInternal(rhsExpr, BindingStrength::Weak, printValueName);

  if (enclosingTightness == BindingStrength::Strong)
    os << ')';
}

void AsmPrinter::Impl::printAffineConstraint(AffineExpr expr, bool isEq) {
  printAffineExprInternal(expr, BindingStrength::Weak);
  isEq ? os << " == 0" : os << " >= 0";
}

void AsmPrinter::Impl::printAffineMap(AffineMap map) {
  // Dimension identifiers.
  os << '(';
  for (int i = 0; i < (int)map.getNumDims() - 1; ++i)
    os << 'd' << i << ", ";
  if (map.getNumDims() >= 1)
    os << 'd' << map.getNumDims() - 1;
  os << ')';

  // Symbolic identifiers.
  if (map.getNumSymbols() != 0) {
    os << '[';
    for (unsigned i = 0; i < map.getNumSymbols() - 1; ++i)
      os << 's' << i << ", ";
    if (map.getNumSymbols() >= 1)
      os << 's' << map.getNumSymbols() - 1;
    os << ']';
  }

  // Result affine expressions.
  os << " -> (";
  interleaveComma(map.getResults(),
                  [&](AffineExpr expr) { printAffineExpr(expr); });
  os << ')';
}

void AsmPrinter::Impl::printIntegerSet(IntegerSet set) {
  // Dimension identifiers.
  os << '(';
  for (unsigned i = 1; i < set.getNumDims(); ++i)
    os << 'd' << i - 1 << ", ";
  if (set.getNumDims() >= 1)
    os << 'd' << set.getNumDims() - 1;
  os << ')';

  // Symbolic identifiers.
  if (set.getNumSymbols() != 0) {
    os << '[';
    for (unsigned i = 0; i < set.getNumSymbols() - 1; ++i)
      os << 's' << i << ", ";
    if (set.getNumSymbols() >= 1)
      os << 's' << set.getNumSymbols() - 1;
    os << ']';
  }

  // Print constraints.
  os << " : (";
  int numConstraints = set.getNumConstraints();
  for (int i = 1; i < numConstraints; ++i) {
    printAffineConstraint(set.getConstraint(i - 1), set.isEq(i - 1));
    os << ", ";
  }
  if (numConstraints >= 1)
    printAffineConstraint(set.getConstraint(numConstraints - 1),
                          set.isEq(numConstraints - 1));
  os << ')';
}

//===----------------------------------------------------------------------===//
// OperationPrinter
//===----------------------------------------------------------------------===//

namespace {
/// This class contains the logic for printing operations, regions, and blocks.
class OperationPrinter : public AsmPrinter::Impl, private OpAsmPrinter {
public:
  using Impl = AsmPrinter::Impl;
  using Impl::printType;

  explicit OperationPrinter(raw_ostream &os, AsmStateImpl &state)
      : Impl(os, state), OpAsmPrinter(static_cast<Impl &>(*this)) {}

  /// Print the given top-level operation.
  void printTopLevelOperation(Operation *op);

  /// Print the given operation, including its left-hand side and its right-hand
  /// side, with its indent and location.
  void printFullOpWithIndentAndLoc(Operation *op);
  /// Print the given operation, including its left-hand side and its right-hand
  /// side, but not including indentation and location.
  void printFullOp(Operation *op);
  /// Print the right-hand size of the given operation in the custom or generic
  /// form.
  void printCustomOrGenericOp(Operation *op) override;
  /// Print the right-hand side of the given operation in the generic form.
  void printGenericOp(Operation *op, bool printOpName) override;

  /// Print the name of the given block.
  void printBlockName(Block *block);

  /// Print the given block. If 'printBlockArgs' is false, the arguments of the
  /// block are not printed. If 'printBlockTerminator' is false, the terminator
  /// operation of the block is not printed.
  void print(Block *block, bool printBlockArgs = true,
             bool printBlockTerminator = true);

  /// Print the ID of the given value, optionally with its result number.
  void printValueID(Value value, bool printResultNo = true,
                    raw_ostream *streamOverride = nullptr) const;

  /// Print the ID of the given operation.
  void printOperationID(Operation *op,
                        raw_ostream *streamOverride = nullptr) const;

  //===--------------------------------------------------------------------===//
  // OpAsmPrinter methods
  //===--------------------------------------------------------------------===//

  /// Print a loc(...) specifier if printing debug info is enabled. Locations
  /// may be deferred with an alias.
  void printOptionalLocationSpecifier(Location loc) override {
    printTrailingLocation(loc);
  }

  /// Print a newline and indent the printer to the start of the current
  /// operation.
  void printNewline() override {
    os << newLine;
    os.indent(currentIndent);
  }

  /// Increase indentation.
  void increaseIndent() override { currentIndent += indentWidth; }

  /// Decrease indentation.
  void decreaseIndent() override { currentIndent -= indentWidth; }

  /// Print a block argument in the usual format of:
  ///   %ssaName : type {attr1=42} loc("here")
  /// where location printing is controlled by the standard internal option.
  /// You may pass omitType=true to not print a type, and pass an empty
  /// attribute list if you don't care for attributes.
  void printRegionArgument(BlockArgument arg,
                           ArrayRef<NamedAttribute> argAttrs = {},
                           bool omitType = false) override;

  /// Print the ID for the given value.
  void printOperand(Value value) override { printValueID(value); }
  void printOperand(Value value, raw_ostream &os) override {
    printValueID(value, /*printResultNo=*/true, &os);
  }

  /// Print an optional attribute dictionary with a given set of elided values.
  void printOptionalAttrDict(ArrayRef<NamedAttribute> attrs,
                             ArrayRef<StringRef> elidedAttrs = {}) override {
    Impl::printOptionalAttrDict(attrs, elidedAttrs);
  }
  void printOptionalAttrDictWithKeyword(
      ArrayRef<NamedAttribute> attrs,
      ArrayRef<StringRef> elidedAttrs = {}) override {
    Impl::printOptionalAttrDict(attrs, elidedAttrs,
                                /*withKeyword=*/true);
  }

  /// Print the given successor.
  void printSuccessor(Block *successor) override;

  /// Print an operation successor with the operands used for the block
  /// arguments.
  void printSuccessorAndUseList(Block *successor,
                                ValueRange succOperands) override;

  /// Print the given region.
  void printRegion(Region &region, bool printEntryBlockArgs,
                   bool printBlockTerminators, bool printEmptyBlock) override;

  /// Renumber the arguments for the specified region to the same names as the
  /// SSA values in namesToUse. This may only be used for IsolatedFromAbove
  /// operations. If any entry in namesToUse is null, the corresponding
  /// argument name is left alone.
  void shadowRegionArgs(Region &region, ValueRange namesToUse) override {
    state.getSSANameState().shadowRegionArgs(region, namesToUse);
  }

  /// Print the given affine map with the symbol and dimension operands printed
  /// inline with the map.
  void printAffineMapOfSSAIds(AffineMapAttr mapAttr,
                              ValueRange operands) override;

  /// Print the given affine expression with the symbol and dimension operands
  /// printed inline with the expression.
  void printAffineExprOfSSAIds(AffineExpr expr, ValueRange dimOperands,
                               ValueRange symOperands) override;

  /// Print users of this operation or id of this operation if it has no result.
  void printUsersComment(Operation *op);

  /// Print users of this block arg.
  void printUsersComment(BlockArgument arg);

  /// Print the users of a value.
  void printValueUsers(Value value);

  /// Print either the ids of the result values or the id of the operation if
  /// the operation has no results.
  void printUserIDs(Operation *user, bool prefixComma = false);

private:
  /// This class represents a resource builder implementation for the MLIR
  /// textual assembly format.
  class ResourceBuilder : public AsmResourceBuilder {
  public:
    using ValueFn = function_ref<void(raw_ostream &)>;
    using PrintFn = function_ref<void(StringRef, ValueFn)>;

    ResourceBuilder(PrintFn printFn) : printFn(printFn) {}
    ~ResourceBuilder() override = default;

    void buildBool(StringRef key, bool data) final {
      printFn(key, [&](raw_ostream &os) { os << (data ? "true" : "false"); });
    }

    void buildString(StringRef key, StringRef data) final {
      printFn(key, [&](raw_ostream &os) {
        os << "\"";
        llvm::printEscapedString(data, os);
        os << "\"";
      });
    }

    void buildBlob(StringRef key, ArrayRef<char> data,
                   uint32_t dataAlignment) final {
      printFn(key, [&](raw_ostream &os) {
        // Store the blob in a hex string containing the alignment and the data.
        llvm::support::ulittle32_t dataAlignmentLE(dataAlignment);
        os << "\"0x"
           << llvm::toHex(StringRef(reinterpret_cast<char *>(&dataAlignmentLE),
                                    sizeof(dataAlignment)))
           << llvm::toHex(StringRef(data.data(), data.size())) << "\"";
      });
    }

  private:
    PrintFn printFn;
  };

  /// Print the metadata dictionary for the file, eliding it if it is empty.
  void printFileMetadataDictionary(Operation *op);

  /// Print the resource sections for the file metadata dictionary.
  /// `checkAddMetadataDict` is used to indicate that metadata is going to be
  /// added, and the file metadata dictionary should be started if it hasn't
  /// yet.
  void printResourceFileMetadata(function_ref<void()> checkAddMetadataDict,
                                 Operation *op);

  // Contains the stack of default dialects to use when printing regions.
  // A new dialect is pushed to the stack before parsing regions nested under an
  // operation implementing `OpAsmOpInterface`, and popped when done. At the
  // top-level we start with "builtin" as the default, so that the top-level
  // `module` operation prints as-is.
  SmallVector<StringRef> defaultDialectStack{"builtin"};

  /// The number of spaces used for indenting nested operations.
  const static unsigned indentWidth = 2;

  // This is the current indentation level for nested structures.
  unsigned currentIndent = 0;
};
} // namespace

void OperationPrinter::printTopLevelOperation(Operation *op) {
  // Output the aliases at the top level that can't be deferred.
  state.getAliasState().printNonDeferredAliases(*this, newLine);

  // Print the module.
  printFullOpWithIndentAndLoc(op);
  os << newLine;

  // Output the aliases at the top level that can be deferred.
  state.getAliasState().printDeferredAliases(*this, newLine);

  // Output any file level metadata.
  printFileMetadataDictionary(op);
}

void OperationPrinter::printFileMetadataDictionary(Operation *op) {
  bool sawMetadataEntry = false;
  auto checkAddMetadataDict = [&] {
    if (!std::exchange(sawMetadataEntry, true))
      os << newLine << "{-#" << newLine;
  };

  // Add the various types of metadata.
  printResourceFileMetadata(checkAddMetadataDict, op);

  // If the file dictionary exists, close it.
  if (sawMetadataEntry)
    os << newLine << "#-}" << newLine;
}

void OperationPrinter::printResourceFileMetadata(
    function_ref<void()> checkAddMetadataDict, Operation *op) {
  // Functor used to add data entries to the file metadata dictionary.
  bool hadResource = false;
  bool needResourceComma = false;
  bool needEntryComma = false;
  auto processProvider = [&](StringRef dictName, StringRef name, auto &provider,
                             auto &&...providerArgs) {
    bool hadEntry = false;
    auto printFn = [&](StringRef key, ResourceBuilder::ValueFn valueFn) {
      checkAddMetadataDict();

      auto printFormatting = [&]() {
        // Emit the top-level resource entry if we haven't yet.
        if (!std::exchange(hadResource, true)) {
          if (needResourceComma)
            os << "," << newLine;
          os << "  " << dictName << "_resources: {" << newLine;
        }
        // Emit the parent resource entry if we haven't yet.
        if (!std::exchange(hadEntry, true)) {
          if (needEntryComma)
            os << "," << newLine;
          os << "    " << name << ": {" << newLine;
        } else {
          os << "," << newLine;
        }
      };

      std::optional<uint64_t> charLimit =
          printerFlags.getLargeResourceStringLimit();
      if (charLimit.has_value()) {
        std::string resourceStr;
        llvm::raw_string_ostream ss(resourceStr);
        valueFn(ss);

        // Only print entry if it's string is small enough
        if (resourceStr.size() > charLimit.value())
          return;

        printFormatting();
        os << "      " << key << ": " << resourceStr;
      } else {
        printFormatting();
        os << "      " << key << ": ";
        valueFn(os);
      }
    };
    ResourceBuilder entryBuilder(printFn);
    provider.buildResources(op, providerArgs..., entryBuilder);

    needEntryComma |= hadEntry;
    if (hadEntry)
      os << newLine << "    }";
  };

  // Print the `dialect_resources` section if we have any dialects with
  // resources.
  for (const OpAsmDialectInterface &interface : state.getDialectInterfaces()) {
    auto &dialectResources = state.getDialectResources();
    StringRef name = interface.getDialect()->getNamespace();
    auto it = dialectResources.find(interface.getDialect());
    if (it != dialectResources.end())
      processProvider("dialect", name, interface, it->second);
    else
      processProvider("dialect", name, interface,
                      SetVector<AsmDialectResourceHandle>());
  }
  if (hadResource)
    os << newLine << "  }";

  // Print the `external_resources` section if we have any external clients with
  // resources.
  needEntryComma = false;
  needResourceComma = hadResource;
  hadResource = false;
  for (const auto &printer : state.getResourcePrinters())
    processProvider("external", printer.getName(), printer);
  if (hadResource)
    os << newLine << "  }";
}

/// Print a block argument in the usual format of:
///   %ssaName : type {attr1=42} loc("here")
/// where location printing is controlled by the standard internal option.
/// You may pass omitType=true to not print a type, and pass an empty
/// attribute list if you don't care for attributes.
void OperationPrinter::printRegionArgument(BlockArgument arg,
                                           ArrayRef<NamedAttribute> argAttrs,
                                           bool omitType) {
  printOperand(arg);
  if (!omitType) {
    os << ": ";
    printType(arg.getType());
  }
  printOptionalAttrDict(argAttrs);
  // TODO: We should allow location aliases on block arguments.
  printTrailingLocation(arg.getLoc(), /*allowAlias*/ false);
}

void OperationPrinter::printFullOpWithIndentAndLoc(Operation *op) {
  // Track the location of this operation.
  state.registerOperationLocation(op, newLine.curLine, currentIndent);

  os.indent(currentIndent);
  printFullOp(op);
  printTrailingLocation(op->getLoc());
  if (printerFlags.shouldPrintValueUsers())
    printUsersComment(op);
}

void OperationPrinter::printFullOp(Operation *op) {
  if (size_t numResults = op->getNumResults()) {
    auto printResultGroup = [&](size_t resultNo, size_t resultCount) {
      printValueID(op->getResult(resultNo), /*printResultNo=*/false);
      if (resultCount > 1)
        os << ':' << resultCount;
    };

    // Check to see if this operation has multiple result groups.
    ArrayRef<int> resultGroups = state.getSSANameState().getOpResultGroups(op);
    if (!resultGroups.empty()) {
      // Interleave the groups excluding the last one, this one will be handled
      // separately.
      interleaveComma(llvm::seq<int>(0, resultGroups.size() - 1), [&](int i) {
        printResultGroup(resultGroups[i],
                         resultGroups[i + 1] - resultGroups[i]);
      });
      os << ", ";
      printResultGroup(resultGroups.back(), numResults - resultGroups.back());

    } else {
      printResultGroup(/*resultNo=*/0, /*resultCount=*/numResults);
    }

    os << " = ";
  }

  printCustomOrGenericOp(op);
}

void OperationPrinter::printUsersComment(Operation *op) {
  unsigned numResults = op->getNumResults();
  if (!numResults && op->getNumOperands()) {
    os << " // id: ";
    printOperationID(op);
  } else if (numResults && op->use_empty()) {
    os << " // unused";
  } else if (numResults && !op->use_empty()) {
    // Print "user" if the operation has one result used to compute one other
    // result, or is used in one operation with no result.
    unsigned usedInNResults = 0;
    unsigned usedInNOperations = 0;
    SmallPtrSet<Operation *, 1> userSet;
    for (Operation *user : op->getUsers()) {
      if (userSet.insert(user).second) {
        ++usedInNOperations;
        usedInNResults += user->getNumResults();
      }
    }

    // We already know that users is not empty.
    bool exactlyOneUniqueUse =
        usedInNResults <= 1 && usedInNOperations <= 1 && numResults == 1;
    os << " // " << (exactlyOneUniqueUse ? "user" : "users") << ": ";
    bool shouldPrintBrackets = numResults > 1;
    auto printOpResult = [&](OpResult opResult) {
      if (shouldPrintBrackets)
        os << "(";
      printValueUsers(opResult);
      if (shouldPrintBrackets)
        os << ")";
    };

    interleaveComma(op->getResults(), printOpResult);
  }
}

void OperationPrinter::printUsersComment(BlockArgument arg) {
  os << "// ";
  printValueID(arg);
  if (arg.use_empty()) {
    os << " is unused";
  } else {
    os << " is used by ";
    printValueUsers(arg);
  }
  os << newLine;
}

void OperationPrinter::printValueUsers(Value value) {
  if (value.use_empty())
    os << "unused";

  // One value might be used as the operand of an operation more than once.
  // Only print the operations results once in that case.
  SmallPtrSet<Operation *, 1> userSet;
  for (auto [index, user] : enumerate(value.getUsers())) {
    if (userSet.insert(user).second)
      printUserIDs(user, index);
  }
}

void OperationPrinter::printUserIDs(Operation *user, bool prefixComma) {
  if (prefixComma)
    os << ", ";

  if (!user->getNumResults()) {
    printOperationID(user);
  } else {
    interleaveComma(user->getResults(),
                    [this](Value result) { printValueID(result); });
  }
}

void OperationPrinter::printCustomOrGenericOp(Operation *op) {
  // If requested, always print the generic form.
  if (!printerFlags.shouldPrintGenericOpForm()) {
    // Check to see if this is a known operation. If so, use the registered
    // custom printer hook.
    if (auto opInfo = op->getRegisteredInfo()) {
      opInfo->printAssembly(op, *this, defaultDialectStack.back());
      return;
    }
    // Otherwise try to dispatch to the dialect, if available.
    if (Dialect *dialect = op->getDialect()) {
      if (auto opPrinter = dialect->getOperationPrinter(op)) {
        // Print the op name first.
        StringRef name = op->getName().getStringRef();
        // Only drop the default dialect prefix when it cannot lead to
        // ambiguities.
        if (name.count('.') == 1)
          name.consume_front((defaultDialectStack.back() + ".").str());
        os << name;

        // Print the rest of the op now.
        opPrinter(op, *this);
        return;
      }
    }
  }

  // Otherwise print with the generic assembly form.
  printGenericOp(op, /*printOpName=*/true);
}

void OperationPrinter::printGenericOp(Operation *op, bool printOpName) {
  if (printOpName)
    printEscapedString(op->getName().getStringRef());
  os << '(';
  interleaveComma(op->getOperands(), [&](Value value) { printValueID(value); });
  os << ')';

  // For terminators, print the list of successors and their operands.
  if (op->getNumSuccessors() != 0) {
    os << '[';
    interleaveComma(op->getSuccessors(),
                    [&](Block *successor) { printBlockName(successor); });
    os << ']';
  }

  // Print the properties.
  if (Attribute prop = op->getPropertiesAsAttribute()) {
    os << " <";
    Impl::printAttribute(prop);
    os << '>';
  }

  // Print regions.
  if (op->getNumRegions() != 0) {
    os << " (";
    interleaveComma(op->getRegions(), [&](Region &region) {
      printRegion(region, /*printEntryBlockArgs=*/true,
                  /*printBlockTerminators=*/true, /*printEmptyBlock=*/true);
    });
    os << ')';
  }

  auto attrs = op->getDiscardableAttrs();
  printOptionalAttrDict(attrs);

  // Print the type signature of the operation.
  os << " : ";
  printFunctionalType(op);
}

void OperationPrinter::printBlockName(Block *block) {
  os << state.getSSANameState().getBlockInfo(block).name;
}

void OperationPrinter::print(Block *block, bool printBlockArgs,
                             bool printBlockTerminator) {
  // Print the block label and argument list if requested.
  if (printBlockArgs) {
    os.indent(currentIndent);
    printBlockName(block);

    // Print the argument list if non-empty.
    if (!block->args_empty()) {
      os << '(';
      interleaveComma(block->getArguments(), [&](BlockArgument arg) {
        printValueID(arg);
        os << ": ";
        printType(arg.getType());
        // TODO: We should allow location aliases on block arguments.
        printTrailingLocation(arg.getLoc(), /*allowAlias*/ false);
      });
      os << ')';
    }
    os << ':';

    // Print out some context information about the predecessors of this block.
    if (!block->getParent()) {
      os << "  // block is not in a region!";
    } else if (block->hasNoPredecessors()) {
      if (!block->isEntryBlock())
        os << "  // no predecessors";
    } else if (auto *pred = block->getSinglePredecessor()) {
      os << "  // pred: ";
      printBlockName(pred);
    } else {
      // We want to print the predecessors in a stable order, not in
      // whatever order the use-list is in, so gather and sort them.
      SmallVector<BlockInfo, 4> predIDs;
      for (auto *pred : block->getPredecessors())
        predIDs.push_back(state.getSSANameState().getBlockInfo(pred));
      llvm::sort(predIDs, [](BlockInfo lhs, BlockInfo rhs) {
        return lhs.ordering < rhs.ordering;
      });

      os << "  // " << predIDs.size() << " preds: ";

      interleaveComma(predIDs, [&](BlockInfo pred) { os << pred.name; });
    }
    os << newLine;
  }

  currentIndent += indentWidth;

  if (printerFlags.shouldPrintValueUsers()) {
    for (BlockArgument arg : block->getArguments()) {
      os.indent(currentIndent);
      printUsersComment(arg);
    }
  }

  bool hasTerminator =
      !block->empty() && block->back().hasTrait<OpTrait::IsTerminator>();
  auto range = llvm::make_range(
      block->begin(),
      std::prev(block->end(),
                (!hasTerminator || printBlockTerminator) ? 0 : 1));
  for (auto &op : range) {
    printFullOpWithIndentAndLoc(&op);
    os << newLine;
  }
  currentIndent -= indentWidth;
}

void OperationPrinter::printValueID(Value value, bool printResultNo,
                                    raw_ostream *streamOverride) const {
  state.getSSANameState().printValueID(value, printResultNo,
                                       streamOverride ? *streamOverride : os);
}

void OperationPrinter::printOperationID(Operation *op,
                                        raw_ostream *streamOverride) const {
  state.getSSANameState().printOperationID(op, streamOverride ? *streamOverride
                                                              : os);
}

void OperationPrinter::printSuccessor(Block *successor) {
  printBlockName(successor);
}

void OperationPrinter::printSuccessorAndUseList(Block *successor,
                                                ValueRange succOperands) {
  printBlockName(successor);
  if (succOperands.empty())
    return;

  os << '(';
  interleaveComma(succOperands,
                  [this](Value operand) { printValueID(operand); });
  os << " : ";
  interleaveComma(succOperands,
                  [this](Value operand) { printType(operand.getType()); });
  os << ')';
}

void OperationPrinter::printRegion(Region &region, bool printEntryBlockArgs,
                                   bool printBlockTerminators,
                                   bool printEmptyBlock) {
  if (printerFlags.shouldSkipRegions()) {
    os << "{...}";
    return;
  }
  os << "{" << newLine;
  if (!region.empty()) {
    auto restoreDefaultDialect =
        llvm::make_scope_exit([&]() { defaultDialectStack.pop_back(); });
    if (auto iface = dyn_cast<OpAsmOpInterface>(region.getParentOp()))
      defaultDialectStack.push_back(iface.getDefaultDialect());
    else
      defaultDialectStack.push_back("");

    auto *entryBlock = &region.front();
    // Force printing the block header if printEmptyBlock is set and the block
    // is empty or if printEntryBlockArgs is set and there are arguments to
    // print.
    bool shouldAlwaysPrintBlockHeader =
        (printEmptyBlock && entryBlock->empty()) ||
        (printEntryBlockArgs && entryBlock->getNumArguments() != 0);
    print(entryBlock, shouldAlwaysPrintBlockHeader, printBlockTerminators);
    for (auto &b : llvm::drop_begin(region.getBlocks(), 1))
      print(&b);
  }
  os.indent(currentIndent) << "}";
}

void OperationPrinter::printAffineMapOfSSAIds(AffineMapAttr mapAttr,
                                              ValueRange operands) {
  if (!mapAttr) {
    os << "<<NULL AFFINE MAP>>";
    return;
  }
  AffineMap map = mapAttr.getValue();
  unsigned numDims = map.getNumDims();
  auto printValueName = [&](unsigned pos, bool isSymbol) {
    unsigned index = isSymbol ? numDims + pos : pos;
    assert(index < operands.size());
    if (isSymbol)
      os << "symbol(";
    printValueID(operands[index]);
    if (isSymbol)
      os << ')';
  };

  interleaveComma(map.getResults(), [&](AffineExpr expr) {
    printAffineExpr(expr, printValueName);
  });
}

void OperationPrinter::printAffineExprOfSSAIds(AffineExpr expr,
                                               ValueRange dimOperands,
                                               ValueRange symOperands) {
  auto printValueName = [&](unsigned pos, bool isSymbol) {
    if (!isSymbol)
      return printValueID(dimOperands[pos]);
    os << "symbol(";
    printValueID(symOperands[pos]);
    os << ')';
  };
  printAffineExpr(expr, printValueName);
}

//===----------------------------------------------------------------------===//
// print and dump methods
//===----------------------------------------------------------------------===//

void Attribute::print(raw_ostream &os, bool elideType) const {
  if (!*this) {
    os << "<<NULL ATTRIBUTE>>";
    return;
  }

  AsmState state(getContext());
  print(os, state, elideType);
}
void Attribute::print(raw_ostream &os, AsmState &state, bool elideType) const {
  using AttrTypeElision = AsmPrinter::Impl::AttrTypeElision;
  AsmPrinter::Impl(os, state.getImpl())
      .printAttribute(*this, elideType ? AttrTypeElision::Must
                                       : AttrTypeElision::Never);
}

void Attribute::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void Type::print(raw_ostream &os) const {
  if (!*this) {
    os << "<<NULL TYPE>>";
    return;
  }

  AsmState state(getContext());
  print(os, state);
}
void Type::print(raw_ostream &os, AsmState &state) const {
  AsmPrinter::Impl(os, state.getImpl()).printType(*this);
}

void Type::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void AffineMap::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void IntegerSet::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void AffineExpr::print(raw_ostream &os) const {
  if (!expr) {
    os << "<<NULL AFFINE EXPR>>";
    return;
  }
  AsmState state(getContext());
  AsmPrinter::Impl(os, state.getImpl()).printAffineExpr(*this);
}

void AffineExpr::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void AffineMap::print(raw_ostream &os) const {
  if (!map) {
    os << "<<NULL AFFINE MAP>>";
    return;
  }
  AsmState state(getContext());
  AsmPrinter::Impl(os, state.getImpl()).printAffineMap(*this);
}

void IntegerSet::print(raw_ostream &os) const {
  AsmState state(getContext());
  AsmPrinter::Impl(os, state.getImpl()).printIntegerSet(*this);
}

void Value::print(raw_ostream &os) { print(os, OpPrintingFlags()); }
void Value::print(raw_ostream &os, const OpPrintingFlags &flags) {
  if (!impl) {
    os << "<<NULL VALUE>>";
    return;
  }

  if (auto *op = getDefiningOp())
    return op->print(os, flags);
  // TODO: Improve BlockArgument print'ing.
  BlockArgument arg = llvm::cast<BlockArgument>(*this);
  os << "<block argument> of type '" << arg.getType()
     << "' at index: " << arg.getArgNumber();
}
void Value::print(raw_ostream &os, AsmState &state) {
  if (!impl) {
    os << "<<NULL VALUE>>";
    return;
  }

  if (auto *op = getDefiningOp())
    return op->print(os, state);

  // TODO: Improve BlockArgument print'ing.
  BlockArgument arg = llvm::cast<BlockArgument>(*this);
  os << "<block argument> of type '" << arg.getType()
     << "' at index: " << arg.getArgNumber();
}

void Value::dump() {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void Value::printAsOperand(raw_ostream &os, AsmState &state) {
  // TODO: This doesn't necessarily capture all potential cases.
  // Currently, region arguments can be shadowed when printing the main
  // operation. If the IR hasn't been printed, this will produce the old SSA
  // name and not the shadowed name.
  state.getImpl().getSSANameState().printValueID(*this, /*printResultNo=*/true,
                                                 os);
}

static Operation *findParent(Operation *op, bool shouldUseLocalScope) {
  do {
    // If we are printing local scope, stop at the first operation that is
    // isolated from above.
    if (shouldUseLocalScope && op->hasTrait<OpTrait::IsIsolatedFromAbove>())
      break;

    // Otherwise, traverse up to the next parent.
    Operation *parentOp = op->getParentOp();
    if (!parentOp)
      break;
    op = parentOp;
  } while (true);
  return op;
}

void Value::printAsOperand(raw_ostream &os, const OpPrintingFlags &flags) {
  Operation *op;
  if (auto result = llvm::dyn_cast<OpResult>(*this)) {
    op = result.getOwner();
  } else {
    op = llvm::cast<BlockArgument>(*this).getOwner()->getParentOp();
    if (!op) {
      os << "<<UNKNOWN SSA VALUE>>";
      return;
    }
  }
  op = findParent(op, flags.shouldUseLocalScope());
  AsmState state(op, flags);
  printAsOperand(os, state);
}

void Operation::print(raw_ostream &os, const OpPrintingFlags &printerFlags) {
  // Find the operation to number from based upon the provided flags.
  Operation *op = findParent(this, printerFlags.shouldUseLocalScope());
  AsmState state(op, printerFlags);
  print(os, state);
}
void Operation::print(raw_ostream &os, AsmState &state) {
  OperationPrinter printer(os, state.getImpl());
  if (!getParent() && !state.getPrinterFlags().shouldUseLocalScope()) {
    state.getImpl().initializeAliases(this);
    printer.printTopLevelOperation(this);
  } else {
    printer.printFullOpWithIndentAndLoc(this);
  }
}

void Operation::dump() {
  print(llvm::errs(), OpPrintingFlags().useLocalScope());
  llvm::errs() << "\n";
}

void Block::print(raw_ostream &os) {
  Operation *parentOp = getParentOp();
  if (!parentOp) {
    os << "<<UNLINKED BLOCK>>\n";
    return;
  }
  // Get the top-level op.
  while (auto *nextOp = parentOp->getParentOp())
    parentOp = nextOp;

  AsmState state(parentOp);
  print(os, state);
}
void Block::print(raw_ostream &os, AsmState &state) {
  OperationPrinter(os, state.getImpl()).print(this);
}

void Block::dump() { print(llvm::errs()); }

/// Print out the name of the block without printing its body.
void Block::printAsOperand(raw_ostream &os, bool printType) {
  Operation *parentOp = getParentOp();
  if (!parentOp) {
    os << "<<UNLINKED BLOCK>>\n";
    return;
  }
  AsmState state(parentOp);
  printAsOperand(os, state);
}
void Block::printAsOperand(raw_ostream &os, AsmState &state) {
  OperationPrinter printer(os, state.getImpl());
  printer.printBlockName(this);
}
