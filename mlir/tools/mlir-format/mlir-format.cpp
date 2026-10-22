#include "mlir/AsmParser/AsmParser.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/ParseUtilities.h"
#include "llvm/ADT/RewriteBuffer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

namespace mlir {
using OperationDefinition = AsmParserState::OperationDefinition;
using BlockDefinition = AsmParserState::BlockDefinition;
using SMDefinition = AsmParserState::SMDefinition;

inline int asMainReturnCode(LogicalResult r) {
  return r.succeeded() ? EXIT_SUCCESS : EXIT_FAILURE;
}

/// Return the source code associated with the OperationDefinition.
SMRange getOpRange(const OperationDefinition &op) {
  const char *startOp = op.scopeLoc.Start.getPointer();
  const char *endOp = op.scopeLoc.End.getPointer();

  for (const auto &res : op.resultGroups) {
    SMRange range = res.definition.loc;
    startOp = std::min(startOp, range.Start.getPointer());
  }
  return {SMLoc::getFromPointer(startOp), SMLoc::getFromPointer(endOp)};
}

class CombinedOpDefIterator {
public:
  using BaseIterator = AsmParserState::OperationDefIterator;
  using value_type = std::pair<OperationDefinition &, OperationDefinition &>;

  // Constructor
  CombinedOpDefIterator(BaseIterator opIter, BaseIterator fmtIter)
      : opIter(opIter), fmtIter(fmtIter) {}

  // Dereference operator to return a pair of references
  value_type operator*() const { return {*opIter, *fmtIter}; }

  // Increment operator
  CombinedOpDefIterator &operator++() {
    ++opIter;
    ++fmtIter;
    return *this;
  }

  // Equality operator
  bool operator==(const CombinedOpDefIterator &other) const {
    return opIter == other.opIter && fmtIter == other.fmtIter;
  }

  // Inequality operator
  bool operator!=(const CombinedOpDefIterator &other) const {
    return !(*this == other);
  }

private:
  BaseIterator opIter;
  BaseIterator fmtIter;
};

// Given the scopeLoc of an operation, extract src locations of the input and
// output type
std::pair<SmallVector<llvm::SMRange>, SmallVector<llvm::SMRange>>
getOpTypeLoc(llvm::SMRange op_loc) {
  SmallVector<llvm::SMRange> inputTypeRanges;
  SmallVector<llvm::SMRange> outputTypeRanges;

  // Extract the string from the range
  const char *startPtr = op_loc.Start.getPointer();
  const char *endPtr = op_loc.End.getPointer();
  StringRef opString(startPtr, endPtr - startPtr);

  // Find the position of the last ':' in the string
  size_t colonPos = opString.rfind(':');
  if (colonPos == StringRef::npos) {
    // No ':' found, return empty vectors
    return {inputTypeRanges, outputTypeRanges};
  }

  // Extract the type definition substring
  StringRef typeDefStr = opString.substr(colonPos + 1).trim();

  // Check if the type definition substring contains '->' (input -> output
  // types)
  size_t arrowPos = typeDefStr.find("->");

  if (arrowPos != StringRef::npos) {
    // Split into input and output type strings
    StringRef inputTypeStr = typeDefStr.substr(0, arrowPos).trim();
    StringRef outputTypeStr = typeDefStr.substr(arrowPos + 2).trim();

    // Parse input type ranges (if any)
    if (!inputTypeStr.empty() && inputTypeStr != "()") {
      SmallVector<StringRef> inputTypeParts;
      inputTypeStr
          .drop_front() // Remove '('
          .drop_back()  // Remove ')'
          .split(inputTypeParts, ',');

      for (const auto &typeStr : inputTypeParts) {
        const char *start = typeStr.trim().data();
        const char *end = start + typeStr.trim().size();
        inputTypeRanges.push_back(
            llvm::SMRange(llvm::SMLoc::getFromPointer(start),
                          llvm::SMLoc::getFromPointer(end)));
      }
    }

    // Parse output type ranges (if any)
    if (!outputTypeStr.empty() && outputTypeStr != "()") {
      SmallVector<StringRef> outputTypeParts;
      outputTypeStr.split(outputTypeParts, ',');

      for (const auto &typeStr : outputTypeParts) {
        const char *start = typeStr.trim().data();
        const char *end = start + typeStr.trim().size();
        outputTypeRanges.push_back(
            llvm::SMRange(llvm::SMLoc::getFromPointer(start),
                          llvm::SMLoc::getFromPointer(end)));
      }
    }
  } else {
    // Single type definition (no '->'), assume it's an output type
    SmallVector<StringRef> typeParts;
    typeDefStr.split(typeParts, ',');

    for (const auto &typeStr : typeParts) {
      const char *start = typeStr.trim().data();
      const char *end = start + typeStr.trim().size();
      outputTypeRanges.push_back(
          llvm::SMRange(llvm::SMLoc::getFromPointer(start),
                        llvm::SMLoc::getFromPointer(end)));
    }
  }

  return {inputTypeRanges, outputTypeRanges};
}

llvm::SMRange getSMRangeFromString(const std::string &str) {
  const char *startPtr = str.data();
  const char *endPtr = startPtr + str.size();
  return llvm::SMRange(llvm::SMLoc::getFromPointer(startPtr),
                       llvm::SMLoc::getFromPointer(endPtr));
}

void replaceTypesInString(std::string &formattedStr,
                          const SmallVector<llvm::SMRange> &inputTypes,
                          const SmallVector<llvm::SMRange> &outputTypes) {
  // Get type locations from the formatted string
  llvm::SMRange formattedLoc = getSMRangeFromString(formattedStr);
  auto formattedTypes = getOpTypeLoc(formattedLoc);

  // Ensure the number of types matches
  if (inputTypes.size() != formattedTypes.first.size() ||
      outputTypes.size() != formattedTypes.second.size()) {
    llvm::errs() << "Error: Mismatched number of input/output types in "
                    "replacement operation.\n";
    return;
  }

  // Perform input type replacements backwards to avoid index issues
  for (size_t i = inputTypes.size(); i-- > 0;) {
    const llvm::SMRange &formattedRange = formattedTypes.first[i];
    const llvm::SMRange &inputRange = inputTypes[i];

    const char *formattedStart = formattedRange.Start.getPointer();
    const char *formattedEnd = formattedRange.End.getPointer();

    const char *inputStart = inputRange.Start.getPointer();
    const char *inputEnd = inputRange.End.getPointer();

    llvm::StringRef formattedType(formattedStart,
                                  formattedEnd - formattedStart);
    llvm::StringRef inputType(inputStart, inputEnd - inputStart);

    // Replace in the formatted string
    size_t pos = formattedStr.find(formattedType.str());
    if (pos != std::string::npos) {
      formattedStr.replace(pos, formattedType.size(), inputType.str());
    } else {
      llvm::errs() << "Warning: Input type not found in formatted string: "
                   << formattedType << "\n";
    }
  }

  // Perform output type replacements backwards to avoid index issues
  for (size_t i = outputTypes.size(); i-- > 0;) {
    const llvm::SMRange &formattedRange = formattedTypes.second[i];
    const llvm::SMRange &outputRange = outputTypes[i];

    const char *formattedStart = formattedRange.Start.getPointer();
    const char *formattedEnd = formattedRange.End.getPointer();

    const char *outputStart = outputRange.Start.getPointer();
    const char *outputEnd = outputRange.End.getPointer();

    llvm::StringRef formattedType(formattedStart,
                                  formattedEnd - formattedStart);
    llvm::StringRef outputType(outputStart, outputEnd - outputStart);

    // Replace in the formatted string
    size_t pos = formattedStr.find(formattedType.str());
    if (pos != std::string::npos) {
      formattedStr.replace(pos, formattedType.size(), outputType.str());
    } else {
      llvm::errs() << "Warning: Output type not found in formatted string: "
                   << formattedType << "\n";
    }
  }
}

// Function to find the character before the previous comma
const char *findPrevComma(const char *start, const char *stop_point) {
  if (!start) {
    llvm::errs() << "Error: Input pointer is null.\n";
    return nullptr;
  }

  const char *current = start - 1; // Start checking backwards
  while (current >= stop_point) {
    if (*current == ',') {
      return current;
    }
    --current;
  }

  llvm::errs() << "Error: No previous comma found before provided pointer.\n";
  return nullptr;
}

// Function to find the next closing parenthesis
const char *findNextCloseParenth(const char *start) {
  if (!start) {
    llvm::errs() << "Error: Input pointer is null.\n";
    return nullptr;
  }

  while (*start != '\0') { // Traverse until null terminator
    if (*start == ')') {
      return start; // Return pointer to the closing parenthesis
    }
    ++start;
  }

  llvm::errs() << "Error: No closing parenthesis found in the string.\n";
  return nullptr;
}

class Formatter {
public:
  static std::unique_ptr<Formatter> init(StringRef inputFilename,
                                         StringRef outputFilename);

  /// Return the OperationDefinition's of the operations parsed.
  iterator_range<AsmParserState::OperationDefIterator> getOpDefs() {
    return asmState.getOpDefs();
  }

  /// Return the OperationDefinition's of the blocks parsed.
  iterator_range<AsmParserState::BlockDefIterator> getBlockDefs() {
    return asmState.getBlockDefs();
  }

  /// Return a pair iterators of OperationDefinitions for the asmState and
  /// fmtAsmState.
  iterator_range<CombinedOpDefIterator> getCombinedOpDefs() {
    auto opBegin = asmState.getOpDefs().begin();
    auto opEnd = asmState.getOpDefs().end();
    auto fmtBegin = fmtAsmState.getOpDefs().begin();
    auto fmtEnd = fmtAsmState.getOpDefs().end();

    assert(std::distance(opBegin, opEnd) == std::distance(fmtBegin, fmtEnd) &&
           "Both iterators must have the same length");

    return llvm::make_range(CombinedOpDefIterator(opBegin, fmtBegin),
                            CombinedOpDefIterator(opEnd, fmtEnd));
  }

  /// Print the parsed operations to the provided output stream.
  void printOps(raw_ostream &os) {
    // Iterate over each operation in the parsedIR block and print it.
    for (Operation &op : parsedIR) {
      op.print(os);
      os << "\n";
    }
  }

  /// Insert the specified string at the specified location in the original
  /// buffer.
  void insertText(SMLoc pos, StringRef str, bool insertAfter = true) {
    rewriteBuffer.InsertText(pos.getPointer() - start, str, insertAfter);
  }

  /// Replace the range of the source text with the corresponding string in the
  /// output.
  void replaceRange(SMRange range, StringRef str) {
    rewriteBuffer.ReplaceText(range.Start.getPointer() - start,
                              range.End.getPointer() - range.Start.getPointer(),
                              str);
  }

  void replaceRangeFmt(SMRange range, StringRef str) {
    fmtRewriteBuffer.ReplaceText(
        range.Start.getPointer() - start,
        range.End.getPointer() - range.Start.getPointer(), str);
  }

  /// Write to stream the result of applying all changes to the
  /// original buffer.
  /// Note that it isn't safe to use this function to overwrite memory mapped
  /// files in-place (PR17960).
  ///
  /// The original buffer is not actually changed.
  raw_ostream &write(raw_ostream &stream) const {
    return rewriteBuffer.write(stream);
  }

  raw_ostream &writeFmt(raw_ostream &stream) const {
    return fmtRewriteBuffer.write(stream);
  }

  /// Generate formatAsmState from the rewriteBuffer
  void formatOps();

private:
  // The context and state required to parse.
  MLIRContext context;
  llvm::SourceMgr sourceMgr;
  llvm::SourceMgr fmtSourceMgr;
  DialectRegistry registry;
  FallbackAsmResourceMap fallbackResourceMap;

  // Storage of textual parsing results.
  AsmParserState asmState;

  // Storage of initial formatted ops.
  AsmParserState fmtAsmState;

  // Parsed IR.
  Block parsedIR;
  Block parsedFmtIR;

  // The RewriteBuffer  is doing most of the real work.
  llvm::RewriteBuffer rewriteBuffer;
  llvm::RewriteBuffer fmtRewriteBuffer;

  // Start of the original input, used to compute offset.
  const char *start;
};

std::unique_ptr<Formatter> Formatter::init(StringRef inputFilename,
                                           StringRef outputFilename) {

  std::unique_ptr<Formatter> f = std::make_unique<Formatter>();
  // Register all the dialects needed.
  registerAllDialects(f->registry);

  // Set up the input file.
  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> file =
      openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }
  f->sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());

  // Set up the MLIR context and error handling.
  f->context.appendDialectRegistry(f->registry);

  // Record the start of the buffer to compute offsets with.
  unsigned curBuf = f->sourceMgr.getMainFileID();
  const llvm::MemoryBuffer *curMB = f->sourceMgr.getMemoryBuffer(curBuf);
  f->start = curMB->getBufferStart();
  f->rewriteBuffer.Initialize(curMB->getBuffer());
  f->fmtRewriteBuffer.Initialize(curMB->getBuffer());

  // Parse and populate the AsmParserState.
  ParserConfig parseConfig(&f->context, /*verifyAfterParse=*/true,
                           &f->fallbackResourceMap);
  // Always allow unregistered.
  f->context.allowUnregisteredDialects(true);
  if (failed(parseAsmSourceFile(f->sourceMgr, &f->parsedIR, parseConfig,
                                &f->asmState)))
    return nullptr;

  return f;
}

void Formatter::formatOps() {
  // Generate formatAsmState from the rewriteBuffer
  ParserConfig parseConfig(&context, /*verifyAfterParse=*/true,
                           &fallbackResourceMap);

  std::string bufferContent;
  llvm::raw_string_ostream stream(bufferContent);
  rewriteBuffer.write(stream);
  stream.flush();

  fmtSourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(bufferContent), SMLoc());

  // Parse and populate the forrmat AsmParserState.
  if (failed(parseAsmSourceFile(fmtSourceMgr, &parsedFmtIR, parseConfig,
                                &fmtAsmState)))
    return;

  // Insert the formatted ops.  Block args should be untouched,
  // and their references will use the correct SSA ID.
  for (auto [opDef, fmtDef] : getCombinedOpDefs()) {
    auto [startOp, endOp] = getOpRange(opDef);

    // Skip if the op is a FuncOp (we format ops in its body)
    // or a ReturnOp (we want to keep the user's preference for
    // `func.return` or plain `return`)
    if (llvm::dyn_cast<mlir::func::FuncOp>(fmtDef.op))
      continue;
    else if (llvm::dyn_cast<mlir::func::ReturnOp>(fmtDef.op))
      continue;

    // Print the fmtDef op and store as a string.
    std::string formattedStr;
    llvm::raw_string_ostream stream(formattedStr);
    fmtDef.op->print(stream);

    // Use the original type aliases
    auto orig_types = getOpTypeLoc(opDef.scopeLoc);
    replaceTypesInString(formattedStr, orig_types.first, orig_types.second);

    // Replace the opDef with this formatted string.
    replaceRangeFmt({startOp, endOp}, formattedStr);

    // Write the updated buffer to llvm::outs()
    writeFmt(llvm::outs());
  }

  std::string getNamedLoc(
      const OperationDefinition::ResultGroupDefinition &resultGroup) {
    auto sm_range = resultGroup.definition.loc;
    const char *start = sm_range.Start.getPointer();
    const int len = sm_range.End.getPointer() - start;

    // Drop the '%' prefix and construct the `loc("name")` string
    auto name = llvm::StringRef(start + 1,
                                len - 1); // Assumes the '%' is always present
    std::string formattedStr = " loc(\"" + name.str() + "\")";

    return formattedStr;
  }

  // To handle ops with multiple result groups, create a dummy "alias" op
  // so that we can each group its own NameLoc
  void insertAliasOp() {}

  LogicalResult markNames(Formatter & formatState, raw_ostream & os) {
    // Get the operation definitions from the AsmParserState.
    for (OperationDefinition &it : formatState.getOpDefs()) {
      auto [startOp, endOp] = getOpRange(it);

      if (it.resultGroups.size() == 1) {
        // Simple case, where we have only one result group for the op,
        // e.g., `%v = op` or `%v:2 = op`
        auto resultGroup = it.resultGroups[0];
        auto nameLoc = getNamedLoc(resultGroup);
        formatState.insertText(endOp, StringRef(nameLoc));
      } else {
        // Complex case, where we have more than one result group, e.g.,
        // `%x, %y = op` or `%xs:2, %ys:3 = op`.
        // In this case we need insert some aliasing ops.
        for (auto &resultGroup : it.resultGroups) {
          auto nameLoc = getNamedLoc(resultGroup);
          // StringRef namedLoc(getNamedLoc(resultGroup));
          llvm::errs() << "Not implemented yet\n";
          return failure();
        }
      }
    }

    // Insert the NameLocs for the block arguments
    for (BlockDefinition &block : formatState.getBlockDefs()) {
      for (size_t i = 0; i < block.arguments.size(); ++i) {
        SMDefinition &arg = block.arguments[i];

        // Find where to insert the NameLoc.  Either before the next argument,
        // or at the end of the arg list
        const char *insertPointPtr;
        const char *arg_end = arg.loc.End.getPointer();
        SMDefinition *nextArg = (i + 1 < block.arguments.size())
                                    ? &block.arguments[i + 1]
                                    : nullptr;
        if (nextArg) {
          const char *nextStart = nextArg->loc.Start.getPointer();
          insertPointPtr = findPrevComma(nextStart, arg_end);
        } else {
          insertPointPtr = findNextCloseParenth(arg.loc.End.getPointer());
        }

        // Drop the % prefix, and put in new string with  `loc("name")` format.
        const char *start = arg.loc.Start.getPointer();
        const int len = arg_end - start;
        auto name = StringRef(start + 1, len - 1);
        std::string formattedStr = " loc(\"" + name.str() + "\")";
        StringRef namedLoc(formattedStr);
        formatState.insertText(SMLoc::getFromPointer(insertPointPtr), namedLoc);
      }
    }
    return success();
  }
} // namespace mlir

int main(int argc, char **argv) {
  registerAsmPrinterCLOptions();
  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::init("-"));

  static llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  llvm::cl::opt<bool> nameLocOnly{
      "insert-name-loc-only", llvm::cl::init(false),
      llvm::cl::desc("Only return a buffer with the NameLocs appended")};

  std::string helpHeader = "mlir-format";

  llvm::cl::ParseCommandLineOptions(argc, argv, helpHeader);

  // Set up formatter buffer.
  auto f = Formatter::init(inputFilename, outputFilename);

  // Append the SSA names as NameLocs
  LogicalResult result = markNames(*f, llvm::outs());
  if (!succeeded(result))
    return mlir::asMainReturnCode(mlir::failure());

  if (nameLocOnly) {
    // Return the original buffer with NameLocs appended to ops
    // e.g., `%alice = memref.load %0[] : memref<i32> loc("alice")`
    f->write(llvm::outs());
    return mlir::asMainReturnCode(mlir::success());
  }

  f->formatOps();

  return mlir::asMainReturnCode(mlir::success());
}
