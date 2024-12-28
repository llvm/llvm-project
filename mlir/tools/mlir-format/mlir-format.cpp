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

  // Write the rewriteBuffer to a stream, that we can then parse
  std::string bufferContent;
  llvm::raw_string_ostream stream(bufferContent);
  rewriteBuffer.write(stream);
  stream.flush();

  // Print the bufferContent to llvm::outs() for debugging.
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
    // Replace the opDef with this formatted string.
    std::string formattedStr;
    llvm::raw_string_ostream stream(formattedStr);
    fmtDef.op->print(stream);

    // Replacing the range:
    replaceRangeFmt({startOp, endOp}, formattedStr);
  }

  // Write the updated buffer to llvm::outs()
  writeFmt(llvm::outs());
}

void markNames(Formatter &formatState, raw_ostream &os) {
  // Get the operation definitions from the AsmParserState.
  for (OperationDefinition &it : formatState.getOpDefs()) {
    auto [startOp, endOp] = getOpRange(it);
    // loop through the resultgroups
    for (auto &resultGroup : it.resultGroups) {
      auto def = resultGroup.definition;
      auto sm_range = def.loc;
      const char *start = sm_range.Start.getPointer();
      int len = sm_range.End.getPointer() - start;
      // Drop the % prefix, and put in new string with  `loc("name")` format.
      auto name = StringRef(start + 1, len - 1);

      // Add loc("{name}") to the end of the op
      std::string formattedStr = " loc(\"" + name.str() + "\")";
      StringRef namedLoc(formattedStr);
      formatState.insertText(endOp, namedLoc);
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
      SMDefinition *nextArg =
          (i + 1 < block.arguments.size()) ? &block.arguments[i + 1] : nullptr;
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
  markNames(*f, llvm::outs());

  if (nameLocOnly) {
    // Return the original buffer with NameLocs appended to ops
    // e.g., `%alice = memref.load %0[] : memref<i32> loc("alice")`
    f->write(llvm::outs());
    return mlir::asMainReturnCode(mlir::success());
  }

  f->formatOps();

  return mlir::asMainReturnCode(mlir::success());
}
