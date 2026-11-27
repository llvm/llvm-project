//===- mlir-rewrite.cpp - MLIR Rewrite Driver -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-rewrite.
//
//===----------------------------------------------------------------------===//

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
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

namespace mlir {
using OperationDefinition = AsmParserState::OperationDefinition;

/// Return the source code associated with the OperationDefinition.
static SMRange getOpRange(const OperationDefinition &op) {
  const char *startOp = op.scopeLoc.Start.getPointer();
  const char *endOp = op.scopeLoc.End.getPointer();

  for (const auto &res : op.resultGroups) {
    SMRange range = res.definition.loc;
    startOp = std::min(startOp, range.Start.getPointer());
  }
  return {SMLoc::getFromPointer(startOp), SMLoc::getFromPointer(endOp)};
}

/// Helper to simplify rewriting the source file.
class RewritePad {
public:
  static std::unique_ptr<RewritePad> init(StringRef inputFilename,
                                          StringRef outputFilename);

  /// Return the context the file was parsed into.
  MLIRContext *getContext() { return &context; }

  /// Return the OperationDefinition's of the operation's parsed.
  iterator_range<AsmParserState::OperationDefIterator> getOpDefs() {
    return asmState.getOpDefs();
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

  /// Replace the range of the operation in the source text with the
  /// corresponding string in the output.
  void replaceDef(const OperationDefinition &opDef, StringRef newDef) {
    replaceRange(getOpRange(opDef), newDef);
  }

  /// Return the source string corresponding to the source range.
  StringRef getSourceString(SMRange range) {
    return StringRef(range.Start.getPointer(),
                     range.End.getPointer() - range.Start.getPointer());
  }

  /// Return the source string corresponding to operation definition.
  StringRef getSourceString(const OperationDefinition &opDef) {
    auto range = getOpRange(opDef);
    return getSourceString(range);
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

  /// Return lines that are purely comments.
  SmallVector<SMRange> getSingleLineComments() {
    unsigned curBuf = sourceMgr.getMainFileID();
    const llvm::MemoryBuffer *curMB = sourceMgr.getMemoryBuffer(curBuf);
    llvm::line_iterator lineIterator(*curMB);
    SmallVector<SMRange> ret;
    for (; !lineIterator.is_at_end(); ++lineIterator) {
      StringRef trimmed = lineIterator->ltrim();
      if (trimmed.starts_with("//")) {
        ret.emplace_back(
            SMLoc::getFromPointer(trimmed.data()),
            SMLoc::getFromPointer(trimmed.data() + trimmed.size()));
      }
    }
    return ret;
  }

  /// Return the IR from parsed file.
  Block *getParsed() { return &parsedIR; }

  /// Return the definition for the given operation, or nullptr if the given
  /// operation does not have a definition.
  const OperationDefinition &getOpDef(Operation *op) const {
    return *asmState.getOpDef(op);
  }

private:
  // The context and state required to parse.
  MLIRContext context;
  llvm::SourceMgr sourceMgr;
  DialectRegistry registry;
  FallbackAsmResourceMap fallbackResourceMap;

  // Storage of textual parsing results.
  AsmParserState asmState;

  // Parsed IR.
  Block parsedIR;

  // The RewriteBuffer  is doing most of the real work.
  llvm::RewriteBuffer rewriteBuffer;

  // Start of the original input, used to compute offset.
  const char *start;
};

std::unique_ptr<RewritePad> RewritePad::init(StringRef inputFilename,
                                             StringRef outputFilename) {
  std::unique_ptr<RewritePad> r = std::make_unique<RewritePad>();

  // Register all the dialects needed.
  registerAllDialects(r->registry);

  // Set up the input file.
  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> file =
      openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }
  r->sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());

  // Set up the MLIR context and error handling.
  r->context.appendDialectRegistry(r->registry);

  // Record the start of the buffer to compute offsets with.
  unsigned curBuf = r->sourceMgr.getMainFileID();
  const llvm::MemoryBuffer *curMB = r->sourceMgr.getMemoryBuffer(curBuf);
  r->start = curMB->getBufferStart();
  r->rewriteBuffer.Initialize(curMB->getBuffer());

  // Parse and populate the AsmParserState.
  ParserConfig parseConfig(&r->context, /*verifyAfterParse=*/true,
                           &r->fallbackResourceMap);
  // Always allow unregistered.
  r->context.allowUnregisteredDialects(true);
  if (failed(parseAsmSourceFile(r->sourceMgr, &r->parsedIR, parseConfig,
                                &r->asmState)))
    return nullptr;

  return r;
}

/// Return the source code associated with the operation name.
static SMRange getOpNameRange(const OperationDefinition &op) { return op.loc; }

/// Return whether the operation was printed using generic syntax in original
/// buffer.
static bool isGeneric(const OperationDefinition &op) {
  return op.loc.Start.getPointer()[0] == '"';
}

static inline int asMainReturnCode(LogicalResult r) {
  return r.succeeded() ? EXIT_SUCCESS : EXIT_FAILURE;
}

/// Reriter function to invoke.
using RewriterFunction = std::function<mlir::LogicalResult(
    mlir::RewritePad &rewriteState, llvm::raw_ostream &os)>;

/// Structure to group information about a rewriter (argument to invoke via
/// mlir-tblgen, description, and rewriter function).
class RewriterInfo {
public:
  /// RewriterInfo constructor should not be invoked directly, instead use
  /// RewriterRegistration or registerRewriter.
  RewriterInfo(StringRef arg, StringRef description, RewriterFunction rewriter)
      : arg(arg), description(description), rewriter(std::move(rewriter)) {}

  /// Invokes the rewriter and returns whether the rewriter failed.
  LogicalResult invoke(mlir::RewritePad &rewriteState, raw_ostream &os) const {
    assert(rewriter && "Cannot call rewriter with null rewriter");
    return rewriter(rewriteState, os);
  }

  /// Returns the command line option that may be passed to 'mlir-rewrite' to
  /// invoke this rewriter.
  StringRef getRewriterArgument() const { return arg; }

  /// Returns a description for the rewriter.
  StringRef getRewriterDescription() const { return description; }

private:
  // The argument with which to invoke the rewriter via mlir-tblgen.
  StringRef arg;

  // Description of the rewriter.
  StringRef description;

  // Rewritererator function.
  RewriterFunction rewriter;
};

static llvm::ManagedStatic<std::vector<RewriterInfo>> rewriterRegistry;

/// Adds command line option for each registered rewriter.
struct RewriterNameParser : public llvm::cl::parser<const RewriterInfo *> {
  RewriterNameParser(llvm::cl::Option &opt);

  void printOptionInfo(const llvm::cl::Option &o,
                       size_t globalWidth) const override;
};

/// RewriterRegistration provides a global initializer that registers a rewriter
/// function.
struct RewriterRegistration {
  RewriterRegistration(StringRef arg, StringRef description,
                       const RewriterFunction &function);
};

RewriterRegistration::RewriterRegistration(StringRef arg, StringRef description,
                                           const RewriterFunction &function) {
  rewriterRegistry->emplace_back(arg, description, function);
}

RewriterNameParser::RewriterNameParser(llvm::cl::Option &opt)
    : llvm::cl::parser<const RewriterInfo *>(opt) {
  for (const auto &kv : *rewriterRegistry) {
    addLiteralOption(kv.getRewriterArgument(), &kv,
                     kv.getRewriterDescription());
  }
}

void RewriterNameParser::printOptionInfo(const llvm::cl::Option &o,
                                         size_t globalWidth) const {
  RewriterNameParser *tp = const_cast<RewriterNameParser *>(this);
  llvm::array_pod_sort(tp->Values.begin(), tp->Values.end(),
                       [](const RewriterNameParser::OptionInfo *vT1,
                          const RewriterNameParser::OptionInfo *vT2) {
                         return vT1->Name.compare(vT2->Name);
                       });
  using llvm::cl::parser;
  parser<const RewriterInfo *>::printOptionInfo(o, globalWidth);
}

} // namespace mlir

// TODO: Make these injectable too in non-global way.
static llvm::cl::OptionCategory clSimpleRenameCategory{"simple-rename options"};
static llvm::cl::opt<std::string> simpleRenameOpName{
    "simple-rename-op-name", llvm::cl::desc("Name of op to match on"),
    llvm::cl::cat(clSimpleRenameCategory)};
static llvm::cl::opt<std::string> simpleRenameMatch{
    "simple-rename-match", llvm::cl::desc("Match string for rename"),
    llvm::cl::cat(clSimpleRenameCategory)};
static llvm::cl::opt<std::string> simpleRenameReplace{
    "simple-rename-replace", llvm::cl::desc("Replace string for rename"),
    llvm::cl::cat(clSimpleRenameCategory)};

// Rewriter that does simple renames.
static LogicalResult simpleRename(RewritePad &rewriteState, raw_ostream &os) {
  StringRef opName = simpleRenameOpName;
  StringRef match = simpleRenameMatch;
  StringRef replace = simpleRenameReplace;
  llvm::Regex regex(match);

  rewriteState.getParsed()->walk([&](Operation *op) {
    if (op->getName().getStringRef() != opName)
      return;

    const OperationDefinition &opDef = rewriteState.getOpDef(op);
    SMRange range = getOpRange(opDef);
    // This is a little bit overkill for simple.
    std::string str = regex.sub(replace, rewriteState.getSourceString(range));
    rewriteState.replaceRange(range, str);
  });
  return success();
}

static mlir::RewriterRegistration rewriteSimpleRename("simple-rename",
                                                      "Perform a simple rename",
                                                      simpleRename);

// Rewriter that insert range markers.
static LogicalResult markRanges(RewritePad &rewriteState, raw_ostream &os) {
  for (const auto &it : rewriteState.getOpDefs()) {
    auto [startOp, endOp] = getOpRange(it);

    rewriteState.insertText(startOp, "<");
    rewriteState.insertText(endOp, ">");

    auto nameRange = getOpNameRange(it);

    if (isGeneric(it)) {
      rewriteState.insertText(nameRange.Start, "[");
      rewriteState.insertText(nameRange.End, "]");
    } else {
      rewriteState.insertText(nameRange.Start, "![");
      rewriteState.insertText(nameRange.End, "]!");
    }
  }

  // Highlight all comment lines.
  // TODO: Could be replaced if this is kept in memory.
  for (auto commentLine : rewriteState.getSingleLineComments()) {
    rewriteState.insertText(commentLine.Start, "{");
    rewriteState.insertText(commentLine.End, "}");
  }

  return success();
}

static mlir::RewriterRegistration
    rewriteMarkRanges("mark-ranges", "Indicate ranges parsed", markRanges);

int main(int argc, char **argv) {
  llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                           llvm::cl::desc("<input file>"),
                                           llvm::cl::init("-"));

  llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  llvm::cl::opt<const mlir::RewriterInfo *, false, mlir::RewriterNameParser>
      rewriter("", llvm::cl::desc("Rewriter to run"));

  std::string helpHeader = "mlir-rewrite";

  llvm::cl::ParseCommandLineOptions(argc, argv, helpHeader);

  // If no rewriter has been selected, exit with error code. Could also just
  // return but its unlikely this was intentionally being used as `cp`.
  if (!rewriter) {
    llvm::errs() << "No rewriter selected!\n";
    return mlir::asMainReturnCode(mlir::failure());
  }

  // Set up rewrite buffer.
  auto rewriterOr = RewritePad::init(inputFilename, outputFilename);
  if (!rewriterOr)
    return mlir::asMainReturnCode(mlir::failure());

  // Set up the output file.
  std::string errorMessage;
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return mlir::asMainReturnCode(mlir::failure());
  }

  LogicalResult result = rewriter->invoke(*rewriterOr, output->os());
  if (succeeded(result)) {
    rewriterOr->write(output->os());
    output->keep();
  }
  return mlir::asMainReturnCode(result);
}
