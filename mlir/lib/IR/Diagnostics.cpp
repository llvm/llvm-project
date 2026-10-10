//===- Diagnostics.cpp - MLIR Diagnostics ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/SourceMgrDiagnosticVerifier.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// DiagnosticArgument
//===----------------------------------------------------------------------===//

/// Construct from an Attribute.
DiagnosticArgument::DiagnosticArgument(Attribute attr)
    : kind(DiagnosticArgumentKind::Attribute),
      opaqueVal(reinterpret_cast<intptr_t>(attr.getAsOpaquePointer())) {}

/// Construct from a Type.
DiagnosticArgument::DiagnosticArgument(Type val)
    : kind(DiagnosticArgumentKind::Type),
      opaqueVal(reinterpret_cast<intptr_t>(val.getAsOpaquePointer())) {}

/// Returns this argument as an Attribute.
Attribute DiagnosticArgument::getAsAttribute() const {
  assert(getKind() == DiagnosticArgumentKind::Attribute);
  return Attribute::getFromOpaquePointer(
      reinterpret_cast<const void *>(opaqueVal));
}

/// Returns this argument as a Type.
Type DiagnosticArgument::getAsType() const {
  assert(getKind() == DiagnosticArgumentKind::Type);
  return Type::getFromOpaquePointer(reinterpret_cast<const void *>(opaqueVal));
}

/// Outputs this argument to a stream.
void DiagnosticArgument::print(raw_ostream &os) const {
  switch (kind) {
  case DiagnosticArgumentKind::Attribute:
    os << getAsAttribute();
    break;
  case DiagnosticArgumentKind::Double:
    os << getAsDouble();
    break;
  case DiagnosticArgumentKind::Integer:
    os << getAsInteger();
    break;
  case DiagnosticArgumentKind::String:
    os << getAsString();
    break;
  case DiagnosticArgumentKind::Type:
    os << '\'' << getAsType() << '\'';
    break;
  case DiagnosticArgumentKind::Unsigned:
    os << getAsUnsigned();
    break;
  }
}

//===----------------------------------------------------------------------===//
// Diagnostic
//===----------------------------------------------------------------------===//

/// Convert a Twine to a StringRef. Memory used for generating the StringRef is
/// stored in 'strings'.
static StringRef
twineToStrRef(const Twine &val,
              SmallVectorImpl<std::unique_ptr<char[]>> &strings) {
  // Allocate memory to hold this string.
  SmallString<64> data;
  auto strRef = val.toStringRef(data);
  if (strRef.empty())
    return strRef;

  strings.push_back(std::unique_ptr<char[]>(new char[strRef.size()]));
  memcpy(&strings.back()[0], strRef.data(), strRef.size());
  // Return a reference to the new string.
  return StringRef(&strings.back()[0], strRef.size());
}

/// Stream in a Twine argument.
Diagnostic &Diagnostic::operator<<(char val) { return *this << Twine(val); }
Diagnostic &Diagnostic::operator<<(const Twine &val) {
  arguments.push_back(DiagnosticArgument(twineToStrRef(val, strings)));
  return *this;
}
Diagnostic &Diagnostic::operator<<(Twine &&val) {
  arguments.push_back(DiagnosticArgument(twineToStrRef(val, strings)));
  return *this;
}

Diagnostic &Diagnostic::operator<<(StringAttr val) {
  arguments.push_back(DiagnosticArgument(val));
  return *this;
}

/// Stream in an OperationName.
Diagnostic &Diagnostic::operator<<(OperationName val) {
  // An OperationName is stored in the context, so we don't need to worry about
  // the lifetime of its data.
  arguments.push_back(DiagnosticArgument(val.getStringRef()));
  return *this;
}

/// Adjusts operation printing flags used in diagnostics for the given severity
/// level.
static OpPrintingFlags adjustPrintingFlags(OpPrintingFlags flags,
                                           DiagnosticSeverity severity) {
  flags.useLocalScope();
  flags.elideLargeElementsAttrs();
  if (severity == DiagnosticSeverity::Error)
    flags.printGenericOpForm();
  return flags;
}

/// Stream in an Operation.
Diagnostic &Diagnostic::operator<<(Operation &op) {
  return appendOp(op, OpPrintingFlags());
}

Diagnostic &Diagnostic::operator<<(OpWithFlags op) {
  return appendOp(*op.getOperation(), op.flags());
}

Diagnostic &Diagnostic::appendOp(Operation &op, const OpPrintingFlags &flags) {
  std::string str;
  llvm::raw_string_ostream os(str);
  op.print(os, adjustPrintingFlags(flags, severity));
  // Print on a new line for better readability if the op will be printed on
  // multiple lines.
  if (str.find('\n') != std::string::npos)
    *this << '\n';
  return *this << str;
}

/// Stream in a Value.
Diagnostic &Diagnostic::operator<<(Value val) {
  std::string str;
  llvm::raw_string_ostream os(str);
  val.print(os, adjustPrintingFlags(OpPrintingFlags(), severity));
  return *this << str;
}

/// Outputs this diagnostic to a stream.
void Diagnostic::print(raw_ostream &os,
                       std::optional<int64_t> messagePartIndex) const {
  if (!messagePartIndex.has_value()) {
    for (auto &arg : getArguments())
      arg.print(os);
    return;
  }

  assert(0 <= *messagePartIndex &&
         *messagePartIndex <= static_cast<int64_t>(messagePartEnds.size()));
  size_t argumentStart =
      *messagePartIndex == 0 ? 0 : messagePartEnds[*messagePartIndex - 1];
  size_t argumentEnd =
      *messagePartIndex == static_cast<int64_t>(messagePartEnds.size())
          ? arguments.size()
          : messagePartEnds[*messagePartIndex];
  for (auto &arg :
       getArguments().slice(argumentStart, argumentEnd - argumentStart))
    arg.print(os);
}

/// Convert the diagnostic to a string.
std::string Diagnostic::str() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  print(os);
  return str;
}

/// Converts each message part to a separate string.
SmallVector<std::string> Diagnostic::strs() const {
  SmallVector<std::string> strs;
  size_t numMessageParts = messagePartEnds.size();

  // Include the current message part if there are no completed parts or if it
  // contains arguments after the last completed part.
  if (messagePartEnds.empty() || messagePartEnds.back() != arguments.size())
    ++numMessageParts;
  for (size_t i = 0; i < numMessageParts; ++i) {
    std::string str;
    llvm::raw_string_ostream os(str);
    print(os, i);
    strs.push_back(str);
  }
  return strs;
}

/// Attaches a note to this diagnostic. A new location may be optionally
/// provided, if not, then the location defaults to the one specified for this
/// diagnostic. Notes may not be attached to other notes.
Diagnostic &Diagnostic::attachNote(std::optional<Location> noteLoc) {
  // We don't allow attaching notes to notes.
  assert(severity != DiagnosticSeverity::Note &&
         "cannot attach a note to a note");

  // If a location wasn't provided then reuse our location.
  if (!noteLoc)
    noteLoc = loc;

  /// Append and return a new note.
  notes.push_back(
      std::make_unique<Diagnostic>(*noteLoc, DiagnosticSeverity::Note));
  return *notes.back();
}

/// Allow a diagnostic to be converted to 'failure'.
Diagnostic::operator LogicalResult() const { return failure(); }

/// Starts a new message part.
void Diagnostic::startNewMessagePart() {
  if (arguments.empty())
    return;
  if (!messagePartEnds.empty() && messagePartEnds.back() == arguments.size())
    return;
  messagePartEnds.push_back(arguments.size());
}

//===----------------------------------------------------------------------===//
// InFlightDiagnostic
//===----------------------------------------------------------------------===//

/// Allow an inflight diagnostic to be converted to 'failure', otherwise
/// 'success' if this is an empty diagnostic.
InFlightDiagnostic::operator LogicalResult() const {
  return failure(isActive());
}

/// Reports the diagnostic to the engine.
void InFlightDiagnostic::report() {
  // If this diagnostic is still inflight and it hasn't been abandoned, then
  // report it.
  if (isInFlight()) {
    owner->emit(std::move(*impl));
    owner = nullptr;
  }
  impl.reset();
}

/// Abandons this diagnostic.
void InFlightDiagnostic::abandon() { owner = nullptr; }

//===----------------------------------------------------------------------===//
// DiagnosticEngineImpl
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
struct DiagnosticEngineImpl {
  /// Emit a diagnostic using the registered issue handle if present, or with
  /// the default behavior if not.
  void emit(Diagnostic &&diag);

  /// A mutex to ensure that diagnostics emission is thread-safe.
  llvm::sys::SmartMutex<true> mutex;

  /// These are the handlers used to report diagnostics.
  llvm::SmallMapVector<DiagnosticEngine::HandlerID, DiagnosticEngine::HandlerTy,
                       2>
      handlers;

  /// This is a unique identifier counter for diagnostic handlers in the
  /// context. This id starts at 1 to allow for 0 to be used as a sentinel.
  DiagnosticEngine::HandlerID uniqueHandlerId = 1;
};
} // namespace detail
} // namespace mlir

/// Emit a diagnostic using the registered issue handle if present, or with
/// the default behavior if not.
void DiagnosticEngineImpl::emit(Diagnostic &&diag) {
  llvm::sys::SmartScopedLock<true> lock(mutex);

  // Try to process the given diagnostic on one of the registered handlers.
  // Handlers are walked in reverse order, so that the most recent handler is
  // processed first.
  for (auto &handlerIt : llvm::reverse(handlers))
    if (succeeded(handlerIt.second(diag)))
      return;

  // Otherwise, if this is an error we emit it to stderr.
  if (diag.getSeverity() != DiagnosticSeverity::Error)
    return;

  auto &os = llvm::errs();
  if (!llvm::isa<UnknownLoc>(diag.getLocation()))
    os << diag.getLocation() << ": ";
  os << "error: ";

  // The default behavior for errors is to emit them to stderr.
  os << diag << '\n';
  os.flush();
}

//===----------------------------------------------------------------------===//
// DiagnosticEngine
//===----------------------------------------------------------------------===//

DiagnosticEngine::DiagnosticEngine() : impl(new DiagnosticEngineImpl()) {}
DiagnosticEngine::~DiagnosticEngine() = default;

/// Register a new handler for diagnostics to the engine. This function returns
/// a unique identifier for the registered handler, which can be used to
/// unregister this handler at a later time.
auto DiagnosticEngine::registerHandler(HandlerTy handler) -> HandlerID {
  llvm::sys::SmartScopedLock<true> lock(impl->mutex);
  auto uniqueID = impl->uniqueHandlerId++;
  impl->handlers.insert({uniqueID, std::move(handler)});
  return uniqueID;
}

/// Erase the registered diagnostic handler with the given identifier.
void DiagnosticEngine::eraseHandler(HandlerID handlerID) {
  llvm::sys::SmartScopedLock<true> lock(impl->mutex);
  impl->handlers.erase(handlerID);
}

/// Emit a diagnostic using the registered issue handler if present, or with
/// the default behavior if not.
void DiagnosticEngine::emit(Diagnostic &&diag) {
  assert(diag.getSeverity() != DiagnosticSeverity::Note &&
         "notes should not be emitted directly");
  impl->emit(std::move(diag));
}

/// Helper function used to emit a diagnostic with an optionally empty twine
/// message. If the message is empty, then it is not inserted into the
/// diagnostic.
static InFlightDiagnostic
emitDiag(Location location, DiagnosticSeverity severity, const Twine &message) {
  MLIRContext *ctx = location->getContext();
  auto &diagEngine = ctx->getDiagEngine();
  auto diag = diagEngine.emit(location, severity);
  if (!message.isTriviallyEmpty())
    diag << message;

  // Add the stack trace as a note if necessary.
  if (ctx->shouldPrintStackTraceOnDiagnostic()) {
    std::string bt;
    {
      llvm::raw_string_ostream stream(bt);
      llvm::sys::PrintStackTrace(stream);
    }
    if (!bt.empty())
      diag.attachNote() << "diagnostic emitted with trace:\n" << bt;
  }

  return diag;
}

/// Emit an error message using this location.
InFlightDiagnostic mlir::emitError(Location loc) { return emitError(loc, {}); }
InFlightDiagnostic mlir::emitError(Location loc, const Twine &message) {
  return emitDiag(loc, DiagnosticSeverity::Error, message);
}

/// Emit a warning message using this location.
InFlightDiagnostic mlir::emitWarning(Location loc) {
  return emitWarning(loc, {});
}
InFlightDiagnostic mlir::emitWarning(Location loc, const Twine &message) {
  return emitDiag(loc, DiagnosticSeverity::Warning, message);
}

/// Emit a remark message using this location.
InFlightDiagnostic mlir::emitRemark(Location loc) {
  return emitRemark(loc, {});
}
InFlightDiagnostic mlir::emitRemark(Location loc, const Twine &message) {
  return emitDiag(loc, DiagnosticSeverity::Remark, message);
}

//===----------------------------------------------------------------------===//
// ScopedDiagnosticHandler
//===----------------------------------------------------------------------===//

ScopedDiagnosticHandler::~ScopedDiagnosticHandler() {
  if (handlerID)
    ctx->getDiagEngine().eraseHandler(handlerID);
}

//===----------------------------------------------------------------------===//
// SourceMgrDiagnosticHandler
//===----------------------------------------------------------------------===//
namespace mlir {
namespace detail {
struct SourceMgrDiagnosticHandlerImpl {
  /// Return the SrcManager buffer id for the specified file, or zero if none
  /// can be found.
  unsigned getSourceMgrBufferIDForFile(llvm::SourceMgr &mgr,
                                       StringRef filename) {
    // Check for an existing mapping to the buffer id for this file.
    auto bufferIt = filenameToBufId.find(filename);
    if (bufferIt != filenameToBufId.end())
      return bufferIt->second;

    // Look for a buffer in the manager that has this filename.
    for (unsigned i = 1, e = mgr.getNumBuffers() + 1; i != e; ++i) {
      auto *buf = mgr.getMemoryBuffer(i);
      if (buf->getBufferIdentifier() == filename)
        return filenameToBufId[filename] = i;
    }

    // Otherwise, try to load the source file.
    std::string ignored;
    unsigned id = mgr.AddIncludeFile(std::string(filename), SMLoc(), ignored);
    filenameToBufId[filename] = id;
    return id;
  }

  /// Mapping between file name and buffer ID's.
  llvm::StringMap<unsigned> filenameToBufId;
};
} // namespace detail
} // namespace mlir

/// Return a processable CallSiteLoc from the given location.
static std::optional<CallSiteLoc> getCallSiteLoc(Location loc) {
  if (isa<NameLoc>(loc))
    return getCallSiteLoc(cast<NameLoc>(loc).getChildLoc());
  if (auto callLoc = dyn_cast<CallSiteLoc>(loc))
    return callLoc;
  if (isa<FusedLoc>(loc)) {
    for (auto subLoc : cast<FusedLoc>(loc).getLocations()) {
      if (auto callLoc = getCallSiteLoc(subLoc)) {
        return callLoc;
      }
    }
    return std::nullopt;
  }
  return std::nullopt;
}

/// Given a diagnostic kind, returns the LLVM DiagKind.
static llvm::SourceMgr::DiagKind getDiagKind(DiagnosticSeverity kind) {
  switch (kind) {
  case DiagnosticSeverity::Note:
    return llvm::SourceMgr::DK_Note;
  case DiagnosticSeverity::Warning:
    return llvm::SourceMgr::DK_Warning;
  case DiagnosticSeverity::Error:
    return llvm::SourceMgr::DK_Error;
  case DiagnosticSeverity::Remark:
    return llvm::SourceMgr::DK_Remark;
  }
  llvm_unreachable("Unknown DiagnosticSeverity");
}

SourceMgrDiagnosticHandler::SourceMgrDiagnosticHandler(
    llvm::SourceMgr &mgr, MLIRContext *ctx, raw_ostream &os,
    ShouldShowLocFn &&shouldShowLocFn)
    : ScopedDiagnosticHandler(ctx), mgr(mgr), os(os),
      shouldShowLocFn(std::move(shouldShowLocFn)),
      impl(new SourceMgrDiagnosticHandlerImpl()) {
  setHandler([this](Diagnostic &diag) { emitDiagnostic(diag); });
}

SourceMgrDiagnosticHandler::SourceMgrDiagnosticHandler(
    llvm::SourceMgr &mgr, MLIRContext *ctx, ShouldShowLocFn &&shouldShowLocFn)
    : SourceMgrDiagnosticHandler(mgr, ctx, llvm::errs(),
                                 std::move(shouldShowLocFn)) {}

SourceMgrDiagnosticHandler::~SourceMgrDiagnosticHandler() = default;

void SourceMgrDiagnosticHandler::emitDiagnostic(Location loc, Twine message,
                                                DiagnosticSeverity kind,
                                                bool displaySourceLine) {
  // Extract a file location from this loc.
  auto fileLoc = loc->findInstanceOf<FileLineColLoc>();

  // If one doesn't exist, then print the raw message without a source location.
  if (!fileLoc) {
    std::string str;
    llvm::raw_string_ostream strOS(str);
    if (!llvm::isa<UnknownLoc>(loc))
      strOS << loc << ": ";
    strOS << message;
    return mgr.PrintMessage(os, SMLoc(), getDiagKind(kind), str);
  }

  // Otherwise if we are displaying the source line, try to convert the file
  // location to an SMLoc.
  if (displaySourceLine) {
    auto smloc = convertLocToSMLoc(fileLoc);
    if (smloc.isValid())
      return mgr.PrintMessage(os, smloc, getDiagKind(kind), message);
  }

  // If the conversion was unsuccessful, create a diagnostic with the file
  // information. We manually combine the line and column to avoid asserts in
  // the constructor of SMDiagnostic that takes a location.
  std::string locStr;
  llvm::raw_string_ostream locOS(locStr);
  locOS << fileLoc.getFilename().getValue() << ":" << fileLoc.getLine() << ":"
        << fileLoc.getColumn();
  llvm::SMDiagnostic diag(locStr, getDiagKind(kind), message.str());
  diag.print(nullptr, os);
}

/// Emit the given diagnostic with the held source manager.
void SourceMgrDiagnosticHandler::emitDiagnostic(Diagnostic &diag) {
  SmallVector<std::pair<Location, StringRef>> locationStack;
  auto addLocToStack = [&](Location loc, StringRef locContext) {
    if (std::optional<Location> showableLoc = findLocToShow(loc))
      locationStack.emplace_back(*showableLoc, locContext);
  };

  // Add locations to display for this diagnostic.
  Location loc = diag.getLocation();
  addLocToStack(loc, /*locContext=*/{});

  // If the diagnostic location was a call site location, add the call stack as
  // well.
  if (auto callLoc = getCallSiteLoc(loc)) {
    // Print the call stack while valid, or until the limit is reached.
    loc = callLoc->getCaller();
    for (unsigned curDepth = 0; curDepth < callStackLimit; ++curDepth) {
      addLocToStack(loc, "called from");
      if ((callLoc = getCallSiteLoc(loc)))
        loc = callLoc->getCaller();
      else
        break;
    }
  }

  // If the location stack is empty, use the initial location.
  if (locationStack.empty()) {
    for (const std::string &str : diag.strs())
      emitDiagnostic(diag.getLocation(), str, diag.getSeverity());

    // Otherwise, use the location stack.
  } else {
    for (const std::string &str : diag.strs())
      emitDiagnostic(locationStack.front().first, str, diag.getSeverity());
    for (auto &it : llvm::drop_begin(locationStack))
      emitDiagnostic(it.first, it.second, DiagnosticSeverity::Note);
  }

  // Emit each of the notes. Only display the source code if the location is
  // different from the previous location.
  for (auto &note : diag.getNotes()) {
    emitDiagnostic(note.getLocation(), note.str(), note.getSeverity(),
                   /*displaySourceLine=*/loc != note.getLocation());
    loc = note.getLocation();
  }
}

void SourceMgrDiagnosticHandler::setCallStackLimit(unsigned limit) {
  callStackLimit = limit;
}

/// Get a memory buffer for the given file, or nullptr if one is not found.
const llvm::MemoryBuffer *
SourceMgrDiagnosticHandler::getBufferForFile(StringRef filename) {
  if (unsigned id = impl->getSourceMgrBufferIDForFile(mgr, filename))
    return mgr.getMemoryBuffer(id);
  return nullptr;
}

std::optional<Location>
SourceMgrDiagnosticHandler::findLocToShow(Location loc) {
  if (!shouldShowLocFn)
    return loc;
  if (!shouldShowLocFn(loc))
    return std::nullopt;

  // Recurse into the child locations of some of location types.
  return TypeSwitch<LocationAttr, std::optional<Location>>(loc)
      .Case([&](CallSiteLoc callLoc) -> std::optional<Location> {
        // We recurse into the callee of a call site, as the caller will be
        // emitted in a different note on the main diagnostic.
        return findLocToShow(callLoc.getCallee());
      })
      .Case([&](FileLineColLoc) -> std::optional<Location> { return loc; })
      .Case([&](FusedLoc fusedLoc) -> std::optional<Location> {
        // Fused location is unique in that we try to find a sub-location to
        // show, rather than the top-level location itself.
        for (Location childLoc : fusedLoc.getLocations())
          if (std::optional<Location> showableLoc = findLocToShow(childLoc))
            return showableLoc;
        return std::nullopt;
      })
      .Case([&](NameLoc nameLoc) -> std::optional<Location> {
        return findLocToShow(nameLoc.getChildLoc());
      })
      .Case([&](OpaqueLoc opaqueLoc) -> std::optional<Location> {
        // OpaqueLoc always falls back to a different source location.
        return findLocToShow(opaqueLoc.getFallbackLocation());
      })
      .Case([](UnknownLoc) -> std::optional<Location> {
        // Prefer not to show unknown locations.
        return std::nullopt;
      });
}

/// Get a memory buffer for the given file, or the main file of the source
/// manager if one doesn't exist. This always returns non-null.
SMLoc SourceMgrDiagnosticHandler::convertLocToSMLoc(FileLineColLoc loc) {
  // The column and line may be zero to represent unknown column and/or unknown
  /// line/column information.
  if (loc.getLine() == 0 || loc.getColumn() == 0)
    return SMLoc();

  unsigned bufferId = impl->getSourceMgrBufferIDForFile(mgr, loc.getFilename());
  if (!bufferId)
    return SMLoc();
  return mgr.FindLocForLineAndColumn(bufferId, loc.getLine(), loc.getColumn());
}

//===----------------------------------------------------------------------===//
// SourceMgrDiagnosticVerifierHandler
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
struct SourceMgrDiagnosticVerifierHandlerImpl {
  SourceMgrDiagnosticVerifierHandlerImpl(
      SourceMgrDiagnosticVerifierHandler::Level level)
      : level(level) {}

  /// The generic SourceMgr-based verifier core that this handler delegates
  /// scanning and matching of 'expected-*' diagnostics to.
  llvm::SourceMgrDiagnosticVerifier verifier;

  /// Verification level.
  SourceMgrDiagnosticVerifierHandler::Level level;
};
} // namespace detail
} // namespace mlir

/// Given a diagnostic kind, return a human readable string for it.
static StringRef getDiagKindStr(DiagnosticSeverity kind) {
  switch (kind) {
  case DiagnosticSeverity::Note:
    return "note";
  case DiagnosticSeverity::Warning:
    return "warning";
  case DiagnosticSeverity::Error:
    return "error";
  case DiagnosticSeverity::Remark:
    return "remark";
  }
  llvm_unreachable("Unknown DiagnosticSeverity");
}

SourceMgrDiagnosticVerifierHandler::SourceMgrDiagnosticVerifierHandler(
    llvm::SourceMgr &srcMgr, MLIRContext *ctx, raw_ostream &out, Level level)
    : SourceMgrDiagnosticHandler(srcMgr, ctx, out),
      impl(new SourceMgrDiagnosticVerifierHandlerImpl(level)) {
  // Compute the expected diagnostics for each of the current files in the
  // source manager.
  for (unsigned i = 0, e = mgr.getNumBuffers(); i != e; ++i)
    (void)impl->verifier.computeExpectedDiags(out, mgr,
                                              mgr.getMemoryBuffer(i + 1));

  // The base class registered a handler that prints every diagnostic. The
  // verifier takes its place: it consumes diagnostics and reports unexpected
  // ones itself.
  setHandler([this](Diagnostic &diag) { process(diag); });
}

SourceMgrDiagnosticVerifierHandler::SourceMgrDiagnosticVerifierHandler(
    llvm::SourceMgr &srcMgr, MLIRContext *ctx, Level level)
    : SourceMgrDiagnosticVerifierHandler(srcMgr, ctx, llvm::errs(), level) {}

SourceMgrDiagnosticVerifierHandler::~SourceMgrDiagnosticVerifierHandler() {
  // Ensure that all expected diagnostics were handled.
  (void)verify();
}

/// Returns the status of the verifier and verifies that all expected
/// diagnostics were emitted. This return success if all diagnostics were
/// verified correctly, failure otherwise.
LogicalResult SourceMgrDiagnosticVerifierHandler::verify() {
  return success(impl->verifier.verify(os, mgr));
}

std::unique_ptr<ScopedDiagnosticHandler>
SourceMgrDiagnosticVerifierHandler::registerInContext(MLIRContext *ctx) {
  return std::make_unique<ScopedDiagnosticHandler>(
      ctx, [this](Diagnostic &diag) { process(diag); });
}

/// Process a diagnostic and its notes.
void SourceMgrDiagnosticVerifierHandler::process(Diagnostic &diag) {
  for (const std::string &str : diag.strs())
    process(diag.getLocation(), str, diag.getSeverity());
  for (auto &note : diag.getNotes())
    process(note);
}

/// Process a diagnostic at a certain location.
void SourceMgrDiagnosticVerifierHandler::process(LocationAttr loc,
                                                 StringRef msg,
                                                 DiagnosticSeverity kind) {
  FileLineColLoc fileLoc = loc.findInstanceOf<FileLineColLoc>();
  const llvm::MemoryBuffer *buf =
      fileLoc ? getBufferForFile(fileLoc.getFilename()) : nullptr;
  bool reportUnexpected = impl->level != Level::OnlyExpected;

  using MatchResult = llvm::SourceMgrDiagnosticVerifier::MatchResult;
  MatchResult result = impl->verifier.process(
      os, mgr, getDiagKind(kind), /*hasLoc=*/static_cast<bool>(fileLoc), buf,
      fileLoc ? fileLoc.getLine() : 0, fileLoc ? fileLoc.getColumn() : 0, msg,
      reportUnexpected);
  if (result == MatchResult::Unexpected)
    emitDiagnostic(loc, "unexpected " + getDiagKindStr(kind) + ": " + msg,
                   DiagnosticSeverity::Error);
}

//===----------------------------------------------------------------------===//
// ParallelDiagnosticHandler
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
struct ParallelDiagnosticHandlerImpl : public llvm::PrettyStackTraceEntry {
  struct ThreadDiagnostic {
    ThreadDiagnostic(size_t id, Diagnostic diag)
        : id(id), diag(std::move(diag)) {}
    bool operator<(const ThreadDiagnostic &rhs) const { return id < rhs.id; }

    /// The id for this diagnostic, this is used for ordering.
    /// Note: This id corresponds to the ordered position of the current element
    ///       being processed by a given thread.
    size_t id;

    /// The diagnostic.
    Diagnostic diag;
  };

  ParallelDiagnosticHandlerImpl(MLIRContext *ctx) : context(ctx) {
    handlerID = ctx->getDiagEngine().registerHandler([this](Diagnostic &diag) {
      uint64_t tid = llvm::get_threadid();
      llvm::sys::SmartScopedLock<true> lock(mutex);

      // If this thread is not tracked, then return failure to let another
      // handler process this diagnostic.
      if (!threadToOrderID.count(tid))
        return failure();

      // Append a new diagnostic.
      diagnostics.emplace_back(threadToOrderID[tid], std::move(diag));
      return success();
    });
  }

  ~ParallelDiagnosticHandlerImpl() override {
    // Erase this handler from the context.
    context->getDiagEngine().eraseHandler(handlerID);

    // Early exit if there are no diagnostics, this is the common case.
    if (diagnostics.empty())
      return;

    // Emit the diagnostics back to the context.
    emitDiagnostics([&](Diagnostic &diag) {
      return context->getDiagEngine().emit(std::move(diag));
    });
  }

  /// Utility method to emit any held diagnostics.
  void emitDiagnostics(llvm::function_ref<void(Diagnostic &)> emitFn) const {
    // Stable sort all of the diagnostics that were emitted. This creates a
    // deterministic ordering for the diagnostics based upon which order id they
    // were emitted for.
    llvm::stable_sort(diagnostics);

    // Emit each diagnostic to the context again.
    for (ThreadDiagnostic &diag : diagnostics)
      emitFn(diag.diag);
  }

  /// Set the order id for the current thread.
  void setOrderIDForThread(size_t orderID) {
    uint64_t tid = llvm::get_threadid();
    llvm::sys::SmartScopedLock<true> lock(mutex);
    threadToOrderID[tid] = orderID;
  }

  /// Remove the order id for the current thread.
  void eraseOrderIDForThread() {
    uint64_t tid = llvm::get_threadid();
    llvm::sys::SmartScopedLock<true> lock(mutex);
    threadToOrderID.erase(tid);
  }

  /// Dump the current diagnostics that were inflight.
  void print(raw_ostream &os) const override {
    // Early exit if there are no diagnostics, this is the common case.
    if (diagnostics.empty())
      return;

    os << "In-Flight Diagnostics:\n";
    emitDiagnostics([&](const Diagnostic &diag) {
      os.indent(4);

      // Print each diagnostic with the format:
      //   "<location>: <kind>: <msg>"
      if (!llvm::isa<UnknownLoc>(diag.getLocation()))
        os << diag.getLocation() << ": ";
      switch (diag.getSeverity()) {
      case DiagnosticSeverity::Error:
        os << "error: ";
        break;
      case DiagnosticSeverity::Warning:
        os << "warning: ";
        break;
      case DiagnosticSeverity::Note:
        os << "note: ";
        break;
      case DiagnosticSeverity::Remark:
        os << "remark: ";
        break;
      }
      os << diag << '\n';
    });
  }

  /// A smart mutex to lock access to the internal state.
  llvm::sys::SmartMutex<true> mutex;

  /// A mapping between the thread id and the current order id.
  DenseMap<uint64_t, size_t> threadToOrderID;

  /// An unordered list of diagnostics that were emitted.
  mutable std::vector<ThreadDiagnostic> diagnostics;

  /// The unique id for the parallel handler.
  DiagnosticEngine::HandlerID handlerID = 0;

  /// The context to emit the diagnostics to.
  MLIRContext *context;
};
} // namespace detail
} // namespace mlir

ParallelDiagnosticHandler::ParallelDiagnosticHandler(MLIRContext *ctx)
    : impl(new ParallelDiagnosticHandlerImpl(ctx)) {}
ParallelDiagnosticHandler::~ParallelDiagnosticHandler() = default;

/// Set the order id for the current thread.
void ParallelDiagnosticHandler::setOrderIDForThread(size_t orderID) {
  impl->setOrderIDForThread(orderID);
}

/// Remove the order id for the current thread. This removes the thread from
/// diagnostics tracking.
void ParallelDiagnosticHandler::eraseOrderIDForThread() {
  impl->eraseOrderIDForThread();
}
