//===- PassRegistry.cpp - Pass Registration Utilities ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace detail;

/// Static mapping of all of the registered passes.
static llvm::ManagedStatic<DenseMap<const PassID *, PassInfo>> passRegistry;

/// Static mapping of all of the registered pass pipelines.
static llvm::ManagedStatic<llvm::StringMap<PassPipelineInfo>>
    passPipelineRegistry;

/// Utility to create a default registry function from a pass instance.
static PassRegistryFunction
buildDefaultRegistryFn(const PassAllocatorFunction &allocator) {
  return [=](OpPassManager &pm, StringRef options) {
    std::unique_ptr<Pass> pass = allocator();
    LogicalResult result = pass->initializeOptions(options);
    pm.addPass(std::move(pass));
    return result;
  };
}

/// Utility to print the help string for a specific option.
static void printOptionHelp(StringRef arg, StringRef desc, size_t indent,
                            size_t descIndent, bool isTopLevel) {
  size_t numSpaces = descIndent - indent - 4;
  llvm::outs().indent(indent)
      << "--" << llvm::left_justify(arg, numSpaces) << "-   " << desc << '\n';
}

//===----------------------------------------------------------------------===//
// PassRegistry
//===----------------------------------------------------------------------===//

/// Print the help information for this pass. This includes the argument,
/// description, and any pass options. `descIndent` is the indent that the
/// descriptions should be aligned.
void PassRegistryEntry::printHelpStr(size_t indent, size_t descIndent) const {
  printOptionHelp(getPassArgument(), getPassDescription(), indent, descIndent,
                  /*isTopLevel=*/true);
  // If this entry has options, print the help for those as well.
  optHandler([=](const PassOptions &options) {
    options.printHelp(indent, descIndent);
  });
}

/// Return the maximum width required when printing the options of this
/// entry.
size_t PassRegistryEntry::getOptionWidth() const {
  size_t maxLen = 0;
  optHandler([&](const PassOptions &options) mutable {
    maxLen = options.getOptionWidth() + 2;
  });
  return maxLen;
}

//===----------------------------------------------------------------------===//
// PassPipelineInfo
//===----------------------------------------------------------------------===//

void mlir::registerPassPipeline(
    StringRef arg, StringRef description, const PassRegistryFunction &function,
    std::function<void(function_ref<void(const PassOptions &)>)> optHandler) {
  PassPipelineInfo pipelineInfo(arg, description, function, optHandler);
  bool inserted = passPipelineRegistry->try_emplace(arg, pipelineInfo).second;
  assert(inserted && "Pass pipeline registered multiple times");
  (void)inserted;
}

//===----------------------------------------------------------------------===//
// PassInfo
//===----------------------------------------------------------------------===//

PassInfo::PassInfo(StringRef arg, StringRef description, const PassID *passID,
                   const PassAllocatorFunction &allocator)
    : PassRegistryEntry(
          arg, description, buildDefaultRegistryFn(allocator),
          // Use a temporary pass to provide an options instance.
          [=](function_ref<void(const PassOptions &)> optHandler) {
            optHandler(allocator()->passOptions);
          }) {}

void mlir::registerPass(StringRef arg, StringRef description,
                        const PassID *passID,
                        const PassAllocatorFunction &function) {
  PassInfo passInfo(arg, description, passID, function);
  bool inserted = passRegistry->try_emplace(passID, passInfo).second;
  assert(inserted && "Pass registered multiple times");
  (void)inserted;
}

/// Returns the pass info for the specified pass class or null if unknown.
const PassInfo *mlir::Pass::lookupPassInfo(const PassID *passID) {
  auto it = passRegistry->find(passID);
  if (it == passRegistry->end())
    return nullptr;
  return &it->getSecond();
}

//===----------------------------------------------------------------------===//
// PassOptions
//===----------------------------------------------------------------------===//

/// Out of line virtual function to provide home for the class.
void detail::PassOptions::OptionBase::anchor() {}

/// Copy the option values from 'other'.
void detail::PassOptions::copyOptionValuesFrom(const PassOptions &other) {
  assert(options.size() == other.options.size());
  if (options.empty())
    return;
  for (auto optionsIt : llvm::zip(options, other.options))
    std::get<0>(optionsIt)->copyValueFrom(*std::get<1>(optionsIt));
}

LogicalResult detail::PassOptions::parseFromString(StringRef options) {
  // TODO(parkers): Handle escaping strings.
  // NOTE: `options` is modified in place to always refer to the unprocessed
  // part of the string.
  while (!options.empty()) {
    size_t spacePos = options.find(' ');
    StringRef arg = options;
    if (spacePos != StringRef::npos) {
      arg = options.substr(0, spacePos);
      options = options.substr(spacePos + 1);
    } else {
      options = StringRef();
    }
    if (arg.empty())
      continue;

    // At this point, arg refers to everything that is non-space in options
    // upto the next space, and options refers to the rest of the string after
    // that point.

    // Split the individual option on '=' to form key and value. If there is no
    // '=', then value is `StringRef()`.
    size_t equalPos = arg.find('=');
    StringRef key = arg;
    StringRef value;
    if (equalPos != StringRef::npos) {
      key = arg.substr(0, equalPos);
      value = arg.substr(equalPos + 1);
    }
    auto it = OptionsMap.find(key);
    if (it == OptionsMap.end()) {
      llvm::errs() << "<Pass-Options-Parser>: no such option " << key << "\n";
      return failure();
    }
    if (llvm::cl::ProvidePositionalOption(it->second, value, 0))
      return failure();
  }

  return success();
}

/// Print the options held by this struct in a form that can be parsed via
/// 'parseFromString'.
void detail::PassOptions::print(raw_ostream &os) {
  // If there are no options, there is nothing left to do.
  if (OptionsMap.empty())
    return;

  // Sort the options to make the ordering deterministic.
  SmallVector<OptionBase *, 4> orderedOps(options.begin(), options.end());
  auto compareOptionArgs = [](OptionBase *const *lhs, OptionBase *const *rhs) {
    return (*lhs)->getArgStr().compare((*rhs)->getArgStr());
  };
  llvm::array_pod_sort(orderedOps.begin(), orderedOps.end(), compareOptionArgs);

  // Interleave the options with ' '.
  os << '{';
  interleave(
      orderedOps, os, [&](OptionBase *option) { option->print(os); }, " ");
  os << '}';
}

/// Print the help string for the options held by this struct. `descIndent` is
/// the indent within the stream that the descriptions should be aligned.
void detail::PassOptions::printHelp(size_t indent, size_t descIndent) const {
  // Sort the options to make the ordering deterministic.
  SmallVector<OptionBase *, 4> orderedOps(options.begin(), options.end());
  auto compareOptionArgs = [](OptionBase *const *lhs, OptionBase *const *rhs) {
    return (*lhs)->getArgStr().compare((*rhs)->getArgStr());
  };
  llvm::array_pod_sort(orderedOps.begin(), orderedOps.end(), compareOptionArgs);
  for (OptionBase *option : orderedOps) {
    // TODO(riverriddle) printOptionInfo assumes a specific indent and will
    // print options with values with incorrect indentation. We should add
    // support to llvm::cl::Option for passing in a base indent to use when
    // printing.
    llvm::outs().indent(indent);
    option->getOption()->printOptionInfo(descIndent - indent);
  }
}

/// Return the maximum width required when printing the help string.
size_t detail::PassOptions::getOptionWidth() const {
  size_t max = 0;
  for (auto *option : options)
    max = std::max(max, option->getOption()->getOptionWidth());
  return max;
}

//===----------------------------------------------------------------------===//
// TextualPassPipeline Parser
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a textual description of a pass pipeline.
class TextualPipeline {
public:
  /// Try to initialize this pipeline with the given pipeline text.
  /// `errorStream` is the output stream to emit errors to.
  LogicalResult initialize(StringRef text, raw_ostream &errorStream);

  /// Add the internal pipeline elements to the provided pass manager.
  LogicalResult addToPipeline(OpPassManager &pm) const;

private:
  /// A functor used to emit errors found during pipeline handling. The first
  /// parameter corresponds to the raw location within the pipeline string. This
  /// should always return failure.
  using ErrorHandlerT = function_ref<LogicalResult(const char *, Twine)>;

  /// A struct to capture parsed pass pipeline names.
  ///
  /// A pipeline is defined as a series of names, each of which may in itself
  /// recursively contain a nested pipeline. A name is either the name of a pass
  /// (e.g. "cse") or the name of an operation type (e.g. "func"). If the name
  /// is the name of a pass, the InnerPipeline is empty, since passes cannot
  /// contain inner pipelines.
  struct PipelineElement {
    PipelineElement(StringRef name) : name(name), registryEntry(nullptr) {}

    StringRef name;
    StringRef options;
    const PassRegistryEntry *registryEntry;
    std::vector<PipelineElement> innerPipeline;
  };

  /// Parse the given pipeline text into the internal pipeline vector. This
  /// function only parses the structure of the pipeline, and does not resolve
  /// its elements.
  LogicalResult parsePipelineText(StringRef text, ErrorHandlerT errorHandler);

  /// Resolve the elements of the pipeline, i.e. connect passes and pipelines to
  /// the corresponding registry entry.
  LogicalResult
  resolvePipelineElements(MutableArrayRef<PipelineElement> elements,
                          ErrorHandlerT errorHandler);

  /// Resolve a single element of the pipeline.
  LogicalResult resolvePipelineElement(PipelineElement &element,
                                       ErrorHandlerT errorHandler);

  /// Add the given pipeline elements to the provided pass manager.
  LogicalResult addToPipeline(ArrayRef<PipelineElement> elements,
                              OpPassManager &pm) const;

  std::vector<PipelineElement> pipeline;
};

} // end anonymous namespace

/// Try to initialize this pipeline with the given pipeline text. An option is
/// given to enable accurate error reporting.
LogicalResult TextualPipeline::initialize(StringRef text,
                                          raw_ostream &errorStream) {
  // Build a source manager to use for error reporting.
  llvm::SourceMgr pipelineMgr;
  pipelineMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(
                                     text, "MLIR Textual PassPipeline Parser"),
                                 llvm::SMLoc());
  auto errorHandler = [&](const char *rawLoc, Twine msg) {
    pipelineMgr.PrintMessage(errorStream, llvm::SMLoc::getFromPointer(rawLoc),
                             llvm::SourceMgr::DK_Error, msg);
    return failure();
  };

  // Parse the provided pipeline string.
  if (failed(parsePipelineText(text, errorHandler)))
    return failure();
  return resolvePipelineElements(pipeline, errorHandler);
}

/// Add the internal pipeline elements to the provided pass manager.
LogicalResult TextualPipeline::addToPipeline(OpPassManager &pm) const {
  return addToPipeline(pipeline, pm);
}

/// Parse the given pipeline text into the internal pipeline vector. This
/// function only parses the structure of the pipeline, and does not resolve
/// its elements.
LogicalResult TextualPipeline::parsePipelineText(StringRef text,
                                                 ErrorHandlerT errorHandler) {
  SmallVector<std::vector<PipelineElement> *, 4> pipelineStack = {&pipeline};
  for (;;) {
    std::vector<PipelineElement> &pipeline = *pipelineStack.back();
    size_t pos = text.find_first_of(",(){");
    pipeline.emplace_back(/*name=*/text.substr(0, pos).trim());

    // If we have a single terminating name, we're done.
    if (pos == text.npos)
      break;

    text = text.substr(pos);
    char sep = text[0];

    // Handle pulling ... from 'pass{...}' out as PipelineElement.options.
    if (sep == '{') {
      text = text.substr(1);

      // Skip over everything until the closing '}' and store as options.
      size_t close = text.find('}');

      // TODO(parkers): Handle skipping over quoted sub-strings.
      if (close == StringRef::npos) {
        return errorHandler(
            /*rawLoc=*/text.data() - 1,
            "missing closing '}' while processing pass options");
      }
      pipeline.back().options = text.substr(0, close);
      text = text.substr(close + 1);

      // Skip checking for '(' because nested pipelines cannot have options.
    } else if (sep == '(') {
      text = text.substr(1);

      // Push the inner pipeline onto the stack to continue processing.
      pipelineStack.push_back(&pipeline.back().innerPipeline);
      continue;
    }

    // When handling the close parenthesis, we greedily consume them to avoid
    // empty strings in the pipeline.
    while (text.consume_front(")")) {
      // If we try to pop the outer pipeline we have unbalanced parentheses.
      if (pipelineStack.size() == 1)
        return errorHandler(/*rawLoc=*/text.data() - 1,
                            "encountered extra closing ')' creating unbalanced "
                            "parentheses while parsing pipeline");

      pipelineStack.pop_back();
    }

    // Check if we've finished parsing.
    if (text.empty())
      break;

    // Otherwise, the end of an inner pipeline always has to be followed by
    // a comma, and then we can continue.
    if (!text.consume_front(","))
      return errorHandler(text.data(), "expected ',' after parsing pipeline");
  }

  // Check for unbalanced parentheses.
  if (pipelineStack.size() > 1)
    return errorHandler(
        text.data(),
        "encountered unbalanced parentheses while parsing pipeline");

  assert(pipelineStack.back() == &pipeline &&
         "wrong pipeline at the bottom of the stack");
  return success();
}

/// Resolve the elements of the pipeline, i.e. connect passes and pipelines to
/// the corresponding registry entry.
LogicalResult TextualPipeline::resolvePipelineElements(
    MutableArrayRef<PipelineElement> elements, ErrorHandlerT errorHandler) {
  for (auto &elt : elements)
    if (failed(resolvePipelineElement(elt, errorHandler)))
      return failure();
  return success();
}

/// Resolve a single element of the pipeline.
LogicalResult
TextualPipeline::resolvePipelineElement(PipelineElement &element,
                                        ErrorHandlerT errorHandler) {
  // If the inner pipeline of this element is not empty, this is an operation
  // pipeline.
  if (!element.innerPipeline.empty())
    return resolvePipelineElements(element.innerPipeline, errorHandler);

  // Otherwise, this must be a pass or pass pipeline.
  // Check to see if a pipeline was registered with this name.
  auto pipelineRegistryIt = passPipelineRegistry->find(element.name);
  if (pipelineRegistryIt != passPipelineRegistry->end()) {
    element.registryEntry = &pipelineRegistryIt->second;
    return success();
  }

  // If not, then this must be a specific pass name.
  for (auto &passIt : *passRegistry) {
    if (passIt.second.getPassArgument() == element.name) {
      element.registryEntry = &passIt.second;
      return success();
    }
  }

  // Emit an error for the unknown pass.
  auto *rawLoc = element.name.data();
  return errorHandler(rawLoc, "'" + element.name +
                                  "' does not refer to a "
                                  "registered pass or pass pipeline");
}

/// Add the given pipeline elements to the provided pass manager.
LogicalResult TextualPipeline::addToPipeline(ArrayRef<PipelineElement> elements,
                                             OpPassManager &pm) const {
  for (auto &elt : elements) {
    if (elt.registryEntry) {
      if (failed(elt.registryEntry->addToPipeline(pm, elt.options)))
        return failure();
    } else if (failed(addToPipeline(elt.innerPipeline, pm.nest(elt.name)))) {
      return failure();
    }
  }
  return success();
}

/// This function parses the textual representation of a pass pipeline, and adds
/// the result to 'pm' on success. This function returns failure if the given
/// pipeline was invalid. 'errorStream' is an optional parameter that, if
/// non-null, will be used to emit errors found during parsing.
LogicalResult mlir::parsePassPipeline(StringRef pipeline, OpPassManager &pm,
                                      raw_ostream &errorStream) {
  TextualPipeline pipelineParser;
  if (failed(pipelineParser.initialize(pipeline, errorStream)))
    return failure();
  if (failed(pipelineParser.addToPipeline(pm)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// PassNameParser
//===----------------------------------------------------------------------===//

namespace {
/// This struct represents the possible data entries in a parsed pass pipeline
/// list.
struct PassArgData {
  PassArgData() : registryEntry(nullptr) {}
  PassArgData(const PassRegistryEntry *registryEntry)
      : registryEntry(registryEntry) {}

  /// This field is used when the parsed option corresponds to a registered pass
  /// or pass pipeline.
  const PassRegistryEntry *registryEntry;

  /// This field is set when instance specific pass options have been provided
  /// on the command line.
  StringRef options;

  /// This field is used when the parsed option corresponds to an explicit
  /// pipeline.
  TextualPipeline pipeline;
};
} // end anonymous namespace

namespace llvm {
namespace cl {
/// Define a valid OptionValue for the command line pass argument.
template <>
struct OptionValue<PassArgData> final
    : OptionValueBase<PassArgData, /*isClass=*/true> {
  OptionValue(const PassArgData &value) { this->setValue(value); }
  OptionValue() = default;
  void anchor() override {}

  bool hasValue() const { return true; }
  const PassArgData &getValue() const { return value; }
  void setValue(const PassArgData &value) { this->value = value; }

  PassArgData value;
};
} // end namespace cl
} // end namespace llvm

namespace {

/// The name for the command line option used for parsing the textual pass
/// pipeline.
static constexpr StringLiteral passPipelineArg = "pass-pipeline";

/// Adds command line option for each registered pass or pass pipeline, as well
/// as textual pass pipelines.
struct PassNameParser : public llvm::cl::parser<PassArgData> {
  PassNameParser(llvm::cl::Option &opt) : llvm::cl::parser<PassArgData>(opt) {}

  void initialize();
  void printOptionInfo(const llvm::cl::Option &opt,
                       size_t globalWidth) const override;
  size_t getOptionWidth(const llvm::cl::Option &opt) const override;
  bool parse(llvm::cl::Option &opt, StringRef argName, StringRef arg,
             PassArgData &value);
};
} // namespace

void PassNameParser::initialize() {
  llvm::cl::parser<PassArgData>::initialize();

  /// Add an entry for the textual pass pipeline option.
  addLiteralOption(passPipelineArg, PassArgData(),
                   "A textual description of a pass pipeline to run");

  /// Add the pass entries.
  for (const auto &kv : *passRegistry) {
    addLiteralOption(kv.second.getPassArgument(), &kv.second,
                     kv.second.getPassDescription());
  }
  /// Add the pass pipeline entries.
  for (const auto &kv : *passPipelineRegistry) {
    addLiteralOption(kv.second.getPassArgument(), &kv.second,
                     kv.second.getPassDescription());
  }
}

void PassNameParser::printOptionInfo(const llvm::cl::Option &opt,
                                     size_t globalWidth) const {
  // Print the information for the top-level option.
  if (opt.hasArgStr()) {
    llvm::outs() << "  --" << opt.ArgStr;
    opt.printHelpStr(opt.HelpStr, globalWidth, opt.ArgStr.size() + 7);
  } else {
    llvm::outs() << "  " << opt.HelpStr << '\n';
  }

  // Print the top-level pipeline argument.
  printOptionHelp(passPipelineArg,
                  "A textual description of a pass pipeline to run",
                  /*indent=*/4, globalWidth, /*isTopLevel=*/!opt.hasArgStr());

  // Functor used to print the ordered entries of a registration map.
  auto printOrderedEntries = [&](StringRef header, auto &map) {
    llvm::SmallVector<PassRegistryEntry *, 32> orderedEntries;
    for (auto &kv : map)
      orderedEntries.push_back(&kv.second);
    llvm::array_pod_sort(
        orderedEntries.begin(), orderedEntries.end(),
        [](PassRegistryEntry *const *lhs, PassRegistryEntry *const *rhs) {
          return (*lhs)->getPassArgument().compare((*rhs)->getPassArgument());
        });

    llvm::outs().indent(4) << header << ":\n";
    for (PassRegistryEntry *entry : orderedEntries)
      entry->printHelpStr(/*indent=*/6, globalWidth);
  };

  // Print the available passes.
  printOrderedEntries("Passes", *passRegistry);

  // Print the available pass pipelines.
  if (!passPipelineRegistry->empty())
    printOrderedEntries("Pass Pipelines", *passPipelineRegistry);
}

size_t PassNameParser::getOptionWidth(const llvm::cl::Option &opt) const {
  size_t maxWidth = llvm::cl::parser<PassArgData>::getOptionWidth(opt) + 2;

  // Check for any wider pass or pipeline options.
  for (auto &entry : *passRegistry)
    maxWidth = std::max(maxWidth, entry.second.getOptionWidth() + 4);
  for (auto &entry : *passPipelineRegistry)
    maxWidth = std::max(maxWidth, entry.second.getOptionWidth() + 4);
  return maxWidth;
}

bool PassNameParser::parse(llvm::cl::Option &opt, StringRef argName,
                           StringRef arg, PassArgData &value) {
  // Handle the pipeline option explicitly.
  if (argName == passPipelineArg)
    return failed(value.pipeline.initialize(arg, llvm::errs()));

  // Otherwise, default to the base for handling.
  if (llvm::cl::parser<PassArgData>::parse(opt, argName, arg, value))
    return true;
  value.options = arg;
  return false;
}

//===----------------------------------------------------------------------===//
// PassPipelineCLParser
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
struct PassPipelineCLParserImpl {
  PassPipelineCLParserImpl(StringRef arg, StringRef description)
      : passList(arg, llvm::cl::desc(description)) {
    passList.setValueExpectedFlag(llvm::cl::ValueExpected::ValueOptional);
  }

  /// The set of passes and pass pipelines to run.
  llvm::cl::list<PassArgData, bool, PassNameParser> passList;
};
} // end namespace detail
} // end namespace mlir

/// Construct a pass pipeline parser with the given command line description.
PassPipelineCLParser::PassPipelineCLParser(StringRef arg, StringRef description)
    : impl(std::make_unique<detail::PassPipelineCLParserImpl>(arg,
                                                              description)) {}
PassPipelineCLParser::~PassPipelineCLParser() {}

/// Returns true if this parser contains any valid options to add.
bool PassPipelineCLParser::hasAnyOccurrences() const {
  return impl->passList.getNumOccurrences() != 0;
}

/// Returns true if the given pass registry entry was registered at the
/// top-level of the parser, i.e. not within an explicit textual pipeline.
bool PassPipelineCLParser::contains(const PassRegistryEntry *entry) const {
  return llvm::any_of(impl->passList, [&](const PassArgData &data) {
    return data.registryEntry == entry;
  });
}

/// Adds the passes defined by this parser entry to the given pass manager.
LogicalResult PassPipelineCLParser::addToPipeline(OpPassManager &pm) const {
  for (auto &passIt : impl->passList) {
    if (passIt.registryEntry) {
      if (failed(passIt.registryEntry->addToPipeline(pm, passIt.options)))
        return failure();
    } else if (failed(passIt.pipeline.addToPipeline(pm))) {
      return failure();
    }
  }
  return success();
}
