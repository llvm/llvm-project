//===- Format.cpp - Utilities for String Format ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities for formatting strings. They are specially
// tailored to the needs of TableGen'ing op definitions and rewrite rules,
// so they are not expected to be used as widely applicable utilities.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Format.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace mlir::tblgen;

// Marker to indicate an error happened when replacing a placeholder.
const char *const kMarkerForNoSubst = "<no-subst-found>";

char FmtError::ID;

FmtOrigin FmtOrigin::at(ArrayRef<llvm::SMLoc> locations,
                        const Twine &description) {
  FmtOrigin result;
  result.definition = {{locations.begin(), locations.end()}, description.str()};
  return result;
}

FmtOrigin &FmtOrigin::withUse(ArrayRef<llvm::SMLoc> locations,
                              const Twine &description) {
  uses.push_back({{locations.begin(), locations.end()}, description.str()});
  return *this;
}

FmtError::FmtError(StringRef format, SmallVector<FmtIssue> issues,
                   SmallVector<FmtBinding> bindings, size_t numPositionalArgs)
    : format(format.str()), issues(std::move(issues)),
      bindings(std::move(bindings)), numPositionalArgs(numPositionalArgs) {
  assert(!this->issues.empty() &&
         "formatting error without failed occurrences");
}

static void printBindings(raw_ostream &os, ArrayRef<FmtBinding> bindings) {
  for (const FmtBinding &binding : bindings) {
    os << '$' << binding.name << " = \"";
    os.write_escaped(binding.replacement, /*UseHexEscapes=*/true) << "\"\n";
  }
}

void FmtError::printBindings(raw_ostream &os) const {
  ::printBindings(os, bindings);
}

StringRef FmtIssue::getDescription() const {
  switch (kind) {
  case FmtIssueKind::MissingNamedBinding:
    return "missing named binding";
  case FmtIssueKind::PositionalOutOfRange:
    return "positional argument out of range";
  case FmtIssueKind::RangeStartOutOfRange:
    return "positional range starts out of range";
  case FmtIssueKind::InvalidPositionalIndex:
    return "invalid positional index";
  }
  llvm_unreachable("unknown formatting issue kind");
}

void FmtError::log(raw_ostream &os) const {
  llvm::interleave(
      getFmtDiagnostics(*this), os,
      [&](const FmtDiagnostic &message) { os << message.message; }, "\n");
}

SmallVector<FmtDiagnostic>
mlir::tblgen::getFmtDiagnostics(const FmtError &error) {
  SmallVector<FmtDiagnostic> messages;
  auto add = [&](bool isError, ArrayRef<llvm::SMLoc> locations,
                 const Twine &message) {
    messages.push_back(
        {isError, {locations.begin(), locations.end()}, message.str()});
  };

  const FmtSite *anchor = nullptr;
  for (const FmtOrigin &origin : error.getOrigins()) {
    const FmtSite &site = origin.getDefinition();
    if (!site.locations.empty() && site.locations.front().isValid()) {
      anchor = &site;
      break;
    }
  }
  ArrayRef<llvm::SMLoc> locations =
      anchor ? ArrayRef<llvm::SMLoc>(anchor->locations)
             : ArrayRef<llvm::SMLoc>{};
  StringRef purpose = "template";
  if (!error.getOrigins().empty())
    purpose = error.getOrigins().front().getDefinition().description;
  const FmtIssue &first = error.getIssues().front();
  add(true, locations,
      "cannot substitute '" + first.placeholder + "' while formatting " +
          purpose + ": " + first.getDescription());
  for (const FmtIssue &issue : error.getIssues().drop_front())
    add(false, {},
        "cannot substitute '" + issue.placeholder +
            "' at template byte offset " + Twine(issue.offset) + ": " +
            issue.getDescription());

  std::string escaped;
  llvm::raw_string_ostream(escaped).write_escaped(error.getTemplate(), true);
  add(false, {}, "template: " + escaped);
  add(false, {},
      "failed occurrence starts at template byte offset " +
          Twine(first.offset));
  for (const auto &[index, origin] : llvm::enumerate(error.getOrigins())) {
    const FmtSite &definition = origin.getDefinition();
    if (index != 0)
      add(false, definition.locations,
          "while formatting " + definition.description);
    for (const FmtSite &use : origin.getUses())
      add(false, use.locations, "while expanding " + use.description);
  }

  std::string available;
  llvm::raw_string_ostream availableStream(available);
  availableStream << "available named substitutions: ";
  if (error.getBindings().empty())
    availableStream << "(none)";
  else
    llvm::interleaveComma(error.getBindings(), availableStream,
                          [&](const FmtBinding &binding) {
                            availableStream << '$' << binding.name;
                          });
  add(false, {}, available);
  size_t count = error.getNumPositionalArgs();
  if (!count)
    add(false, {}, "no positional substitutions were supplied");
  else
    add(false, {},
        "positional substitutions supplied: " + Twine(count) +
            " (available indices: $0 through $" + Twine(count - 1) + ")");
  return messages;
}

llvm::Error mlir::tblgen::addFmtOrigin(llvm::Error error,
                                       const FmtOrigin &origin) {
  return llvm::handleErrors(
      std::move(error), [&](std::unique_ptr<FmtError> payload) -> llvm::Error {
        payload->appendOrigin(origin);
        return llvm::Error(std::move(payload));
      });
}

FmtContext::FmtContext(ArrayRef<std::pair<StringRef, StringRef>> subs) {
  for (auto &sub : subs)
    addSubst(sub.first, sub.second);
}

FmtContext &FmtContext::addSubst(StringRef placeholder, const Twine &subst) {
  substMap[placeholder] = subst.str();
  return *this;
}

FmtContext &FmtContext::withBuilder(Twine subst) {
  return addSubst("_builder", subst);
}

FmtContext &FmtContext::withSelf(Twine subst) {
  return addSubst("_self", subst);
}

// GCC also diagnoses deprecated types inside their out-of-line definitions.
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH
std::optional<StringRef>
FmtContext::getSubstFor(FmtContext::PHKind placeholder) const {
  switch (placeholder) {
  case PHKind::Builder:
    return getSubstFor("_builder");
  case PHKind::Self:
    return getSubstFor("_self");
  case PHKind::None:
  case PHKind::Custom:
    return {};
  }
  llvm_unreachable("unknown placeholder kind");
}

LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP

std::optional<StringRef> FmtContext::getSubstFor(StringRef placeholder) const {
  auto it = substMap.find(placeholder);
  if (it == substMap.end())
    return {};
  return StringRef(it->second);
}

SmallVector<FmtBinding> FmtContext::getBindings() const {
  SmallVector<FmtBinding> bindings;
  bindings.reserve(substMap.size());
  for (const auto &entry : substMap)
    bindings.push_back({entry.getKey().str(), entry.getValue()});
  llvm::sort(bindings, [](const FmtBinding &lhs, const FmtBinding &rhs) {
    return lhs.name < rhs.name;
  });
  return bindings;
}

void FmtContext::printBindings(raw_ostream &os) const {
  ::printBindings(os, getBindings());
}

LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH
FmtContext::PHKind FmtContext::getPlaceHolderKind(StringRef str) {
  return StringSwitch<FmtContext::PHKind>(str)
      .Case("_builder", FmtContext::PHKind::Builder)
      .Case("_self", FmtContext::PHKind::Self)
      .Case("", FmtContext::PHKind::None)
      .Default(FmtContext::PHKind::Custom);
}

std::pair<FmtReplacement, StringRef>
FmtObjectBase::splitFmtSegment(StringRef fmt) {
  size_t begin = fmt.find_first_of('$');
  if (begin == StringRef::npos) {
    // No placeholders: the whole format string should be returned as a
    // literal string.
    return {FmtReplacement{fmt}, StringRef()};
  }
  if (begin != 0) {
    // The first placeholder is not at the beginning: we can split the format
    // string into a literal string and the rest.
    return {FmtReplacement{fmt.substr(0, begin)}, fmt.substr(begin)};
  }

  // The first placeholder is at the beginning

  if (fmt.size() == 1) {
    // The whole format string just contains '$': treat as literal.
    return {FmtReplacement{fmt}, StringRef()};
  }

  // Allow escaping dollar with '$$'
  if (fmt[1] == '$') {
    return {FmtReplacement{fmt.substr(0, 1)}, fmt.substr(2)};
  }

  // First try to see if it's a positional placeholder, and then handle special
  // placeholders.

  size_t end = fmt.find_if_not(llvm::isDigit, /*From=*/1);
  if (end != 1) {
    // We have a positional placeholder. Parse the index.
    size_t index = 0;
    if (end == StringRef::npos)
      end = fmt.size();
    bool invalid = fmt.slice(1, end).getAsInteger(10, index);
    bool isRange = fmt.substr(end).starts_with("...");
    size_t tokenEnd = end + (isRange ? 3 : 0);
    if (invalid) {
      FmtReplacement replacement(fmt.take_front(tokenEnd));
      replacement.type = FmtReplacement::Type::InvalidPH;
      return {replacement, fmt.drop_front(tokenEnd)};
    }

    // Check if this is the part of a range specification.
    if (isRange) {
      // Currently only ranges without upper bound are supported.
      return {
          FmtReplacement{fmt.substr(0, end + 3), index, FmtReplacement::kUnset},
          fmt.substr(end + 3)};
    }

    if (end == fmt.size()) {
      // All the remaining characters are part of the positional placeholder.
      return {FmtReplacement{fmt, index}, StringRef()};
    }
    return {FmtReplacement{fmt.substr(0, end), index}, fmt.substr(end)};
  }

  end = fmt.find_if_not([](char c) { return llvm::isAlnum(c) || c == '_'; }, 1);
  auto placeholder = FmtContext::getPlaceHolderKind(fmt.substr(1, end - 1));
  if (end == StringRef::npos) {
    // All the remaining characters are part of the special placeholder.
    return {FmtReplacement{fmt, placeholder}, StringRef()};
  }
  return {FmtReplacement{fmt.substr(0, end), placeholder}, fmt.substr(end)};
}

std::vector<FmtReplacement> FmtObjectBase::parseFormatString(StringRef fmt) {
  std::vector<FmtReplacement> replacements;
  FmtReplacement repl;
  while (!fmt.empty()) {
    std::tie(repl, fmt) = splitFmtSegment(fmt);
    if (repl.type != FmtReplacement::Type::Empty)
      replacements.push_back(repl);
  }
  return replacements;
}

void FmtObjectBase::format(raw_ostream &s) const {
  for (auto &repl : replacements) {
    if (repl.type == FmtReplacement::Type::Empty)
      continue;

    if (repl.type == FmtReplacement::Type::Literal) {
      s << repl.spec;
      continue;
    }

    if (repl.type == FmtReplacement::Type::InvalidPH) {
      s << repl.spec << kMarkerForNoSubst;
      continue;
    }

    if (repl.type == FmtReplacement::Type::SpecialPH) {
      if (repl.placeholder == FmtContext::PHKind::None) {
        s << repl.spec;
      } else if (!context) {
        // We need the context to replace special placeholders.
        s << repl.spec << kMarkerForNoSubst;
      } else {
        std::optional<StringRef> subst;
        // Builtin and custom placeholders use the same lookup path.
        subst = context->getSubstFor(repl.spec.drop_front());
        if (subst)
          s << *subst;
        else
          s << repl.spec << kMarkerForNoSubst;
      }
      continue;
    }

    if (repl.type == FmtReplacement::Type::PositionalRangePH) {
      if (repl.index >= adapters.size()) {
        s << repl.spec << kMarkerForNoSubst;
        continue;
      }
      auto range = llvm::ArrayRef(adapters);
      range = range.drop_front(repl.index);
      if (repl.end != FmtReplacement::kUnset)
        range = range.drop_back(adapters.size() - repl.end);
      llvm::interleaveComma(range, s, [&](auto &x) { x(s, /*Options=*/""); });
      continue;
    }

    assert(repl.type == FmtReplacement::Type::PositionalPH);

    if (repl.index >= adapters.size()) {
      s << repl.spec << kMarkerForNoSubst;
      continue;
    }
    adapters[repl.index](s, /*Options=*/"");
  }
}

llvm::Expected<std::string> FmtObjectBase::strChecked() const {
  SmallVector<FmtIssue> issues;
  for (const FmtReplacement &repl : replacements) {
    std::optional<FmtIssueKind> kind;
    switch (repl.type) {
    case FmtReplacement::Type::Empty:
    case FmtReplacement::Type::Literal:
      break;
    case FmtReplacement::Type::InvalidPH:
      kind = FmtIssueKind::InvalidPositionalIndex;
      break;
    case FmtReplacement::Type::SpecialPH:
      if (repl.placeholder != FmtContext::PHKind::None &&
          (!context || !context->getSubstFor(repl.spec.drop_front())))
        kind = FmtIssueKind::MissingNamedBinding;
      break;
    case FmtReplacement::Type::PositionalPH:
      if (repl.index >= adapters.size())
        kind = FmtIssueKind::PositionalOutOfRange;
      break;
    case FmtReplacement::Type::PositionalRangePH:
      if (repl.index >= adapters.size())
        kind = FmtIssueKind::RangeStartOutOfRange;
      break;
    }
    if (kind)
      issues.push_back({*kind, repl.spec.str(),
                        static_cast<size_t>(repl.spec.data() - fmt.data())});
  }
  if (!issues.empty())
    return llvm::make_error<FmtError>(fmt, std::move(issues),
                                      context ? context->getBindings()
                                              : SmallVector<FmtBinding>{},
                                      adapters.size());
  return str();
}

FmtStrVecObject::FmtStrVecObject(StringRef fmt, const FmtContext *ctx,
                                 ArrayRef<std::string> params)
    : FmtObjectBase(fmt, ctx, params.size()) {
  parameters.reserve(params.size());
  for (std::string p : params)
    parameters.emplace_back(std::move(p));

  adapters.reserve(parameters.size());
  for (auto &p : parameters)
    adapters.push_back(p);
}

FmtStrVecObject::FmtStrVecObject(FmtStrVecObject &&that)
    : FmtObjectBase(std::move(that)), parameters(std::move(that.parameters)) {
  adapters.reserve(parameters.size());
  for (auto &p : parameters)
    adapters.push_back(p);
}

LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
