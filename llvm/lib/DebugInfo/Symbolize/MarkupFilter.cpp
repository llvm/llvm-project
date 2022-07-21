//===-- lib/DebugInfo/Symbolize/MarkupFilter.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the implementation of a filter that replaces symbolizer
/// markup with human-readable expressions.
///
/// See https://llvm.org/docs/SymbolizerMarkupFormat.html
///
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/Symbolize/MarkupFilter.h"

#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/DebugInfo/Symbolize/Markup.h"
#include "llvm/Debuginfod/Debuginfod.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::symbolize;

MarkupFilter::MarkupFilter(raw_ostream &OS, Optional<bool> ColorsEnabled)
    : OS(OS), ColorsEnabled(ColorsEnabled.value_or(
                  WithColor::defaultAutoDetectFunction()(OS))) {}

void MarkupFilter::filter(StringRef Line) {
  this->Line = Line;
  resetColor();

  Parser.parseLine(Line);
  SmallVector<MarkupNode> DeferredNodes;
  // See if the line is a contextual (i.e. contains a contextual element).
  // In this case, anything after the contextual element is elided, or the whole
  // line may be elided.
  while (Optional<MarkupNode> Node = Parser.nextNode()) {
    // If this was a contextual line, then summarily stop processing.
    if (tryContextualElement(*Node, DeferredNodes))
      return;
    // This node may yet be part of an elided contextual line.
    DeferredNodes.push_back(*Node);
  }

  // This was not a contextual line, so nothing in it should be elided.
  endAnyModuleInfoLine();
  for (const MarkupNode &Node : DeferredNodes)
    filterNode(Node);
}

void MarkupFilter::finish() {
  Parser.flush();
  while (Optional<MarkupNode> Node = Parser.nextNode())
    filterNode(*Node);
  endAnyModuleInfoLine();
  resetColor();
  Modules.clear();
  MMaps.clear();
}

// See if the given node is a contextual element and handle it if so. This may
// either output or defer the element; in the former case, it will first emit
// any DeferredNodes.
//
// Returns true if the given element was a contextual element. In this case,
// DeferredNodes should be considered handled and should not be emitted. The
// rest of the containing line must also be ignored in case the element was
// deferred to a following line.
bool MarkupFilter::tryContextualElement(
    const MarkupNode &Node, const SmallVector<MarkupNode> &DeferredNodes) {
  if (tryMMap(Node, DeferredNodes))
    return true;
  if (tryReset(Node, DeferredNodes))
    return true;
  return tryModule(Node, DeferredNodes);
}

bool MarkupFilter::tryMMap(const MarkupNode &Node,
                           const SmallVector<MarkupNode> &DeferredNodes) {
  if (Node.Tag != "mmap")
    return false;
  Optional<MMap> ParsedMMap = parseMMap(Node);
  if (!ParsedMMap)
    return true;

  if (const MMap *M = overlappingMMap(*ParsedMMap)) {
    WithColor::error(errs())
        << formatv("overlapping mmap: #{0:x} [{1:x},{2:x})\n", M->Mod->ID,
                   M->Addr, M->Addr + M->Size);
    reportLocation(Node.Fields[0].begin());
    return true;
  }

  auto Res = MMaps.emplace(ParsedMMap->Addr, std::move(*ParsedMMap));
  assert(Res.second && "Overlap check should ensure emplace succeeds.");
  MMap &MMap = Res.first->second;

  if (!MIL || MIL->Mod != MMap.Mod) {
    endAnyModuleInfoLine();
    for (const MarkupNode &Node : DeferredNodes)
      filterNode(Node);
    beginModuleInfoLine(MMap.Mod);
    OS << "; adds";
  }
  MIL->MMaps.push_back(&MMap);
  return true;
}

bool MarkupFilter::tryReset(const MarkupNode &Node,
                            const SmallVector<MarkupNode> &DeferredNodes) {
  if (Node.Tag != "reset")
    return false;
  if (!checkNumFields(Node, 0))
    return true;

  if (!Modules.empty() || !MMaps.empty()) {
    Modules.clear();
    MMaps.clear();

    endAnyModuleInfoLine();
    for (const MarkupNode &Node : DeferredNodes)
      filterNode(Node);
    highlight();
    OS << "[[[reset]]]" << lineEnding();
    restoreColor();
  }
  return true;
}

bool MarkupFilter::tryModule(const MarkupNode &Node,
                             const SmallVector<MarkupNode> &DeferredNodes) {
  if (Node.Tag != "module")
    return false;
  Optional<Module> ParsedModule = parseModule(Node);
  if (!ParsedModule)
    return true;

  auto Res = Modules.try_emplace(
      ParsedModule->ID, std::make_unique<Module>(std::move(*ParsedModule)));
  if (!Res.second) {
    WithColor::error(errs()) << "duplicate module ID\n";
    reportLocation(Node.Fields[0].begin());
    return true;
  }
  Module &Module = *Res.first->second;

  endAnyModuleInfoLine();
  for (const MarkupNode &Node : DeferredNodes)
    filterNode(Node);
  beginModuleInfoLine(&Module);
  OS << "; BuildID=";
  highlightValue();
  OS << toHex(Module.BuildID, /*LowerCase=*/true);
  highlight();
  return true;
}

void MarkupFilter::beginModuleInfoLine(const Module *M) {
  highlight();
  OS << "[[[ELF module";
  highlightValue();
  OS << formatv(" #{0:x} \"{1}\"", M->ID, M->Name);
  highlight();
  MIL = ModuleInfoLine{M};
}

void MarkupFilter::endAnyModuleInfoLine() {
  if (!MIL)
    return;
  llvm::stable_sort(MIL->MMaps, [](const MMap *A, const MMap *B) {
    return A->Addr < B->Addr;
  });
  for (const MMap *M : MIL->MMaps) {
    OS << (M == MIL->MMaps.front() ? ' ' : '-');
    highlightValue();
    OS << formatv("{0:x}", M->Addr);
    highlight();
    OS << '(';
    highlightValue();
    OS << M->Mode;
    highlight();
    OS << ')';
  }
  OS << "]]]" << lineEnding();
  restoreColor();
  MIL.reset();
}

// Handle a node that is known not to be a contextual element.
void MarkupFilter::filterNode(const MarkupNode &Node) {
  if (!checkTag(Node))
    return;
  if (tryPresentation(Node))
    return;
  if (trySGR(Node))
    return;

  OS << Node.Text;
}

bool MarkupFilter::tryPresentation(const MarkupNode &Node) {
  return trySymbol(Node);
}

bool MarkupFilter::trySymbol(const MarkupNode &Node) {
  if (Node.Tag != "symbol")
    return false;
  if (!checkNumFields(Node, 1))
    return true;

  highlight();
  OS << llvm::demangle(Node.Fields.front().str());
  restoreColor();
  return true;
}

bool MarkupFilter::trySGR(const MarkupNode &Node) {
  if (Node.Text == "\033[0m") {
    resetColor();
    return true;
  }
  if (Node.Text == "\033[1m") {
    Bold = true;
    if (ColorsEnabled)
      OS.changeColor(raw_ostream::Colors::SAVEDCOLOR, Bold);
    return true;
  }
  auto SGRColor = StringSwitch<Optional<raw_ostream::Colors>>(Node.Text)
                      .Case("\033[30m", raw_ostream::Colors::BLACK)
                      .Case("\033[31m", raw_ostream::Colors::RED)
                      .Case("\033[32m", raw_ostream::Colors::GREEN)
                      .Case("\033[33m", raw_ostream::Colors::YELLOW)
                      .Case("\033[34m", raw_ostream::Colors::BLUE)
                      .Case("\033[35m", raw_ostream::Colors::MAGENTA)
                      .Case("\033[36m", raw_ostream::Colors::CYAN)
                      .Case("\033[37m", raw_ostream::Colors::WHITE)
                      .Default(llvm::None);
  if (SGRColor) {
    Color = *SGRColor;
    if (ColorsEnabled)
      OS.changeColor(*Color);
    return true;
  }

  return false;
}

// Begin highlighting text by picking a different color than the current color
// state.
void MarkupFilter::highlight() {
  if (!ColorsEnabled)
    return;
  OS.changeColor(Color == raw_ostream::Colors::BLUE ? raw_ostream::Colors::CYAN
                                                    : raw_ostream::Colors::BLUE,
                 Bold);
}

// Begin highlighting a field within a highlighted markup string.
void MarkupFilter::highlightValue() {
  if (!ColorsEnabled)
    return;
  OS.changeColor(raw_ostream::Colors::GREEN, Bold);
}

// Set the output stream's color to the current color and bold state of the SGR
// abstract machine.
void MarkupFilter::restoreColor() {
  if (!ColorsEnabled)
    return;
  if (Color) {
    OS.changeColor(*Color, Bold);
  } else {
    OS.resetColor();
    if (Bold)
      OS.changeColor(raw_ostream::Colors::SAVEDCOLOR, Bold);
  }
}

// Set the SGR and output stream's color and bold states back to the default.
void MarkupFilter::resetColor() {
  if (!Color && !Bold)
    return;
  Color.reset();
  Bold = false;
  if (ColorsEnabled)
    OS.resetColor();
}

// This macro helps reduce the amount of indirection done through Optional
// below, since the usual case upon returning a None Optional is to return None.
#define ASSIGN_OR_RETURN_NONE(TYPE, NAME, EXPR)                                \
  auto NAME##Opt = (EXPR);                                                     \
  if (!NAME##Opt)                                                              \
    return None;                                                               \
  TYPE NAME = std::move(*NAME##Opt)

Optional<MarkupFilter::Module>
MarkupFilter::parseModule(const MarkupNode &Element) const {
  if (!checkNumFieldsAtLeast(Element, 3))
    return None;
  ASSIGN_OR_RETURN_NONE(uint64_t, ID, parseModuleID(Element.Fields[0]));
  StringRef Name = Element.Fields[1];
  StringRef Type = Element.Fields[2];
  if (Type != "elf") {
    WithColor::error() << "unknown module type\n";
    reportLocation(Type.begin());
    return None;
  }
  if (!checkNumFields(Element, 4))
    return None;
  ASSIGN_OR_RETURN_NONE(SmallVector<uint8_t>, BuildID,
                        parseBuildID(Element.Fields[3]));
  return Module{ID, Name.str(), std::move(BuildID)};
}

Optional<MarkupFilter::MMap>
MarkupFilter::parseMMap(const MarkupNode &Element) const {
  if (!checkNumFieldsAtLeast(Element, 3))
    return None;
  ASSIGN_OR_RETURN_NONE(uint64_t, Addr, parseAddr(Element.Fields[0]));
  ASSIGN_OR_RETURN_NONE(uint64_t, Size, parseSize(Element.Fields[1]));
  StringRef Type = Element.Fields[2];
  if (Type != "load") {
    WithColor::error() << "unknown mmap type\n";
    reportLocation(Type.begin());
    return None;
  }
  if (!checkNumFields(Element, 6))
    return None;
  ASSIGN_OR_RETURN_NONE(uint64_t, ID, parseModuleID(Element.Fields[3]));
  ASSIGN_OR_RETURN_NONE(std::string, Mode, parseMode(Element.Fields[4]));
  auto It = Modules.find(ID);
  if (It == Modules.end()) {
    WithColor::error() << "unknown module ID\n";
    reportLocation(Element.Fields[3].begin());
    return None;
  }
  ASSIGN_OR_RETURN_NONE(uint64_t, ModuleRelativeAddr,
                        parseAddr(Element.Fields[5]));
  return MMap{Addr, Size, It->second.get(), std::move(Mode),
              ModuleRelativeAddr};
}

// Parse an address (%p in the spec).
Optional<uint64_t> MarkupFilter::parseAddr(StringRef Str) const {
  if (Str.empty()) {
    reportTypeError(Str, "address");
    return None;
  }
  if (all_of(Str, [](char C) { return C == '0'; }))
    return 0;
  if (!Str.startswith("0x")) {
    reportTypeError(Str, "address");
    return None;
  }
  uint64_t Addr;
  if (Str.drop_front(2).getAsInteger(16, Addr)) {
    reportTypeError(Str, "address");
    return None;
  }
  return Addr;
}

// Parse a module ID (%i in the spec).
Optional<uint64_t> MarkupFilter::parseModuleID(StringRef Str) const {
  uint64_t ID;
  if (Str.getAsInteger(0, ID)) {
    reportTypeError(Str, "module ID");
    return None;
  }
  return ID;
}

// Parse a size (%i in the spec).
Optional<uint64_t> MarkupFilter::parseSize(StringRef Str) const {
  uint64_t ID;
  if (Str.getAsInteger(0, ID)) {
    reportTypeError(Str, "size");
    return None;
  }
  return ID;
}

// Parse a build ID (%x in the spec).
Optional<SmallVector<uint8_t>> MarkupFilter::parseBuildID(StringRef Str) const {
  std::string Bytes;
  if (Str.empty() || Str.size() % 2 || !tryGetFromHex(Str, Bytes)) {
    reportTypeError(Str, "build ID");
    return None;
  }
  ArrayRef<uint8_t> BuildID(reinterpret_cast<const uint8_t *>(Bytes.data()),
                            Bytes.size());
  return SmallVector<uint8_t>(BuildID.begin(), BuildID.end());
}

// Parses the mode string for an mmap element.
Optional<std::string> MarkupFilter::parseMode(StringRef Str) const {
  if (Str.empty()) {
    reportTypeError(Str, "mode");
    return None;
  }

  // Pop off each of r/R, w/W, and x/X from the front, in that order.
  StringRef Remainder = Str;
  if (!Remainder.empty() && tolower(Remainder.front()) == 'r')
    Remainder = Remainder.drop_front();
  if (!Remainder.empty() && tolower(Remainder.front()) == 'w')
    Remainder = Remainder.drop_front();
  if (!Remainder.empty() && tolower(Remainder.front()) == 'x')
    Remainder = Remainder.drop_front();

  // If anything remains, then the string wasn't a mode.
  if (!Remainder.empty()) {
    reportTypeError(Str, "mode");
    return None;
  }

  // Normalize the mode.
  return Str.lower();
}

bool MarkupFilter::checkTag(const MarkupNode &Node) const {
  if (any_of(Node.Tag, [](char C) { return C < 'a' || C > 'z'; })) {
    WithColor::error(errs()) << "tags must be all lowercase characters\n";
    reportLocation(Node.Tag.begin());
    return false;
  }
  return true;
}

bool MarkupFilter::checkNumFields(const MarkupNode &Element,
                                  size_t Size) const {
  if (Element.Fields.size() != Size) {
    WithColor::error(errs()) << "expected " << Size << " fields; found "
                             << Element.Fields.size() << "\n";
    reportLocation(Element.Tag.end());
    return false;
  }
  return true;
}

bool MarkupFilter::checkNumFieldsAtLeast(const MarkupNode &Element,
                                         size_t Size) const {
  if (Element.Fields.size() < Size) {
    WithColor::error(errs())
        << "expected at least " << Size << " fields; found "
        << Element.Fields.size() << "\n";
    reportLocation(Element.Tag.end());
    return false;
  }
  return true;
}

void MarkupFilter::reportTypeError(StringRef Str, StringRef TypeName) const {
  WithColor::error(errs()) << "expected " << TypeName << "; found '" << Str
                           << "'\n";
  reportLocation(Str.begin());
}

// Prints two lines that point out the given location in the current Line using
// a caret. The iterator must be within the bounds of the most recent line
// passed to beginLine().
void MarkupFilter::reportLocation(StringRef::iterator Loc) const {
  errs() << Line;
  WithColor(errs().indent(Loc - Line.begin()), HighlightColor::String) << '^';
  errs() << '\n';
}

// Checks for an existing mmap that overlaps the given one and returns a
// pointer to one of them.
const MarkupFilter::MMap *MarkupFilter::overlappingMMap(const MMap &Map) const {
  // If the given map contains the start of another mmap, they overlap.
  auto I = MMaps.upper_bound(Map.Addr);
  if (I != MMaps.end() && Map.contains(I->second.Addr))
    return &I->second;

  // If no element starts inside the given mmap, the only possible overlap would
  // be if the preceding mmap contains the start point of the given mmap.
  if (I != MMaps.begin()) {
    --I;
    if (I->second.contains(Map.Addr))
      return &I->second;
  }
  return nullptr;
}

StringRef MarkupFilter::lineEnding() const {
  return Line.endswith("\r\n") ? "\r\n" : "\n";
}

bool MarkupFilter::MMap::contains(uint64_t Addr) const {
  return this->Addr <= Addr && Addr < this->Addr + Size;
}
