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
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/Symbolize/MarkupFilter.h"

#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::symbolize;

MarkupFilter::MarkupFilter(raw_ostream &OS, Optional<bool> ColorsEnabled)
    : OS(OS), ColorsEnabled(ColorsEnabled.getValueOr(
                  WithColor::defaultAutoDetectFunction()(OS))) {}

void MarkupFilter::beginLine(StringRef Line) {
  this->Line = Line;
  resetColor();
}

void MarkupFilter::filter(const MarkupNode &Node) {
  if (!checkTag(Node))
    return;

  if (trySGR(Node))
    return;

  if (Node.Tag == "symbol") {
    if (!checkNumFields(Node, 1))
      return;
    highlight();
    OS << llvm::demangle(Node.Fields.front().str());
    restoreColor();
    return;
  }

  OS << Node.Text;
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

bool MarkupFilter::checkTag(const MarkupNode &Node) const {
  if (any_of(Node.Tag, [](char C) { return C < 'a' || C > 'z'; })) {
    WithColor::error(errs()) << "tags must be all lowercase characters\n";
    reportLocation(Node.Tag.begin());
    return false;
  }
  return true;
}

bool MarkupFilter::checkNumFields(const MarkupNode &Node, size_t Size) const {
  if (Node.Fields.size() != Size) {
    WithColor::error(errs()) << "expected " << Size << " fields; found "
                             << Node.Fields.size() << "\n";
    reportLocation(Node.Tag.end());
    return false;
  }
  return true;
}

void MarkupFilter::reportLocation(StringRef::iterator Loc) const {
  errs() << Line;
  WithColor(errs().indent(Loc - Line.begin()), HighlightColor::String) << '^';
  errs() << '\n';
}
