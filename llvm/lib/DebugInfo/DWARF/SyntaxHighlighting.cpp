//===- SyntaxHighlighting.cpp ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SyntaxHighlighting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace dwarf;
using namespace syntax;

static cl::opt<cl::boolOrDefault>
    UseColor("color",
             cl::desc("use colored syntax highlighting (default=autodetect)"),
             cl::init(cl::BOU_UNSET));

WithColor::WithColor(raw_ostream &OS, enum HighlightColor Type) : OS(OS) {
  // Detect color from terminal type unless the user passed the --color option.
  if (UseColor == cl::BOU_UNSET ? OS.has_colors() : UseColor == cl::BOU_TRUE) {
    switch (Type) {
    case Address:    OS.changeColor(raw_ostream::YELLOW);  break;
    case String:     OS.changeColor(raw_ostream::GREEN);   break;
    case Tag:        OS.changeColor(raw_ostream::BLUE);    break;
    case Attribute:  OS.changeColor(raw_ostream::CYAN);    break;
    case Enumerator: OS.changeColor(raw_ostream::MAGENTA); break;
    case Macro:      OS.changeColor(raw_ostream::RED);     break;
    }
  }
}

WithColor::~WithColor() {
  if (UseColor == cl::BOU_UNSET ? OS.has_colors() : UseColor == cl::BOU_TRUE)
    OS.resetColor();
}
