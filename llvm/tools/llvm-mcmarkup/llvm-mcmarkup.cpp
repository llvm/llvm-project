//===-- llvm-mcmarkup.cpp - Parse the MC assembly markup tags -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Example simple parser implementation for the MC assembly markup language.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

static cl::list<std::string>
       InputFilenames(cl::Positional, cl::desc("<input files>"),
                      cl::ZeroOrMore);
static cl::opt<bool>
DumpTags("dump-tags", cl::desc("List all tags encountered in input"));

static StringRef ToolName;

/// Trivial lexer for the markup parser. Input is always handled a character
/// at a time. The lexer just encapsulates EOF and lookahead handling.
class MarkupLexer {
  StringRef::const_iterator Start;
  StringRef::const_iterator CurPtr;
  StringRef::const_iterator End;
public:
  MarkupLexer(StringRef Source)
    : Start(Source.begin()), CurPtr(Source.begin()), End(Source.end()) {}
  // When processing non-markup, input is consumed a character at a time.
  bool isEOF() { return CurPtr == End; }
  int getNextChar() {
    if (CurPtr == End) return EOF;
    return *CurPtr++;
  }
  int peekNextChar() {
    if (CurPtr == End) return EOF;
    return *CurPtr;
  }
  StringRef::const_iterator getPosition() const { return CurPtr; }
};

/// A markup tag is a name and a (usually empty) list of modifiers.
class MarkupTag {
  StringRef Name;
  StringRef Modifiers;
  SMLoc StartLoc;
public:
  MarkupTag(StringRef n, StringRef m, SMLoc Loc)
    : Name(n), Modifiers(m), StartLoc(Loc) {}
  StringRef getName() const { return Name; }
  StringRef getModifiers() const { return Modifiers; }
  SMLoc getLoc() const { return StartLoc; }
};

/// A simple parser implementation for creating MarkupTags from input text.
class MarkupParser {
  MarkupLexer &Lex;
  SourceMgr &SM;
public:
  MarkupParser(MarkupLexer &lex, SourceMgr &SrcMgr) : Lex(lex), SM(SrcMgr) {}
  /// Create a MarkupTag from the current position in the MarkupLexer.
  /// The parseTag() method should be called when the lexer has processed
  /// the opening '<' character. Input will be consumed up to and including
  /// the ':' which terminates the tag open.
  MarkupTag parseTag();
  /// Issue a diagnostic and terminate program execution.
  void FatalError(SMLoc Loc, StringRef Msg);
};

void MarkupParser::FatalError(SMLoc Loc, StringRef Msg) {
  SM.PrintMessage(Loc, SourceMgr::DK_Error, Msg);
  exit(1);
}

// Example handler for when a tag is recognized.
static void processStartTag(MarkupTag &Tag) {
  // If we're just printing the tags, do that, otherwise do some simple
  // colorization.
  if (DumpTags) {
    outs() << Tag.getName();
    if (Tag.getModifiers().size())
      outs() << " " << Tag.getModifiers();
    outs() << "\n";
    return;
  }

  if (!outs().has_colors())
    return;
  // Color registers as red and immediates as cyan. Those don't have nested
  // tags, so don't bother keeping a stack of colors to reset to.
  if (Tag.getName() == "reg")
    outs().changeColor(raw_ostream::RED);
  else if (Tag.getName() == "imm")
    outs().changeColor(raw_ostream::CYAN);
}

// Example handler for when the end of a tag is recognized.
static void processEndTag(MarkupTag &Tag) {
  // If we're printing the tags, there's nothing more to do here. Otherwise,
  // set the color back the normal.
  if (DumpTags)
    return;
  if (!outs().has_colors())
    return;
  // Just reset to basic white.
  outs().changeColor(raw_ostream::WHITE, false);
}

MarkupTag MarkupParser::parseTag() {
  // First off, extract the tag into it's own StringRef so we can look at it
  // outside of the context of consuming input.
  StringRef::const_iterator Start = Lex.getPosition();
  SMLoc Loc = SMLoc::getFromPointer(Start - 1);
  while(Lex.getNextChar() != ':') {
    // EOF is an error.
    if (Lex.isEOF())
      FatalError(SMLoc::getFromPointer(Start), "unterminated markup tag");
  }
  StringRef RawTag(Start, Lex.getPosition() - Start - 1);
  std::pair<StringRef, StringRef> SplitTag = RawTag.split(' ');
  return MarkupTag(SplitTag.first, SplitTag.second, Loc);
}

static void parseMCMarkup(StringRef Filename) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferPtr =
      MemoryBuffer::getFileOrSTDIN(Filename);
  if (std::error_code EC = BufferPtr.getError()) {
    errs() << ToolName << ": " << EC.message() << '\n';
    return;
  }
  std::unique_ptr<MemoryBuffer> &Buffer = BufferPtr.get();

  SourceMgr SrcMgr;

  StringRef InputSource = Buffer->getBuffer();

  // Tell SrcMgr about this buffer, which is what the parser will pick up.
  SrcMgr.AddNewSourceBuffer(std::move(Buffer), SMLoc());

  MarkupLexer Lex(InputSource);
  MarkupParser Parser(Lex, SrcMgr);

  SmallVector<MarkupTag, 4> TagStack;

  for (int CurChar = Lex.getNextChar();
       CurChar != EOF;
       CurChar = Lex.getNextChar()) {
    switch (CurChar) {
    case '<': {
      // A "<<" is output as a literal '<' and does not start a markup tag.
      if (Lex.peekNextChar() == '<') {
        (void)Lex.getNextChar();
        break;
      }
      // Parse the markup entry.
      TagStack.push_back(Parser.parseTag());

      // Do any special handling for the start of a tag.
      processStartTag(TagStack.back());
      continue;
    }
    case '>': {
      SMLoc Loc = SMLoc::getFromPointer(Lex.getPosition() - 1);
      // A ">>" is output as a literal '>' and does not end a markup tag.
      if (Lex.peekNextChar() == '>') {
        (void)Lex.getNextChar();
        break;
      }
      // Close out the innermost tag.
      if (TagStack.empty())
        Parser.FatalError(Loc, "'>' without matching '<'");

      // Do any special handling for the end of a tag.
      processEndTag(TagStack.back());

      TagStack.pop_back();
      continue;
    }
    default:
      break;
    }
    // For anything else, just echo the character back out.
    if (!DumpTags && CurChar != EOF)
      outs() << (char)CurChar;
  }

  // If there are any unterminated markup tags, issue diagnostics for them.
  while (!TagStack.empty()) {
    MarkupTag &Tag = TagStack.back();
    SrcMgr.PrintMessage(Tag.getLoc(), SourceMgr::DK_Error,
                        "unterminated markup tag");
    TagStack.pop_back();
  }
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);

  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, "llvm MC markup parser\n");

  ToolName = argv[0];

  // If no input files specified, read from stdin.
  if (InputFilenames.size() == 0)
    InputFilenames.push_back("-");

  llvm::for_each(InputFilenames, parseMCMarkup);
  return 0;
}
