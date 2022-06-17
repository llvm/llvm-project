//===- lib/DebugInfo/Symbolize/Markup.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the log symbolizer markup data model and parser.
///
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/Symbolize/Markup.h"

#include "llvm/ADT/StringExtras.h"

namespace llvm {
namespace symbolize {

// Matches the following:
//   "\033[0m"
//   "\033[1m"
//   "\033[30m" -- "\033[37m"
static const char SGRSyntaxStr[] = "\033\\[([0-1]|3[0-7])m";

MarkupParser::MarkupParser() : SGRSyntax(SGRSyntaxStr) {}

static StringRef takeTo(StringRef Str, StringRef::iterator Pos) {
  return Str.take_front(Pos - Str.begin());
}
static void advanceTo(StringRef &Str, StringRef::iterator Pos) {
  Str = Str.drop_front(Pos - Str.begin());
}

void MarkupParser::parseLine(StringRef Line) {
  Buffer.clear();
  while (!Line.empty()) {
    // Find the first valid markup element, if any.
    if (Optional<MarkupNode> Element = parseElement(Line)) {
      parseTextOutsideMarkup(takeTo(Line, Element->Text.begin()));
      Buffer.push_back(std::move(*Element));
      advanceTo(Line, Element->Text.end());
    } else {
      // The line doesn't contain any more markup elements, so emit it as text.
      parseTextOutsideMarkup(Line);
      return;
    }
  }
}

// Finds and returns the next valid markup element in the given line. Returns
// None if the line contains no valid elements.
Optional<MarkupNode> MarkupParser::parseElement(StringRef Line) {
  while (true) {
    // Find next element using begin and end markers.
    size_t BeginPos = Line.find("{{{");
    if (BeginPos == StringRef::npos)
      return None;
    size_t EndPos = Line.find("}}}", BeginPos + 3);
    if (EndPos == StringRef::npos)
      return None;
    EndPos += 3;
    MarkupNode Element;
    Element.Text = Line.slice(BeginPos, EndPos);
    Line = Line.substr(EndPos);

    // Parse tag.
    StringRef Content = Element.Text.drop_front(3).drop_back(3);
    StringRef FieldsContent;
    std::tie(Element.Tag, FieldsContent) = Content.split(':');
    if (Element.Tag.empty())
      continue;

    // Parse fields.
    if (!FieldsContent.empty())
      FieldsContent.split(Element.Fields, ":");
    else if (Content.back() == ':')
      Element.Fields.push_back(FieldsContent);

    return Element;
  }
}

static MarkupNode textNode(StringRef Text) {
  MarkupNode Node;
  Node.Text = Text;
  return Node;
}

// Parses a region of text known to be outside any markup elements. Such text
// may still contain SGR control codes, so the region is further subdivided into
// control codes and true text regions.
void MarkupParser::parseTextOutsideMarkup(StringRef Text) {
  if (Text.empty())
    return;
  SmallVector<StringRef> Matches;
  while (SGRSyntax.match(Text, &Matches)) {
    // Emit any text before the SGR element.
    if (Matches.begin()->begin() != Text.begin())
      Buffer.push_back(textNode(takeTo(Text, Matches.begin()->begin())));

    Buffer.push_back(textNode(*Matches.begin()));
    advanceTo(Text, Matches.begin()->end());
  }
  if (!Text.empty())
    Buffer.push_back(textNode(Text));
}

} // end namespace symbolize
} // end namespace llvm
