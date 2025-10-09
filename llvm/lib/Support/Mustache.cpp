//===-- Mustache.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/Support/Mustache.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <optional>
#include <sstream>

#define DEBUG_TYPE "mustache"

using namespace llvm;
using namespace llvm::mustache;

namespace {

using Accessor = ArrayRef<StringRef>;

static bool isFalsey(const json::Value &V) {
  return V.getAsNull() || (V.getAsBoolean() && !V.getAsBoolean().value()) ||
         (V.getAsArray() && V.getAsArray()->empty());
}

static bool isContextFalsey(const json::Value *V) {
  // A missing context (represented by a nullptr) is defined as falsey.
  if (!V)
    return true;
  return isFalsey(*V);
}

static Accessor splitMustacheString(StringRef Str, MustacheContext &Ctx) {
  // We split the mustache string into an accessor.
  // For example:
  //    "a.b.c" would be split into {"a", "b", "c"}
  // We make an exception for a single dot which
  // refers to the current context.
  SmallVector<StringRef> Tokens;
  if (Str == ".") {
    // "." is a special accessor that refers to the current context.
    // It's a literal, so it doesn't need to be saved.
    Tokens.push_back(".");
  } else {
    while (!Str.empty()) {
      StringRef Part;
      std::tie(Part, Str) = Str.split('.');
      // Each part of the accessor needs to be saved to the arena
      // to ensure it has a stable address.
      Tokens.push_back(Ctx.Saver.save(Part.trim()));
    }
  }
  // Now, allocate memory for the array of StringRefs in the arena.
  StringRef *ArenaTokens = Ctx.Allocator.Allocate<StringRef>(Tokens.size());
  // Copy the StringRefs from the stack vector to the arena.
  std::copy(Tokens.begin(), Tokens.end(), ArenaTokens);
  // Return an ArrayRef pointing to the stable arena memory.
  return ArrayRef<StringRef>(ArenaTokens, Tokens.size());
}
} // namespace

namespace llvm::mustache {

class MustacheOutputStream : public raw_ostream {
public:
  MustacheOutputStream() = default;
  ~MustacheOutputStream() override = default;

  virtual void suspendIndentation() {}
  virtual void resumeIndentation() {}

private:
  void anchor() override;
};

void MustacheOutputStream::anchor() {}

class RawMustacheOutputStream : public MustacheOutputStream {
public:
  RawMustacheOutputStream(raw_ostream &OS) : OS(OS) { SetUnbuffered(); }

private:
  raw_ostream &OS;

  void write_impl(const char *Ptr, size_t Size) override {
    OS.write(Ptr, Size);
  }
  uint64_t current_pos() const override { return OS.tell(); }
};

class Token {
public:
  enum class Type {
    Text,
    Variable,
    Partial,
    SectionOpen,
    SectionClose,
    InvertSectionOpen,
    UnescapeVariable,
    Comment,
    SetDelimiter,
  };

  Token(StringRef Str)
      : TokenType(Type::Text), RawBody(Str), TokenBody(RawBody),
        AccessorValue({}), Indentation(0) {};

  Token(StringRef RawBody, StringRef TokenBody, char Identifier,
        MustacheContext &Ctx)
      : RawBody(RawBody), TokenBody(TokenBody), Indentation(0) {
    TokenType = getTokenType(Identifier);
    if (TokenType == Type::Comment)
      return;
    StringRef AccessorStr(this->TokenBody);
    if (TokenType != Type::Variable)
      AccessorStr = AccessorStr.substr(1);
    AccessorValue = splitMustacheString(StringRef(AccessorStr).trim(), Ctx);
  }

  ArrayRef<StringRef> getAccessor() const { return AccessorValue; }

  Type getType() const { return TokenType; }

  void setIndentation(size_t NewIndentation) { Indentation = NewIndentation; }

  size_t getIndentation() const { return Indentation; }

  static Type getTokenType(char Identifier) {
    switch (Identifier) {
    case '#':
      return Type::SectionOpen;
    case '/':
      return Type::SectionClose;
    case '^':
      return Type::InvertSectionOpen;
    case '!':
      return Type::Comment;
    case '>':
      return Type::Partial;
    case '&':
      return Type::UnescapeVariable;
    case '=':
      return Type::SetDelimiter;
    default:
      return Type::Variable;
    }
  }

  Type TokenType;
  // RawBody is the original string that was tokenized.
  StringRef RawBody;
  // TokenBody is the original string with the identifier removed.
  StringRef TokenBody;
  ArrayRef<StringRef> AccessorValue;
  size_t Indentation;
};

using EscapeMap = DenseMap<char, std::string>;

class ASTNode : public ilist_node<ASTNode> {
public:
  enum Type {
    Root,
    Text,
    Partial,
    Variable,
    UnescapeVariable,
    Section,
    InvertSection,
  };

  ASTNode(MustacheContext &Ctx)
      : Ctx(Ctx), Ty(Type::Root), Parent(nullptr), ParentContext(nullptr) {}

  ASTNode(MustacheContext &Ctx, StringRef Body, ASTNode *Parent)
      : Ctx(Ctx), Ty(Type::Text), Body(Body), Parent(Parent),
        ParentContext(nullptr) {}

  // Constructor for Section/InvertSection/Variable/UnescapeVariable Nodes
  ASTNode(MustacheContext &Ctx, Type Ty, ArrayRef<StringRef> Accessor,
          ASTNode *Parent)
      : Ctx(Ctx), Ty(Ty), Parent(Parent), AccessorValue(Accessor),
        ParentContext(nullptr) {}

  void addChild(AstPtr Child) { Children.push_back(Child); };

  void setRawBody(StringRef NewBody) { RawBody = NewBody; };

  void setIndentation(size_t NewIndentation) { Indentation = NewIndentation; };

  void render(const llvm::json::Value &Data, MustacheOutputStream &OS);

private:
  void renderLambdas(const llvm::json::Value &Contexts,
                     MustacheOutputStream &OS, Lambda &L);

  void renderSectionLambdas(const llvm::json::Value &Contexts,
                            MustacheOutputStream &OS, SectionLambda &L);

  void renderPartial(const llvm::json::Value &Contexts,
                     MustacheOutputStream &OS, ASTNode *Partial);

  void renderChild(const llvm::json::Value &Context, MustacheOutputStream &OS);

  const llvm::json::Value *findContext();

  void renderRoot(const json::Value &CurrentCtx, MustacheOutputStream &OS);
  void renderText(MustacheOutputStream &OS);
  void renderPartial(const json::Value &CurrentCtx, MustacheOutputStream &OS);
  void renderVariable(const json::Value &CurrentCtx, MustacheOutputStream &OS);
  void renderUnescapeVariable(const json::Value &CurrentCtx,
                              MustacheOutputStream &OS);
  void renderSection(const json::Value &CurrentCtx, MustacheOutputStream &OS);
  void renderInvertSection(const json::Value &CurrentCtx,
                           MustacheOutputStream &OS);

  MustacheContext &Ctx;
  Type Ty;
  size_t Indentation = 0;
  StringRef RawBody;
  StringRef Body;
  ASTNode *Parent;
  ASTNodeList Children;
  const ArrayRef<StringRef> AccessorValue;
  const llvm::json::Value *ParentContext;
};

// A wrapper for arena allocator for ASTNodes
static AstPtr createRootNode(MustacheContext &Ctx) {
  return new (Ctx.Allocator.Allocate<ASTNode>()) ASTNode(Ctx);
}

static AstPtr createNode(MustacheContext &Ctx, ASTNode::Type T,
                         ArrayRef<StringRef> A, ASTNode *Parent) {
  return new (Ctx.Allocator.Allocate<ASTNode>()) ASTNode(Ctx, T, A, Parent);
}

static AstPtr createTextNode(MustacheContext &Ctx, StringRef Body,
                             ASTNode *Parent) {
  return new (Ctx.Allocator.Allocate<ASTNode>()) ASTNode(Ctx, Body, Parent);
}

// Function to check if there is meaningful text behind.
// We determine if a token has meaningful text behind
// if the right of previous token contains anything that is
// not a newline.
// For example:
//  "Stuff {{#Section}}" (returns true)
//   vs
//  "{{#Section}} \n" (returns false)
// We make an exception for when previous token is empty
// and the current token is the second token.
// For example:
//  "{{#Section}}"
static bool hasTextBehind(size_t Idx, const ArrayRef<Token> &Tokens) {
  if (Idx == 0)
    return true;

  size_t PrevIdx = Idx - 1;
  if (Tokens[PrevIdx].getType() != Token::Type::Text)
    return true;

  const Token &PrevToken = Tokens[PrevIdx];
  StringRef TokenBody = StringRef(PrevToken.RawBody).rtrim(" \r\t\v");
  return !TokenBody.ends_with("\n") && !(TokenBody.empty() && Idx == 1);
}

// Function to check if there's no meaningful text ahead.
// We determine if a token has text ahead if the left of previous
// token does not start with a newline.
static bool hasTextAhead(size_t Idx, const ArrayRef<Token> &Tokens) {
  if (Idx >= Tokens.size() - 1)
    return true;

  size_t NextIdx = Idx + 1;
  if (Tokens[NextIdx].getType() != Token::Type::Text)
    return true;

  const Token &NextToken = Tokens[NextIdx];
  StringRef TokenBody = StringRef(NextToken.RawBody).ltrim(" ");
  return !TokenBody.starts_with("\r\n") && !TokenBody.starts_with("\n");
}

static bool requiresCleanUp(Token::Type T) {
  // We must clean up all the tokens that could contain child nodes.
  return T == Token::Type::SectionOpen || T == Token::Type::InvertSectionOpen ||
         T == Token::Type::SectionClose || T == Token::Type::Comment ||
         T == Token::Type::Partial || T == Token::Type::SetDelimiter;
}

// Adjust next token body if there is no text ahead.
// For example:
// The template string
//  "{{! Comment }} \nLine 2"
// would be considered as no text ahead and should be rendered as
//  " Line 2"
static void stripTokenAhead(SmallVectorImpl<Token> &Tokens, size_t Idx) {
  Token &NextToken = Tokens[Idx + 1];
  StringRef NextTokenBody = NextToken.TokenBody;
  // Cut off the leading newline which could be \n or \r\n.
  if (NextTokenBody.starts_with("\r\n"))
    NextToken.TokenBody = NextTokenBody.substr(2);
  else if (NextTokenBody.starts_with("\n"))
    NextToken.TokenBody = NextTokenBody.substr(1);
}

// Adjust previous token body if there no text behind.
// For example:
//  The template string
//  " \t{{#section}}A{{/section}}"
// would be considered as having no text ahead and would be render as:
//  "A"
void stripTokenBefore(SmallVectorImpl<Token> &Tokens, size_t Idx,
                      Token &CurrentToken, Token::Type CurrentType) {
  Token &PrevToken = Tokens[Idx - 1];
  StringRef PrevTokenBody = PrevToken.TokenBody;
  StringRef Unindented = PrevTokenBody.rtrim(" \r\t\v");
  size_t Indentation = PrevTokenBody.size() - Unindented.size();
  PrevToken.TokenBody = Unindented;
  CurrentToken.setIndentation(Indentation);
}

struct Tag {
  enum class Kind {
    None,
    Normal, // {{...}}
    Triple, // {{{...}}}
  };

  Kind TagKind = Kind::None;
  StringRef Content;   // The content between the delimiters.
  StringRef FullMatch; // The entire tag, including delimiters.
  size_t StartPosition = StringRef::npos;
};

[[maybe_unused]] static const char *tagKindToString(Tag::Kind K) {
  switch (K) {
  case Tag::Kind::None:
    return "None";
  case Tag::Kind::Normal:
    return "Normal";
  case Tag::Kind::Triple:
    return "Triple";
  }
  llvm_unreachable("Unknown Tag::Kind");
}

[[maybe_unused]] static const char *jsonKindToString(json::Value::Kind K) {
  switch (K) {
  case json::Value::Kind::Null:
    return "JSON_KIND_NULL";
  case json::Value::Kind::Boolean:
    return "JSON_KIND_BOOLEAN";
  case json::Value::Kind::Number:
    return "JSON_KIND_NUMBER";
  case json::Value::Kind::String:
    return "JSON_KIND_STRING";
  case json::Value::Kind::Array:
    return "JSON_KIND_ARRAY";
  case json::Value::Kind::Object:
    return "JSON_KIND_OBJECT";
  }
  llvm_unreachable("Unknown json::Value::Kind");
}

static Tag findNextTag(StringRef Template, size_t StartPos, StringRef Open,
                       StringRef Close) {
  const StringLiteral TripleOpen("{{{");
  const StringLiteral TripleClose("}}}");

  size_t NormalOpenPos = Template.find(Open, StartPos);
  size_t TripleOpenPos = Template.find(TripleOpen, StartPos);

  Tag Result;

  // Determine which tag comes first.
  if (TripleOpenPos != StringRef::npos &&
      (NormalOpenPos == StringRef::npos || TripleOpenPos <= NormalOpenPos)) {
    // Found a triple mustache tag.
    size_t EndPos =
        Template.find(TripleClose, TripleOpenPos + TripleOpen.size());
    if (EndPos == StringRef::npos)
      return Result; // No closing tag found.

    Result.TagKind = Tag::Kind::Triple;
    Result.StartPosition = TripleOpenPos;
    size_t ContentStart = TripleOpenPos + TripleOpen.size();
    Result.Content = Template.substr(ContentStart, EndPos - ContentStart);
    Result.FullMatch = Template.substr(
        TripleOpenPos, (EndPos + TripleClose.size()) - TripleOpenPos);
  } else if (NormalOpenPos != StringRef::npos) {
    // Found a normal mustache tag.
    size_t EndPos = Template.find(Close, NormalOpenPos + Open.size());
    if (EndPos == StringRef::npos)
      return Result; // No closing tag found.

    Result.TagKind = Tag::Kind::Normal;
    Result.StartPosition = NormalOpenPos;
    size_t ContentStart = NormalOpenPos + Open.size();
    Result.Content = Template.substr(ContentStart, EndPos - ContentStart);
    Result.FullMatch =
        Template.substr(NormalOpenPos, (EndPos + Close.size()) - NormalOpenPos);
  }

  return Result;
}

static std::optional<std::pair<StringRef, StringRef>>
processTag(const Tag &T, SmallVectorImpl<Token> &Tokens, MustacheContext &Ctx) {
  LLVM_DEBUG(dbgs() << "[Tag] " << T.FullMatch << ", Content: " << T.Content
                    << ", Kind: " << tagKindToString(T.TagKind) << "\n");
  if (T.TagKind == Tag::Kind::Triple) {
    Tokens.emplace_back(T.FullMatch, Ctx.Saver.save("&" + T.Content), '&', Ctx);
    return std::nullopt;
  }
  StringRef Interpolated = T.Content;
  if (!Interpolated.trim().starts_with("=")) {
    char Front = Interpolated.empty() ? ' ' : Interpolated.trim().front();
    Tokens.emplace_back(T.FullMatch, Interpolated, Front, Ctx);
    return std::nullopt;
  }
  Tokens.emplace_back(T.FullMatch, Interpolated, '=', Ctx);
  StringRef DelimSpec = Interpolated.trim();
  DelimSpec = DelimSpec.drop_front(1);
  DelimSpec = DelimSpec.take_until([](char C) { return C == '='; });
  DelimSpec = DelimSpec.trim();

  std::pair<StringRef, StringRef> Ret = DelimSpec.split(' ');
  LLVM_DEBUG(dbgs() << "[Set Delimiter] NewOpen: " << Ret.first
                    << ", NewClose: " << Ret.second << "\n");
  return Ret;
}

// Simple tokenizer that splits the template into tokens.
// The mustache spec allows {{{ }}} to unescape variables,
// but we don't support that here. An unescape variable
// is represented only by {{& variable}}.
static SmallVector<Token> tokenize(StringRef Template, MustacheContext &Ctx) {
  LLVM_DEBUG(dbgs() << "[Tokenize Template] \"" << Template << "\"\n");
  SmallVector<Token> Tokens;
  SmallString<8> Open("{{");
  SmallString<8> Close("}}");
  size_t Start = 0;

  while (Start < Template.size()) {
    LLVM_DEBUG(dbgs() << "[Tokenize Loop] Start:" << Start << ", Open:'" << Open
                      << "', Close:'" << Close << "'\n");
    Tag T = findNextTag(Template, Start, Open, Close);

    if (T.TagKind == Tag::Kind::None) {
      // No more tags, the rest is text.
      Tokens.emplace_back(Template.substr(Start));
      break;
    }

    // Add the text before the tag.
    if (T.StartPosition > Start) {
      StringRef Text = Template.substr(Start, T.StartPosition - Start);
      Tokens.emplace_back(Text);
    }

    if (auto NewDelims = processTag(T, Tokens, Ctx)) {
      std::tie(Open, Close) = *NewDelims;
    }

    // Move past the tag.
    Start = T.StartPosition + T.FullMatch.size();
  }

  // Fix up white spaces for:
  //   - open sections
  //   - inverted sections
  //   - close sections
  //   - comments
  //
  // This loop attempts to find standalone tokens and tries to trim out
  // the surrounding whitespace.
  // For example:
  // if you have the template string
  //  {{#section}} \n Example \n{{/section}}
  // The output should would be
  // For example:
  //  \n Example \n
  size_t LastIdx = Tokens.size() - 1;
  for (size_t Idx = 0, End = Tokens.size(); Idx < End; ++Idx) {
    Token &CurrentToken = Tokens[Idx];
    Token::Type CurrentType = CurrentToken.getType();
    // Check if token type requires cleanup.
    bool RequiresCleanUp = requiresCleanUp(CurrentType);

    if (!RequiresCleanUp)
      continue;

    // We adjust the token body if there's no text behind or ahead.
    // A token is considered to have no text ahead if the right of the previous
    // token is a newline followed by spaces.
    // A token is considered to have no text behind if the left of the next
    // token is spaces followed by a newline.
    // eg.
    //  "Line 1\n {{#section}} \n Line 2 \n {{/section}} \n Line 3"
    bool HasTextBehind = hasTextBehind(Idx, Tokens);
    bool HasTextAhead = hasTextAhead(Idx, Tokens);

    if ((!HasTextAhead && !HasTextBehind) || (!HasTextAhead && Idx == 0))
      stripTokenAhead(Tokens, Idx);

    if ((!HasTextBehind && !HasTextAhead) || (!HasTextBehind && Idx == LastIdx))
      stripTokenBefore(Tokens, Idx, CurrentToken, CurrentType);
  }
  return Tokens;
}

// Custom stream to escape strings.
class EscapeStringStream : public MustacheOutputStream {
public:
  explicit EscapeStringStream(llvm::raw_ostream &WrappedStream,
                              EscapeMap &Escape)
      : Escape(Escape), EscapeChars(Escape.keys().begin(), Escape.keys().end()),
        WrappedStream(WrappedStream) {
    SetUnbuffered();
  }

protected:
  void write_impl(const char *Ptr, size_t Size) override {
    StringRef Data(Ptr, Size);
    size_t Start = 0;
    while (Start < Size) {
      // Find the next character that needs to be escaped.
      size_t Next = Data.find_first_of(EscapeChars.str(), Start);

      // If no escapable characters are found, write the rest of the string.
      if (Next == StringRef::npos) {
        WrappedStream << Data.substr(Start);
        return;
      }

      // Write the chunk of text before the escapable character.
      if (Next > Start)
        WrappedStream << Data.substr(Start, Next - Start);

      // Look up and write the escaped version of the character.
      WrappedStream << Escape[Data[Next]];
      Start = Next + 1;
    }
  }

  uint64_t current_pos() const override { return WrappedStream.tell(); }

private:
  EscapeMap &Escape;
  SmallString<8> EscapeChars;
  llvm::raw_ostream &WrappedStream;
};

// Custom stream to add indentation used to for rendering partials.
class AddIndentationStringStream : public MustacheOutputStream {
public:
  explicit AddIndentationStringStream(raw_ostream &WrappedStream,
                                      size_t Indentation)
      : Indentation(Indentation), WrappedStream(WrappedStream),
        NeedsIndent(true), IsSuspended(false) {
    SetUnbuffered();
  }

  void suspendIndentation() override { IsSuspended = true; }
  void resumeIndentation() override { IsSuspended = false; }

protected:
  void write_impl(const char *Ptr, size_t Size) override {
    llvm::StringRef Data(Ptr, Size);
    SmallString<0> Indent;
    Indent.resize(Indentation, ' ');

    for (char C : Data) {
      LLVM_DEBUG(dbgs() << "[Indentation Stream] NeedsIndent:" << NeedsIndent
                        << ", C:'" << C << "', Indentation:" << Indentation
                        << "\n");
      if (NeedsIndent && C != '\n') {
        WrappedStream << Indent;
        NeedsIndent = false;
      }
      WrappedStream << C;
      if (C == '\n' && !IsSuspended)
        NeedsIndent = true;
    }
  }

  uint64_t current_pos() const override { return WrappedStream.tell(); }

private:
  size_t Indentation;
  raw_ostream &WrappedStream;
  bool NeedsIndent;
  bool IsSuspended;
};

class Parser {
public:
  Parser(StringRef TemplateStr, MustacheContext &Ctx)
      : Ctx(Ctx), TemplateStr(TemplateStr) {}

  AstPtr parse();

private:
  void parseMustache(ASTNode *Parent);
  void parseSection(ASTNode *Parent, ASTNode::Type Ty, const Accessor &A);

  MustacheContext &Ctx;
  SmallVector<Token> Tokens;
  size_t CurrentPtr;
  StringRef TemplateStr;
};

void Parser::parseSection(ASTNode *Parent, ASTNode::Type Ty,
                          const Accessor &A) {
  AstPtr CurrentNode = createNode(Ctx, Ty, A, Parent);
  size_t Start = CurrentPtr;
  parseMustache(CurrentNode);
  const size_t End = CurrentPtr - 1;
  SmallString<128> RawBody;
  for (std::size_t I = Start; I < End; I++)
    RawBody += Tokens[I].RawBody;
  CurrentNode->setRawBody(Ctx.Saver.save(StringRef(RawBody)));
  Parent->addChild(CurrentNode);
}

AstPtr Parser::parse() {
  Tokens = tokenize(TemplateStr, Ctx);
  CurrentPtr = 0;
  AstPtr RootNode = createRootNode(Ctx);
  parseMustache(RootNode);
  return RootNode;
}

void Parser::parseMustache(ASTNode *Parent) {

  while (CurrentPtr < Tokens.size()) {
    Token CurrentToken = Tokens[CurrentPtr];
    CurrentPtr++;
    ArrayRef<StringRef> A = CurrentToken.getAccessor();
    AstPtr CurrentNode;

    switch (CurrentToken.getType()) {
    case Token::Type::Text: {
      CurrentNode = createTextNode(Ctx, CurrentToken.TokenBody, Parent);
      Parent->addChild(CurrentNode);
      break;
    }
    case Token::Type::Variable: {
      CurrentNode = createNode(Ctx, ASTNode::Variable, A, Parent);
      Parent->addChild(CurrentNode);
      break;
    }
    case Token::Type::UnescapeVariable: {
      CurrentNode = createNode(Ctx, ASTNode::UnescapeVariable, A, Parent);
      Parent->addChild(CurrentNode);
      break;
    }
    case Token::Type::Partial: {
      CurrentNode = createNode(Ctx, ASTNode::Partial, A, Parent);
      CurrentNode->setIndentation(CurrentToken.getIndentation());
      Parent->addChild(CurrentNode);
      break;
    }
    case Token::Type::SectionOpen: {
      parseSection(Parent, ASTNode::Section, A);
      break;
    }
    case Token::Type::InvertSectionOpen: {
      parseSection(Parent, ASTNode::InvertSection, A);
      break;
    }
    case Token::Type::Comment:
    case Token::Type::SetDelimiter:
      break;
    case Token::Type::SectionClose:
      return;
    }
  }
}
static void toMustacheString(const json::Value &Data, raw_ostream &OS) {
  LLVM_DEBUG(dbgs() << "[To Mustache String] Kind: "
                    << jsonKindToString(Data.kind()) << ", Data: " << Data
                    << "\n");
  switch (Data.kind()) {
  case json::Value::Null:
    return;
  case json::Value::Number: {
    auto Num = *Data.getAsNumber();
    std::ostringstream SS;
    SS << Num;
    OS << SS.str();
    return;
  }
  case json::Value::String: {
    auto Str = *Data.getAsString();
    OS << Str.str();
    return;
  }

  case json::Value::Array: {
    auto Arr = *Data.getAsArray();
    if (Arr.empty())
      return;
    [[fallthrough]];
  }
  case json::Value::Object:
  case json::Value::Boolean: {
    llvm::json::OStream JOS(OS, 2);
    JOS.value(Data);
    break;
  }
  }
}

void ASTNode::renderRoot(const json::Value &CurrentCtx,
                         MustacheOutputStream &OS) {
  renderChild(CurrentCtx, OS);
}

void ASTNode::renderText(MustacheOutputStream &OS) { OS << Body; }

void ASTNode::renderPartial(const json::Value &CurrentCtx,
                            MustacheOutputStream &OS) {
  LLVM_DEBUG(dbgs() << "[Render Partial] Accessor:" << AccessorValue[0]
                    << ", Indentation:" << Indentation << "\n");
  auto Partial = Ctx.Partials.find(AccessorValue[0]);
  if (Partial != Ctx.Partials.end())
    renderPartial(CurrentCtx, OS, Partial->getValue());
}

void ASTNode::renderVariable(const json::Value &CurrentCtx,
                             MustacheOutputStream &OS) {
  auto Lambda = Ctx.Lambdas.find(AccessorValue[0]);
  if (Lambda != Ctx.Lambdas.end()) {
    renderLambdas(CurrentCtx, OS, Lambda->getValue());
  } else if (const json::Value *ContextPtr = findContext()) {
    EscapeStringStream ES(OS, Ctx.Escapes);
    toMustacheString(*ContextPtr, ES);
  }
}

void ASTNode::renderUnescapeVariable(const json::Value &CurrentCtx,
                                     MustacheOutputStream &OS) {
  LLVM_DEBUG(dbgs() << "[Render UnescapeVariable] Accessor:" << AccessorValue[0]
                    << "\n");
  auto Lambda = Ctx.Lambdas.find(AccessorValue[0]);
  if (Lambda != Ctx.Lambdas.end()) {
    renderLambdas(CurrentCtx, OS, Lambda->getValue());
  } else if (const json::Value *ContextPtr = findContext()) {
    OS.suspendIndentation();
    toMustacheString(*ContextPtr, OS);
    OS.resumeIndentation();
  }
}

void ASTNode::renderSection(const json::Value &CurrentCtx,
                            MustacheOutputStream &OS) {
  auto SectionLambda = Ctx.SectionLambdas.find(AccessorValue[0]);
  if (SectionLambda != Ctx.SectionLambdas.end()) {
    renderSectionLambdas(CurrentCtx, OS, SectionLambda->getValue());
    return;
  }

  const json::Value *ContextPtr = findContext();
  if (isContextFalsey(ContextPtr))
    return;

  if (const json::Array *Arr = ContextPtr->getAsArray()) {
    for (const json::Value &V : *Arr)
      renderChild(V, OS);
    return;
  }
  renderChild(*ContextPtr, OS);
}

void ASTNode::renderInvertSection(const json::Value &CurrentCtx,
                                  MustacheOutputStream &OS) {
  bool IsLambda = Ctx.SectionLambdas.contains(AccessorValue[0]);
  const json::Value *ContextPtr = findContext();
  if (isContextFalsey(ContextPtr) && !IsLambda) {
    renderChild(CurrentCtx, OS);
  }
}

void ASTNode::render(const llvm::json::Value &Data, MustacheOutputStream &OS) {
  if (Ty != Root && Ty != Text && AccessorValue.empty())
    return;
  // Set the parent context to the incoming context so that we
  // can walk up the context tree correctly in findContext().
  ParentContext = &Data;

  switch (Ty) {
  case Root:
    renderRoot(Data, OS);
    return;
  case Text:
    renderText(OS);
    return;
  case Partial:
    renderPartial(Data, OS);
    return;
  case Variable:
    renderVariable(Data, OS);
    return;
  case UnescapeVariable:
    renderUnescapeVariable(Data, OS);
    return;
  case Section:
    renderSection(Data, OS);
    return;
  case InvertSection:
    renderInvertSection(Data, OS);
    return;
  }
  llvm_unreachable("Invalid ASTNode type");
}

const json::Value *ASTNode::findContext() {
  // The mustache spec allows for dot notation to access nested values
  // a single dot refers to the current context.
  // We attempt to find the JSON context in the current node, if it is not
  // found, then we traverse the parent nodes to find the context until we
  // reach the root node or the context is found.
  if (AccessorValue.empty())
    return nullptr;
  if (AccessorValue[0] == ".")
    return ParentContext;

  const json::Object *CurrentContext = ParentContext->getAsObject();
  StringRef CurrentAccessor = AccessorValue[0];
  ASTNode *CurrentParent = Parent;

  while (!CurrentContext || !CurrentContext->get(CurrentAccessor)) {
    if (CurrentParent->Ty != Root) {
      CurrentContext = CurrentParent->ParentContext->getAsObject();
      CurrentParent = CurrentParent->Parent;
      continue;
    }
    return nullptr;
  }
  const json::Value *Context = nullptr;
  for (auto [Idx, Acc] : enumerate(AccessorValue)) {
    const json::Value *CurrentValue = CurrentContext->get(Acc);
    if (!CurrentValue)
      return nullptr;
    if (Idx < AccessorValue.size() - 1) {
      CurrentContext = CurrentValue->getAsObject();
      if (!CurrentContext)
        return nullptr;
    } else {
      Context = CurrentValue;
    }
  }
  return Context;
}

void ASTNode::renderChild(const json::Value &Contexts,
                          MustacheOutputStream &OS) {
  for (ASTNode &Child : Children)
    Child.render(Contexts, OS);
}

void ASTNode::renderPartial(const json::Value &Contexts,
                            MustacheOutputStream &OS, ASTNode *Partial) {
  LLVM_DEBUG(dbgs() << "[Render Partial Indentation] Indentation: " << Indentation << "\n");
  AddIndentationStringStream IS(OS, Indentation);
  Partial->render(Contexts, IS);
}

void ASTNode::renderLambdas(const llvm::json::Value &Contexts,
                            MustacheOutputStream &OS, Lambda &L) {
  json::Value LambdaResult = L();
  std::string LambdaStr;
  raw_string_ostream Output(LambdaStr);
  toMustacheString(LambdaResult, Output);
  Parser P(LambdaStr, Ctx);
  AstPtr LambdaNode = P.parse();

  EscapeStringStream ES(OS, Ctx.Escapes);
  if (Ty == Variable) {
    LambdaNode->render(Contexts, ES);
    return;
  }
  LambdaNode->render(Contexts, OS);
}

void ASTNode::renderSectionLambdas(const llvm::json::Value &Contexts,
                                   MustacheOutputStream &OS, SectionLambda &L) {
  json::Value Return = L(RawBody.str());
  if (isFalsey(Return))
    return;
  std::string LambdaStr;
  raw_string_ostream Output(LambdaStr);
  toMustacheString(Return, Output);
  Parser P(LambdaStr, Ctx);
  AstPtr LambdaNode = P.parse();
  LambdaNode->render(Contexts, OS);
}

void Template::render(const llvm::json::Value &Data, llvm::raw_ostream &OS) {
  RawMustacheOutputStream MOS(OS);
  Tree->render(Data, MOS);
}

void Template::registerPartial(std::string Name, std::string Partial) {
  StringRef SavedPartial = Ctx.Saver.save(Partial);
  Parser P(SavedPartial, Ctx);
  AstPtr PartialTree = P.parse();
  Ctx.Partials.insert(std::make_pair(Name, PartialTree));
}

void Template::registerLambda(std::string Name, Lambda L) {
  Ctx.Lambdas[Name] = L;
}

void Template::registerLambda(std::string Name, SectionLambda L) {
  Ctx.SectionLambdas[Name] = L;
}

void Template::overrideEscapeCharacters(EscapeMap E) {
  Ctx.Escapes = std::move(E);
}

Template::Template(StringRef TemplateStr, MustacheContext &Ctx) : Ctx(Ctx) {
  Parser P(TemplateStr, Ctx);
  Tree = P.parse();
  // The default behavior is to escape html entities.
  const EscapeMap HtmlEntities = {{'&', "&amp;"},
                                  {'<', "&lt;"},
                                  {'>', "&gt;"},
                                  {'"', "&quot;"},
                                  {'\'', "&#39;"}};
  overrideEscapeCharacters(HtmlEntities);
}

Template::Template(Template &&Other) noexcept
    : Ctx(Other.Ctx), Tree(Other.Tree) {
  Other.Tree = nullptr;
}

Template::~Template() = default;

} // namespace llvm::mustache

#undef DEBUG_TYPE
