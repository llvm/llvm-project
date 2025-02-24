//===-- Mustache.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/Support/Mustache.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>

using namespace llvm;
using namespace llvm::mustache;

namespace {

using Accessor = SmallVector<std::string>;

static bool isFalsey(const json::Value &V) {
  return V.getAsNull() || (V.getAsBoolean() && !V.getAsBoolean().value()) ||
         (V.getAsArray() && V.getAsArray()->empty());
}

static Accessor splitMustacheString(StringRef Str) {
  // We split the mustache string into an accessor.
  // For example:
  //    "a.b.c" would be split into {"a", "b", "c"}
  // We make an exception for a single dot which
  // refers to the current context.
  Accessor Tokens;
  if (Str == ".") {
    Tokens.emplace_back(Str);
    return Tokens;
  }
  while (!Str.empty()) {
    StringRef Part;
    std::tie(Part, Str) = Str.split(".");
    Tokens.emplace_back(Part.trim());
  }
  return Tokens;
}

} // namespace

namespace llvm::mustache {

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
  };

  Token(std::string Str)
      : TokenType(Type::Text), RawBody(std::move(Str)), TokenBody(RawBody),
        Accessor({}), Indentation(0){};

  Token(std::string RawBody, std::string TokenBody, char Identifier)
      : RawBody(std::move(RawBody)), TokenBody(std::move(TokenBody)),
        Indentation(0) {
    TokenType = getTokenType(Identifier);
    if (TokenType == Type::Comment)
      return;
    StringRef AccessorStr(this->TokenBody);
    if (TokenType != Type::Variable)
      AccessorStr = AccessorStr.substr(1);
    Accessor = splitMustacheString(StringRef(AccessorStr).trim());
  }

  Accessor getAccessor() const { return Accessor; }

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
    default:
      return Type::Variable;
    }
  }

  Type TokenType;
  // RawBody is the original string that was tokenized.
  std::string RawBody;
  // TokenBody is the original string with the identifier removed.
  std::string TokenBody;
  Accessor Accessor;
  size_t Indentation;
};

class ASTNode {
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

  ASTNode(llvm::BumpPtrAllocator &Alloc, llvm::StringMap<ASTNode *> &Partials,
          llvm::StringMap<Lambda> &Lambdas,
          llvm::StringMap<SectionLambda> &SectionLambdas,
          llvm::DenseMap<char, std::string> &Escapes)
      : Allocator(Alloc), Partials(Partials), Lambdas(Lambdas),
        SectionLambdas(SectionLambdas), Escapes(Escapes), Ty(Type::Root),
        Parent(nullptr), ParentContext(nullptr) {}

  ASTNode(std::string Body, ASTNode *Parent, llvm::BumpPtrAllocator &Alloc,
          llvm::StringMap<ASTNode *> &Partials,
          llvm::StringMap<Lambda> &Lambdas,
          llvm::StringMap<SectionLambda> &SectionLambdas,
          llvm::DenseMap<char, std::string> &Escapes)
      : Allocator(Alloc), Partials(Partials), Lambdas(Lambdas),
        SectionLambdas(SectionLambdas), Escapes(Escapes), Ty(Type::Text),
        Body(std::move(Body)), Parent(Parent), ParentContext(nullptr) {}

  // Constructor for Section/InvertSection/Variable/UnescapeVariable Nodes
  ASTNode(Type Ty, Accessor Accessor, ASTNode *Parent,
          llvm::BumpPtrAllocator &Alloc, llvm::StringMap<ASTNode *> &Partials,
          llvm::StringMap<Lambda> &Lambdas,
          llvm::StringMap<SectionLambda> &SectionLambdas,
          llvm::DenseMap<char, std::string> &Escapes)
      : Allocator(Alloc), Partials(Partials), Lambdas(Lambdas),
        SectionLambdas(SectionLambdas), Escapes(Escapes), Ty(Ty),
        Parent(Parent), Accessor(std::move(Accessor)), ParentContext(nullptr) {}

  void addChild(ASTNode *Child) { Children.emplace_back(Child); };

  void setRawBody(std::string NewBody) { RawBody = std::move(NewBody); };

  void setIndentation(size_t NewIndentation) { Indentation = NewIndentation; };

  void render(const llvm::json::Value &Data, llvm::raw_ostream &OS);

private:
  void renderLambdas(const llvm::json::Value &Contexts, llvm::raw_ostream &OS,
                     Lambda &L);

  void renderSectionLambdas(const llvm::json::Value &Contexts,
                            llvm::raw_ostream &OS, SectionLambda &L);

  void renderPartial(const llvm::json::Value &Contexts, llvm::raw_ostream &OS,
                     ASTNode *Partial);

  void renderChild(const llvm::json::Value &Context, llvm::raw_ostream &OS);

  const llvm::json::Value *findContext();

  llvm::BumpPtrAllocator &Allocator;
  StringMap<ASTNode *> &Partials;
  StringMap<Lambda> &Lambdas;
  StringMap<SectionLambda> &SectionLambdas;
  DenseMap<char, std::string> &Escapes;
  Type Ty;
  size_t Indentation = 0;
  std::string RawBody;
  std::string Body;
  ASTNode *Parent;
  // TODO: switch implementation to SmallVector<T>
  std::vector<ASTNode *> Children;
  const Accessor Accessor;
  const llvm::json::Value *ParentContext;
};

// A wrapper for arena allocator for ASTNodes
ASTNode *createRootNode(void *Node, llvm::BumpPtrAllocator &Alloc,
                        llvm::StringMap<ASTNode *> &Partials,
                        llvm::StringMap<Lambda> &Lambdas,
                        llvm::StringMap<SectionLambda> &SectionLambdas,
                        llvm::DenseMap<char, std::string> &Escapes) {
  return new (Node) ASTNode(Alloc, Partials, Lambdas, SectionLambdas, Escapes);
};

ASTNode *createNode(void *Node, ASTNode::Type T, Accessor A, ASTNode *Parent,
                    llvm::BumpPtrAllocator &Alloc,
                    llvm::StringMap<ASTNode *> &Partials,
                    llvm::StringMap<Lambda> &Lambdas,
                    llvm::StringMap<SectionLambda> &SectionLambdas,
                    llvm::DenseMap<char, std::string> &Escapes) {
  return new (Node) ASTNode(T, std::move(A), Parent, Alloc, Partials, Lambdas,
                            SectionLambdas, Escapes);
};

ASTNode *createTextNode(void *Node, std::string Body, ASTNode *Parent,
                        llvm::BumpPtrAllocator &Alloc,
                        llvm::StringMap<ASTNode *> &Partials,
                        llvm::StringMap<Lambda> &Lambdas,
                        llvm::StringMap<SectionLambda> &SectionLambdas,
                        llvm::DenseMap<char, std::string> &Escapes) {
  return new (Node) ASTNode(std::move(Body), Parent, Alloc, Partials, Lambdas,
                            SectionLambdas, Escapes);
};

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
bool hasTextBehind(size_t Idx, const ArrayRef<Token> &Tokens) {
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
bool hasTextAhead(size_t Idx, const ArrayRef<Token> &Tokens) {
  if (Idx >= Tokens.size() - 1)
    return true;

  size_t NextIdx = Idx + 1;
  if (Tokens[NextIdx].getType() != Token::Type::Text)
    return true;

  const Token &NextToken = Tokens[NextIdx];
  StringRef TokenBody = StringRef(NextToken.RawBody).ltrim(" ");
  return !TokenBody.starts_with("\r\n") && !TokenBody.starts_with("\n");
}

bool requiresCleanUp(Token::Type T) {
  // We must clean up all the tokens that could contain child nodes.
  return T == Token::Type::SectionOpen || T == Token::Type::InvertSectionOpen ||
         T == Token::Type::SectionClose || T == Token::Type::Comment ||
         T == Token::Type::Partial;
}

// Adjust next token body if there is no text ahead.
// For example:
// The template string
//  "{{! Comment }} \nLine 2"
// would be considered as no text ahead and should be rendered as
//  " Line 2"
void stripTokenAhead(SmallVectorImpl<Token> &Tokens, size_t Idx) {
  Token &NextToken = Tokens[Idx + 1];
  StringRef NextTokenBody = NextToken.TokenBody;
  // Cut off the leading newline which could be \n or \r\n.
  if (NextTokenBody.starts_with("\r\n"))
    NextToken.TokenBody = std::move(NextTokenBody.substr(2).str());
  else if (NextTokenBody.starts_with("\n"))
    NextToken.TokenBody = std::move(NextTokenBody.substr(1).str());
}

// Adjust previous token body if there no text behind.
// For example:
//  The template string
//  " \t{{#section}}A{{/section}}"
// would be considered as having no text ahead and would be render as
//  "A"
// The exception for this is partial tag which requires us to
// keep track of the indentation once it's rendered.
void stripTokenBefore(SmallVectorImpl<Token> &Tokens, size_t Idx,
                      Token &CurrentToken, Token::Type CurrentType) {
  Token &PrevToken = Tokens[Idx - 1];
  StringRef PrevTokenBody = PrevToken.TokenBody;
  StringRef Unindented = PrevTokenBody.rtrim(" \r\t\v");
  size_t Indentation = PrevTokenBody.size() - Unindented.size();
  if (CurrentType != Token::Type::Partial)
    PrevToken.TokenBody = std::move(Unindented.str());
  CurrentToken.setIndentation(Indentation);
}

// Simple tokenizer that splits the template into tokens.
// The mustache spec allows {{{ }}} to unescape variables,
// but we don't support that here. An unescape variable
// is represented only by {{& variable}}.
SmallVector<Token> tokenize(StringRef Template) {
  SmallVector<Token> Tokens;
  StringLiteral Open("{{");
  StringLiteral Close("}}");
  size_t Start = 0;
  size_t DelimiterStart = Template.find(Open);
  if (DelimiterStart == StringRef::npos) {
    Tokens.emplace_back(Template.str());
    return Tokens;
  }
  while (DelimiterStart != StringRef::npos) {
    if (DelimiterStart != Start)
      Tokens.emplace_back(Template.substr(Start, DelimiterStart - Start).str());
    size_t DelimiterEnd = Template.find(Close, DelimiterStart);
    if (DelimiterEnd == StringRef::npos)
      break;

    // Extract the Interpolated variable without delimiters.
    size_t InterpolatedStart = DelimiterStart + Open.size();
    size_t InterpolatedEnd = DelimiterEnd - DelimiterStart - Close.size();
    std::string Interpolated =
        Template.substr(InterpolatedStart, InterpolatedEnd).str();
    std::string RawBody = Open.str() + Interpolated + Close.str();
    Tokens.emplace_back(RawBody, Interpolated, Interpolated[0]);
    Start = DelimiterEnd + Close.size();
    DelimiterStart = Template.find(Open, Start);
  }

  if (Start < Template.size())
    Tokens.emplace_back(Template.substr(Start).str());

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
class EscapeStringStream : public raw_ostream {
public:
  explicit EscapeStringStream(llvm::raw_ostream &WrappedStream,
                              DenseMap<char, std::string> &Escape)
      : Escape(Escape), WrappedStream(WrappedStream) {
    SetUnbuffered();
  }

protected:
  void write_impl(const char *Ptr, size_t Size) override {
    llvm::StringRef Data(Ptr, Size);
    for (char C : Data) {
      auto It = Escape.find(C);
      if (It != Escape.end())
        WrappedStream << It->getSecond();
      else
        WrappedStream << C;
    }
  }

  uint64_t current_pos() const override { return WrappedStream.tell(); }

private:
  DenseMap<char, std::string> &Escape;
  llvm::raw_ostream &WrappedStream;
};

// Custom stream to add indentation used to for rendering partials.
class AddIndentationStringStream : public raw_ostream {
public:
  explicit AddIndentationStringStream(llvm::raw_ostream &WrappedStream,
                                      size_t Indentation)
      : Indentation(Indentation), WrappedStream(WrappedStream) {
    SetUnbuffered();
  }

protected:
  void write_impl(const char *Ptr, size_t Size) override {
    llvm::StringRef Data(Ptr, Size);
    SmallString<0> Indent;
    Indent.resize(Indentation, ' ');
    for (char C : Data) {
      WrappedStream << C;
      if (C == '\n')
        WrappedStream << Indent;
    }
  }

  uint64_t current_pos() const override { return WrappedStream.tell(); }

private:
  size_t Indentation;
  llvm::raw_ostream &WrappedStream;
};

class Parser {
public:
  Parser(StringRef TemplateStr, BumpPtrAllocator &Allocator)
      : ASTAllocator(Allocator), TemplateStr(TemplateStr) {}

  ASTNode *parse(llvm::BumpPtrAllocator &RenderAlloc,
                 llvm::StringMap<ASTNode *> &Partials,
                 llvm::StringMap<Lambda> &Lambdas,
                 llvm::StringMap<SectionLambda> &SectionLambdas,
                 llvm::DenseMap<char, std::string> &Escapes);

private:
  void parseMustache(ASTNode *Parent, llvm::BumpPtrAllocator &Alloc,
                     llvm::StringMap<ASTNode *> &Partials,
                     llvm::StringMap<Lambda> &Lambdas,
                     llvm::StringMap<SectionLambda> &SectionLambdas,
                     llvm::DenseMap<char, std::string> &Escapes);

  BumpPtrAllocator &ASTAllocator;
  SmallVector<Token> Tokens;
  size_t CurrentPtr;
  StringRef TemplateStr;
};

ASTNode *Parser::parse(llvm::BumpPtrAllocator &RenderAlloc,
                       llvm::StringMap<ASTNode *> &Partials,
                       llvm::StringMap<Lambda> &Lambdas,
                       llvm::StringMap<SectionLambda> &SectionLambdas,
                       llvm::DenseMap<char, std::string> &Escapes) {
  Tokens = tokenize(TemplateStr);
  CurrentPtr = 0;
  void *Root = ASTAllocator.Allocate(sizeof(ASTNode), alignof(ASTNode));
  ASTNode *RootNode = createRootNode(Root, RenderAlloc, Partials, Lambdas,
                                     SectionLambdas, Escapes);
  parseMustache(RootNode, RenderAlloc, Partials, Lambdas, SectionLambdas,
                Escapes);
  return RootNode;
}

void Parser::parseMustache(ASTNode *Parent, llvm::BumpPtrAllocator &Alloc,
                           llvm::StringMap<ASTNode *> &Partials,
                           llvm::StringMap<Lambda> &Lambdas,
                           llvm::StringMap<SectionLambda> &SectionLambdas,
                           llvm::DenseMap<char, std::string> &Escapes) {

  while (CurrentPtr < Tokens.size()) {
    Token CurrentToken = Tokens[CurrentPtr];
    CurrentPtr++;
    Accessor A = CurrentToken.getAccessor();
    ASTNode *CurrentNode;
    void *Node = ASTAllocator.Allocate(sizeof(ASTNode), alignof(ASTNode));

    switch (CurrentToken.getType()) {
    case Token::Type::Text: {
      CurrentNode =
          createTextNode(Node, std::move(CurrentToken.TokenBody), Parent, Alloc,
                         Partials, Lambdas, SectionLambdas, Escapes);
      Parent->addChild(CurrentNode);
      break;
    }
    case Token::Type::Variable: {
      CurrentNode =
          createNode(Node, ASTNode::Variable, std::move(A), Parent, Alloc,
                     Partials, Lambdas, SectionLambdas, Escapes);
      Parent->addChild(CurrentNode);
      break;
    }
    case Token::Type::UnescapeVariable: {
      CurrentNode =
          createNode(Node, ASTNode::UnescapeVariable, std::move(A), Parent,
                     Alloc, Partials, Lambdas, SectionLambdas, Escapes);
      Parent->addChild(CurrentNode);
      break;
    }
    case Token::Type::Partial: {
      CurrentNode =
          createNode(Node, ASTNode::Partial, std::move(A), Parent, Alloc,
                     Partials, Lambdas, SectionLambdas, Escapes);
      CurrentNode->setIndentation(CurrentToken.getIndentation());
      Parent->addChild(CurrentNode);
      break;
    }
    case Token::Type::SectionOpen: {
      CurrentNode = createNode(Node, ASTNode::Section, A, Parent, Alloc,
                               Partials, Lambdas, SectionLambdas, Escapes);
      size_t Start = CurrentPtr;
      parseMustache(CurrentNode, Alloc, Partials, Lambdas, SectionLambdas,
                    Escapes);
      const size_t End = CurrentPtr - 1;
      std::string RawBody;
      for (std::size_t I = Start; I < End; I++)
        RawBody += Tokens[I].RawBody;
      CurrentNode->setRawBody(std::move(RawBody));
      Parent->addChild(CurrentNode);
      break;
    }
    case Token::Type::InvertSectionOpen: {
      CurrentNode = createNode(Node, ASTNode::InvertSection, A, Parent, Alloc,
                               Partials, Lambdas, SectionLambdas, Escapes);
      size_t Start = CurrentPtr;
      parseMustache(CurrentNode, Alloc, Partials, Lambdas, SectionLambdas,
                    Escapes);
      const size_t End = CurrentPtr - 1;
      std::string RawBody;
      for (size_t Idx = Start; Idx < End; Idx++)
        RawBody += Tokens[Idx].RawBody;
      CurrentNode->setRawBody(std::move(RawBody));
      Parent->addChild(CurrentNode);
      break;
    }
    case Token::Type::Comment:
      break;
    case Token::Type::SectionClose:
      return;
    }
  }
}
void toMustacheString(const json::Value &Data, raw_ostream &OS) {
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

void ASTNode::render(const json::Value &Data, raw_ostream &OS) {
  ParentContext = &Data;
  const json::Value *ContextPtr = Ty == Root ? ParentContext : findContext();
  const json::Value &Context = ContextPtr ? *ContextPtr : nullptr;

  switch (Ty) {
  case Root:
    renderChild(Data, OS);
    return;
  case Text:
    OS << Body;
    return;
  case Partial: {
    auto Partial = Partials.find(Accessor[0]);
    if (Partial != Partials.end())
      renderPartial(Data, OS, Partial->getValue());
    return;
  }
  case Variable: {
    auto Lambda = Lambdas.find(Accessor[0]);
    if (Lambda != Lambdas.end())
      renderLambdas(Data, OS, Lambda->getValue());
    else {
      EscapeStringStream ES(OS, Escapes);
      toMustacheString(Context, ES);
    }
    return;
  }
  case UnescapeVariable: {
    auto Lambda = Lambdas.find(Accessor[0]);
    if (Lambda != Lambdas.end())
      renderLambdas(Data, OS, Lambda->getValue());
    else
      toMustacheString(Context, OS);
    return;
  }
  case Section: {
    // Sections are not rendered if the context is falsey.
    auto SectionLambda = SectionLambdas.find(Accessor[0]);
    bool IsLambda = SectionLambda != SectionLambdas.end();
    if (isFalsey(Context) && !IsLambda)
      return;

    if (IsLambda) {
      renderSectionLambdas(Data, OS, SectionLambda->getValue());
      return;
    }

    if (Context.getAsArray()) {
      const json::Array *Arr = Context.getAsArray();
      for (const json::Value &V : *Arr)
        renderChild(V, OS);
      return;
    }
    renderChild(Context, OS);
    return;
  }
  case InvertSection: {
    bool IsLambda = SectionLambdas.find(Accessor[0]) != SectionLambdas.end();
    if (!isFalsey(Context) || IsLambda)
      return;
    renderChild(Context, OS);
    return;
  }
  }
  llvm_unreachable("Invalid ASTNode type");
}

const json::Value *ASTNode::findContext() {
  // The mustache spec allows for dot notation to access nested values
  // a single dot refers to the current context.
  // We attempt to find the JSON context in the current node, if it is not
  // found, then we traverse the parent nodes to find the context until we
  // reach the root node or the context is found.
  if (Accessor.empty())
    return nullptr;
  if (Accessor[0] == ".")
    return ParentContext;

  const json::Object *CurrentContext = ParentContext->getAsObject();
  StringRef CurrentAccessor = Accessor[0];
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
  for (auto [Idx, Acc] : enumerate(Accessor)) {
    const json::Value *CurrentValue = CurrentContext->get(Acc);
    if (!CurrentValue)
      return nullptr;
    if (Idx < Accessor.size() - 1) {
      CurrentContext = CurrentValue->getAsObject();
      if (!CurrentContext)
        return nullptr;
    } else
      Context = CurrentValue;
  }
  return Context;
}

void ASTNode::renderChild(const json::Value &Contexts, llvm::raw_ostream &OS) {
  for (ASTNode *Child : Children)
    Child->render(Contexts, OS);
}

void ASTNode::renderPartial(const json::Value &Contexts, llvm::raw_ostream &OS,
                            ASTNode *Partial) {
  AddIndentationStringStream IS(OS, Indentation);
  Partial->render(Contexts, IS);
}

void ASTNode::renderLambdas(const json::Value &Contexts, llvm::raw_ostream &OS,
                            Lambda &L) {
  json::Value LambdaResult = L();
  std::string LambdaStr;
  raw_string_ostream Output(LambdaStr);
  toMustacheString(LambdaResult, Output);
  Parser P = Parser(LambdaStr, Allocator);
  ASTNode *LambdaNode =
      P.parse(Allocator, Partials, Lambdas, SectionLambdas, Escapes);

  EscapeStringStream ES(OS, Escapes);
  if (Ty == Variable) {
    LambdaNode->render(Contexts, ES);
    return;
  }
  LambdaNode->render(Contexts, OS);
}

void ASTNode::renderSectionLambdas(const json::Value &Contexts,
                                   llvm::raw_ostream &OS, SectionLambda &L) {
  json::Value Return = L(RawBody);
  if (isFalsey(Return))
    return;
  std::string LambdaStr;
  raw_string_ostream Output(LambdaStr);
  toMustacheString(Return, Output);
  Parser P = Parser(LambdaStr, Allocator);
  ASTNode *LambdaNode =
      P.parse(Allocator, Partials, Lambdas, SectionLambdas, Escapes);
  LambdaNode->render(Contexts, OS);
  return;
}

void Template::render(const json::Value &Data, llvm::raw_ostream &OS) {
  Tree->render(Data, OS);
  RenderAllocator.Reset();
}

void Template::registerPartial(std::string Name, std::string Partial) {
  Parser P = Parser(Partial, AstAllocator);
  ASTNode *PartialTree =
      P.parse(RenderAllocator, Partials, Lambdas, SectionLambdas, Escapes);
  Partials.insert(std::make_pair(Name, PartialTree));
}

void Template::registerLambda(std::string Name, Lambda L) { Lambdas[Name] = L; }

void Template::registerLambda(std::string Name, SectionLambda L) {
  SectionLambdas[Name] = L;
}

void Template::overrideEscapeCharacters(DenseMap<char, std::string> E) {
  Escapes = std::move(E);
}

Template::Template(StringRef TemplateStr) {
  Parser P = Parser(TemplateStr, AstAllocator);
  Tree = P.parse(RenderAllocator, Partials, Lambdas, SectionLambdas, Escapes);
  // The default behavior is to escape html entities.
  DenseMap<char, std::string> HtmlEntities = {{'&', "&amp;"},
                                              {'<', "&lt;"},
                                              {'>', "&gt;"},
                                              {'"', "&quot;"},
                                              {'\'', "&#39;"}};
  overrideEscapeCharacters(HtmlEntities);
}

Template::Template(Template &&Other) noexcept
    : Partials(std::move(Other.Partials)), Lambdas(std::move(Other.Lambdas)),
      SectionLambdas(std::move(Other.SectionLambdas)),
      Escapes(std::move(Other.Escapes)), Tree(Other.Tree),
      AstAllocator(std::move(Other.AstAllocator)),
      RenderAllocator(std::move(Other.RenderAllocator)) {
  Other.Tree = nullptr;
}

Template &Template::operator=(Template &&Other) noexcept {
  if (this != &Other) {
    Partials = std::move(Other.Partials);
    Lambdas = std::move(Other.Lambdas);
    SectionLambdas = std::move(Other.SectionLambdas);
    Escapes = std::move(Other.Escapes);
    Tree = Other.Tree;
    AstAllocator = std::move(Other.AstAllocator);
    RenderAllocator = std::move(Other.RenderAllocator);
    Other.Tree = nullptr;
  }
  return *this;
}
} // namespace llvm::mustache
