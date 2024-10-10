//===-- Mustache.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Mustache.h"
#include "llvm/Support/Error.h"
#include <sstream>

using namespace llvm;
using namespace llvm::json;

namespace llvm {
namespace mustache {

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
  
  Token(StringRef Str);

  Token(StringRef RawBody, StringRef Str, char Identifier);

  StringRef getTokenBody() const { return TokenBody; };

  StringRef getRawBody() const { return RawBody; };

  void setTokenBody(StringRef NewBody) { TokenBody = NewBody.str(); };

  Accessor getAccessor() const { return Accessor; };

  Type getType() const { return TokenType; };

  void setIndentation(size_t NewIndentation) { Indentation = NewIndentation; };

  size_t getIndentation() const { return Indentation; };

  static Type getTokenType(char Identifier);

private:
  size_t Indentation;
  Type TokenType;
  // RawBody is the original string that was tokenized
  SmallString<0> RawBody;
  Accessor Accessor;
  // TokenBody is the original string with the identifier removed
  SmallString<0> TokenBody;
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

  ASTNode() : T(Type::Root), ParentContext(nullptr) {};

  ASTNode(StringRef Body, ASTNode *Parent)
      : T(Type::Text), Body(Body), Parent(Parent), ParentContext(nullptr),
        Indentation(0) {};

  // Constructor for Section/InvertSection/Variable/UnescapeVariable
  ASTNode(Type T, Accessor Accessor, ASTNode *Parent)
      : T(T), Parent(Parent), Children({}), Accessor(Accessor),
        ParentContext(nullptr), Indentation(0) {};

  void addChild(ASTNode *Child) { Children.emplace_back(Child); };

  void setBody(StringRef NewBody) { Body = NewBody; };

  void setRawBody(StringRef NewBody) { RawBody = NewBody; };

  void setIndentation(size_t NewIndentation) { Indentation = NewIndentation; };

  void render(const llvm::json::Value &Data, SmallString<0> &Output);

  void setUpNode(llvm::BumpPtrAllocator &Alloc, StringMap<ASTNode *> &Partials,
                 StringMap<Lambda> &Lambdas,
                 StringMap<SectionLambda> &SectionLambdas,
                 DenseMap<char, StringRef> &Escapes);

private:
  void renderLambdas(const llvm::json::Value &Contexts, SmallString<0> &Output,
                     Lambda &L);

  void renderSectionLambdas(const llvm::json::Value &Contexts,
                            SmallString<0> &Output, SectionLambda &L);

  void renderPartial(const llvm::json::Value &Contexts, SmallString<0> &Output,
                     ASTNode *Partial);

  void renderChild(const llvm::json::Value &Context, SmallString<0> &Output);

  const llvm::json::Value *findContext();

  llvm::BumpPtrAllocator *Allocator;
  StringMap<ASTNode *> *Partials;
  StringMap<Lambda> *Lambdas;
  StringMap<SectionLambda> *SectionLambdas;
  DenseMap<char, StringRef> *Escapes;
  Type T;
  size_t Indentation;
  SmallString<0> RawBody;
  SmallString<0> Body;
  ASTNode *Parent;
  std::vector<ASTNode *> Children;
  const Accessor Accessor;
  const llvm::json::Value *ParentContext;
};

SmallString<0> escapeString(StringRef Input,
                            DenseMap<char, StringRef> &Escape) {
  SmallString<0> Output;
  Output.reserve(Input.size());
  for (char C : Input) {
    auto It = Escape.find(C);
    if (It != Escape.end())
      Output += It->getSecond();
    else
      Output += C;
  }
  return Output;
}

Accessor split(StringRef Str, char Delimiter) {
  Accessor Tokens;
  if (Str == ".") {
    Tokens.emplace_back(Str);
    return Tokens;
  }
  StringRef Ref(Str);
  while (!Ref.empty()) {
    StringRef Part;
    std::tie(Part, Ref) = Ref.split(Delimiter);
    Tokens.emplace_back(Part.trim());
  }
  return Tokens;
}

void addIndentation(llvm::SmallString<0> &PartialStr, size_t IndentationSize) {
  std::string Indent(IndentationSize, ' ');
  llvm::SmallString<0> Result;
  for (size_t I = 0; I < PartialStr.size(); ++I) {
    Result.push_back(PartialStr[I]);
    if (PartialStr[I] == '\n' && I < PartialStr.size() - 1)
      Result.append(Indent);
  }
  PartialStr = Result;
}

Token::Token(StringRef RawBody, StringRef InnerBody, char Identifier)
    : RawBody(RawBody), TokenBody(InnerBody), Indentation(0) {
  TokenType = getTokenType(Identifier);
  if (TokenType == Type::Comment)
    return;

  StringRef AccessorStr =
      TokenType == Type::Variable ? InnerBody : InnerBody.substr(1);

  Accessor = split(AccessorStr.trim(), '.');
}

Token::Token(StringRef Str)
    : TokenType(Type::Text), RawBody(Str), Accessor({}), TokenBody(Str),
      Indentation(0) {}

Token::Type Token::getTokenType(char Identifier) {
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

// Function to check if there's no meaningful text behind
bool noTextBehind(size_t Idx, const SmallVector<Token, 0> &Tokens) {
  if (Idx == 0)
    return false;

  int PrevIdx = Idx - 1;
  if (Tokens[PrevIdx].getType() != Token::Type::Text)
    return false;

  const Token &PrevToken = Tokens[Idx - 1];
  StringRef TokenBody = PrevToken.getRawBody().rtrim(" \t\v\t");
  // We make an exception for when previous token is empty
  // and the current token is the second token
  // ex.
  //  " {{#section}}A{{/section}}"
  // the output of this is
  //  "A"
  return TokenBody.ends_with("\n") || (TokenBody.empty() && Idx == 1);
}

// Function to check if there's no meaningful text ahead
bool noTextAhead(size_t Idx, const SmallVector<Token, 0> &Tokens) {
  if (Idx >= Tokens.size() - 1)
    return false;

  int PrevIdx = Idx + 1;
  if (Tokens[Idx + 1].getType() != Token::Type::Text)
    return false;

  const Token &NextToken = Tokens[Idx + 1];
  StringRef TokenBody = NextToken.getRawBody().ltrim(" ");
  return TokenBody.starts_with("\r\n") || TokenBody.starts_with("\n");
}

bool requiresCleanUp(Token::Type T) {
  // We must clean up all the tokens that could contain child nodes
  return T == Token::Type::SectionOpen || T == Token::Type::InvertSectionOpen ||
         T == Token::Type::SectionClose || T == Token::Type::Comment ||
         T == Token::Type::Partial;
}

// Adjust next token body if there is no text ahead
// eg.
//  The template string
//  "{{! Comment }} \nLine 2"
// would be considered as no text ahead and should be render as
//  " Line 2"
void stripTokenAhead(SmallVector<Token, 0> &Tokens, size_t Idx) {
  Token &NextToken = Tokens[Idx + 1];
  StringRef NextTokenBody = NextToken.getTokenBody();
  // cut off the leading newline which could be \n or \r\n
  if (NextTokenBody.starts_with("\r\n"))
    NextToken.setTokenBody(NextTokenBody.substr(2));
  else if (NextTokenBody.starts_with("\n"))
    NextToken.setTokenBody(NextTokenBody.substr(1));
}

// Adjust previous token body if there no text behind
// eg.
//  The template string
//  " \t{{#section}}A{{/section}}"
// would be considered as no text ahead and should be render as
//  "A"
// The exception for this is partial tag which requires us to
// keep track of the indentation once it's rendered
void stripTokenBefore(SmallVector<Token, 0> &Tokens, size_t Idx,
                      Token &CurrentToken, Token::Type CurrentType) {
  Token &PrevToken = Tokens[Idx - 1];
  StringRef PrevTokenBody = PrevToken.getTokenBody();
  StringRef Unindented = PrevTokenBody.rtrim(" \t\v");
  size_t Indentation = PrevTokenBody.size() - Unindented.size();
  if (CurrentType != Token::Type::Partial)
    PrevToken.setTokenBody(Unindented);
  CurrentToken.setIndentation(Indentation);
}

// Simple tokenizer that splits the template into tokens.
// The mustache spec allows {{{ }}} to unescape variables
// but we don't support that here. An unescape variable
// is represented only by {{& variable}}.
SmallVector<Token, 0> tokenize(StringRef Template) {
  SmallVector<Token, 0> Tokens;
  StringRef Open("{{");
  StringRef Close("}}");
  size_t Start = 0;
  size_t DelimiterStart = Template.find(Open);
  if (DelimiterStart == StringRef::npos) {
    Tokens.emplace_back(Template);
    return Tokens;
  }
  while (DelimiterStart != StringRef::npos) {
    if (DelimiterStart != Start) {
      Tokens.emplace_back(Template.substr(Start, DelimiterStart - Start));
    }

    size_t DelimiterEnd = Template.find(Close, DelimiterStart);
    if (DelimiterEnd == StringRef::npos) {
      break;
    }

    // Extract the Interpolated variable without delimiters {{ and }}
    size_t InterpolatedStart = DelimiterStart + Open.size();
    size_t InterpolatedEnd = DelimiterEnd - DelimiterStart - Close.size();
    SmallString<0> Interpolated =
        Template.substr(InterpolatedStart, InterpolatedEnd);
    SmallString<0> RawBody({Open, Interpolated, Close});
    Tokens.emplace_back(RawBody, Interpolated, Interpolated[0]);
    Start = DelimiterEnd + Close.size();
    DelimiterStart = Template.find(Open, Start);
  }

  if (Start < Template.size())
    Tokens.emplace_back(Template.substr(Start));

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
  //  "Line 1\n {{#section}} \n Line 2 \n {{/section}} \n Line 3"
  // The output would be
  //  "Line 1\n Line 2\n Line 3"
  size_t LastIdx = Tokens.size() - 1;
  for (size_t Idx = 0, End = Tokens.size(); Idx < End; ++Idx) {
    Token &CurrentToken = Tokens[Idx];
    Token::Type CurrentType = CurrentToken.getType();
    // Check if token type requires cleanup
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
    bool NoTextBehind = noTextBehind(Idx, Tokens);
    bool NoTextAhead = noTextAhead(Idx, Tokens);

    if ((NoTextBehind && NoTextAhead) || (NoTextAhead && Idx == 0)) {
      stripTokenAhead(Tokens, Idx);
    }

    if (((NoTextBehind && NoTextAhead) || (NoTextBehind && Idx == LastIdx))) {
      stripTokenBefore(Tokens, Idx, CurrentToken, CurrentType);
    }
  }
  return Tokens;
}

class Parser {
public:
  Parser(StringRef TemplateStr, BumpPtrAllocator &Allocator)
      : Allocator(Allocator), TemplateStr(TemplateStr) {}

  ASTNode *parse();

private:
  void parseMustache(ASTNode *Parent);

  BumpPtrAllocator &Allocator;
  SmallVector<Token, 0> Tokens;
  size_t CurrentPtr;
  StringRef TemplateStr;
};

ASTNode *Parser::parse() {
  Tokens = tokenize(TemplateStr);
  CurrentPtr = 0;
  void *Root = Allocator.Allocate(sizeof(ASTNode), alignof(ASTNode));
  ASTNode *RootNode = new (Root) ASTNode();
  parseMustache(RootNode);
  return RootNode;
}

void Parser::parseMustache(ASTNode *Parent) {

  while (CurrentPtr < Tokens.size()) {
    Token CurrentToken = Tokens[CurrentPtr];
    CurrentPtr++;
    Accessor A = CurrentToken.getAccessor();
    ASTNode *CurrentNode;
    void *Node = Allocator.Allocate(sizeof(ASTNode), alignof(ASTNode));

    switch (CurrentToken.getType()) {
    case Token::Type::Text: {
      CurrentNode = new (Node) ASTNode(CurrentToken.getTokenBody(), Parent);
      Parent->addChild(CurrentNode);
      break;
    }
    case Token::Type::Variable: {
      CurrentNode = new (Node) ASTNode(ASTNode::Variable, A, Parent);
      Parent->addChild(CurrentNode);
      break;
    }
    case Token::Type::UnescapeVariable: {
      CurrentNode = new (Node) ASTNode(ASTNode::UnescapeVariable, A, Parent);
      Parent->addChild(CurrentNode);
      break;
    }
    case Token::Type::Partial: {
      CurrentNode = new (Node) ASTNode(ASTNode::Partial, A, Parent);
      CurrentNode->setIndentation(CurrentToken.getIndentation());
      Parent->addChild(CurrentNode);
      break;
    }
    case Token::Type::SectionOpen: {
      CurrentNode = new (Node) ASTNode(ASTNode::Section, A, Parent);
      size_t Start = CurrentPtr;
      parseMustache(CurrentNode);
      size_t End = CurrentPtr;
      SmallString<0> RawBody;
      for (std::size_t I = Start; I < End - 1; I++)
        RawBody += Tokens[I].getRawBody();
      CurrentNode->setRawBody(RawBody);
      Parent->addChild(CurrentNode);
      break;
    }
    case Token::Type::InvertSectionOpen: {
      CurrentNode = new (Node) ASTNode(ASTNode::InvertSection, A, Parent);
      size_t Start = CurrentPtr;
      parseMustache(CurrentNode);
      size_t End = CurrentPtr;
      SmallString<0> RawBody;
      for (size_t Idx = Start; Idx < End - 1; Idx++)
        RawBody += Tokens[Idx].getRawBody();
      CurrentNode->setRawBody(RawBody);
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

StringRef Template::render(Value &Data) {
  Tree->render(Data, Output);
  LocalAllocator.Reset();
  return Output.str();
}

void Template::registerPartial(StringRef Name, StringRef Partial) {
  Parser P = Parser(Partial, Allocator);
  ASTNode *PartialTree = P.parse();
  PartialTree->setUpNode(LocalAllocator, Partials, Lambdas, SectionLambdas,
                         Escapes);
  Partials.insert(std::make_pair(Name, PartialTree));
}

void Template::registerLambda(StringRef Name, Lambda L) { Lambdas[Name] = L; }

void Template::registerLambda(StringRef Name, SectionLambda L) {
  SectionLambdas[Name] = L;
}

void Template::registerEscape(DenseMap<char, StringRef> E) { Escapes = E; }

Template::Template(StringRef TemplateStr) {
  Parser P = Parser(TemplateStr, Allocator);
  Tree = P.parse();
  // the default behaviour is to escape html entities
  DenseMap<char, StringRef> HtmlEntities = {{'&', "&amp;"},
                                            {'<', "&lt;"},
                                            {'>', "&gt;"},
                                            {'"', "&quot;"},
                                            {'\'', "&#39;"}};
  registerEscape(HtmlEntities);
  Tree->setUpNode(LocalAllocator, Partials, Lambdas, SectionLambdas, Escapes);
}

void toJsonString(const Value &Data, SmallString<0> &Output) {
  if (Data.getAsNull())
    return;
  if (auto *Arr = Data.getAsArray())
    if (Arr->empty())
      return;
  if (Data.getAsString()) {
    Output = Data.getAsString()->str();
    return;
  }
  if (auto Num = Data.getAsNumber()) {
    std::ostringstream Oss;
    Oss << *Num;
    Output = Oss.str();
    return;
  }
  Output = formatv("{0:2}", Data);
}

bool isFalsey(const Value &V) {
  return V.getAsNull() || (V.getAsBoolean() && !V.getAsBoolean().value()) ||
         (V.getAsArray() && V.getAsArray()->empty()) ||
         (V.getAsObject() && V.getAsObject()->empty());
}

void ASTNode::render(const Value &Data, SmallString<0> &Output) {

  ParentContext = &Data;
  const Value *ContextPtr = T == Root ? ParentContext : findContext();
  const Value &Context = ContextPtr ? *ContextPtr : nullptr;

  switch (T) {
  case Root:
    renderChild(Data, Output);
    return;
  case Text:
    Output = Body;
    return;
  case Partial: {
    auto Partial = Partials->find(Accessor[0]);
    if (Partial != Partials->end())
      renderPartial(Data, Output, Partial->getValue());
    return;
  }
  case Variable: {
    auto Lambda = Lambdas->find(Accessor[0]);
    if (Lambda != Lambdas->end())
      renderLambdas(Data, Output, Lambda->getValue());
    else {
      toJsonString(Context, Output);
      Output = escapeString(Output, *Escapes);
    }
    return;
  }
  case UnescapeVariable: {
    auto Lambda = Lambdas->find(Accessor[0]);
    if (Lambda != Lambdas->end())
      renderLambdas(Data, Output, Lambda->getValue());
    else
      toJsonString(Context, Output);
    return;
  }
  case Section: {
    // Sections are not rendered if the context is falsey
    auto SectionLambda = SectionLambdas->find(Accessor[0]);
    bool IsLambda = SectionLambda != SectionLambdas->end();
    if (isFalsey(Context) && !IsLambda)
      return;

    if (IsLambda) {
      renderSectionLambdas(Data, Output, SectionLambda->getValue());
      return;
    }

    if (Context.getAsArray()) {
      SmallString<0> Result;
      const json::Array *Arr = Context.getAsArray();
      for (const Value &V : *Arr)
        renderChild(V, Result);
      Output = Result;
      return;
    }

    renderChild(Context, Output);
    return;
  }
  case InvertSection: {
    bool IsLambda = SectionLambdas->find(Accessor[0]) != SectionLambdas->end();

    if (!isFalsey(Context) || IsLambda)
      return;

    renderChild(Context, Output);
    return;
  }
  }
  llvm_unreachable("Invalid ASTNode type");
}

const Value *ASTNode::findContext() {
  // The mustache spec allows for dot notation to access nested values
  // a single dot refers to the current context.
  // We attempt to find the JSON context in the current node, if it is not
  // found, then we traverse the parent nodes to find the context until we
  // reach the root node or the context is found
  if (Accessor.empty())
    return nullptr;
  if (Accessor[0] == ".")
    return ParentContext;

  const json::Object *CurrentContext = ParentContext->getAsObject();
  StringRef CurrentAccessor = Accessor[0];
  ASTNode *CurrentParent = Parent;

  while (!CurrentContext || !CurrentContext->get(CurrentAccessor)) {
    if (CurrentParent->T != Root) {
      CurrentContext = CurrentParent->ParentContext->getAsObject();
      CurrentParent = CurrentParent->Parent;
      continue;
    }
    return nullptr;
  }
  const Value *Context = nullptr;
  for (auto [Idx, Acc] : enumerate(Accessor)) {
    const Value *CurrentValue = CurrentContext->get(Acc);
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

void ASTNode::renderChild(const Value &Contexts, SmallString<0> &Output) {
  for (ASTNode *Child : Children) {
    SmallString<0> ChildResult;
    Child->render(Contexts, ChildResult);
    Output += ChildResult;
  }
}

void ASTNode::renderPartial(const Value &Contexts, SmallString<0> &Output,
                            ASTNode *Partial) {
  Partial->render(Contexts, Output);
  addIndentation(Output, Indentation);
}

void ASTNode::renderLambdas(const Value &Contexts, SmallString<0> &Output,
                            Lambda &L) {
  Value LambdaResult = L();
  SmallString<0> LambdaStr;
  toJsonString(LambdaResult, LambdaStr);
  Parser P = Parser(LambdaStr, *Allocator);
  ASTNode *LambdaNode = P.parse();
  LambdaNode->setUpNode(*Allocator, *Partials, *Lambdas, *SectionLambdas,
                        *Escapes);
  LambdaNode->render(Contexts, Output);
  if (T == Variable)
    Output = escapeString(Output, *Escapes);
  return;
}

void ASTNode::renderSectionLambdas(const Value &Contexts,
                                   SmallString<0> &Output, SectionLambda &L) {
  Value Return = L(RawBody);
  if (isFalsey(Return))
    return;
  SmallString<0> LambdaStr;
  toJsonString(Return, LambdaStr);
  Parser P = Parser(LambdaStr, *Allocator);
  ASTNode *LambdaNode = P.parse();
  LambdaNode->setUpNode(*Allocator, *Partials, *Lambdas, *SectionLambdas,
                        *Escapes);
  LambdaNode->render(Contexts, Output);
  return;
}

void ASTNode::setUpNode(llvm::BumpPtrAllocator &Alloc,
                        StringMap<ASTNode *> &Par, StringMap<Lambda> &L,
                        StringMap<SectionLambda> &SC,
                        DenseMap<char, StringRef> &E) {
  // Passed down datastructures needed for rendering to
  // the children nodes. This must be called before rendering
  Allocator = &Alloc;
  Partials = &Par;
  Lambdas = &L;
  SectionLambdas = &SC;
  Escapes = &E;
  for (ASTNode *Child : Children)
    Child->setUpNode(Alloc, Par, L, SC, E);
}
} // namespace mustache
} // namespace llvm
