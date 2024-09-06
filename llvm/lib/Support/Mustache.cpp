//===-- Mustache.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Mustache.h"
#include "llvm/Support/Error.h"

using namespace llvm;
using namespace llvm::json;
using namespace llvm::mustache;

SmallString<128> escapeString(StringRef Input,
                              DenseMap<char, StringRef> &Escape) {
  SmallString<128> EscapedString("");
  for (char C : Input) {
    if (Escape.find(C) != Escape.end())
      EscapedString += Escape[C];
    else
      EscapedString += C;
  }
  return EscapedString;
}

std::vector<SmallString<128>> split(StringRef Str, char Delimiter) {
  std::vector<SmallString<128>> Tokens;
  if (Str == ".") {
    Tokens.push_back(Str);
    return Tokens;
  }
  StringRef Ref(Str);
  while (!Ref.empty()) {
    llvm::StringRef Part;
    std::tie(Part, Ref) = Ref.split(Delimiter);
    Tokens.push_back(Part.trim());
  }
  return Tokens;
}

Token::Token(StringRef RawBody, StringRef InnerBody, char Identifier)
    : RawBody(RawBody), TokenBody(InnerBody) {

  TokenType = getTokenType(Identifier);
  if (TokenType == Type::Comment)
    return;

  StringRef AccessorStr = InnerBody;
  if (TokenType != Type::Variable)
    AccessorStr = InnerBody.substr(1);

  Accessor = split(AccessorStr.trim(), '.');
}

Token::Token(StringRef Str)
    : RawBody(Str), TokenType(Type::Text), TokenBody(Str), Accessor({}) {}

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

std::vector<Token> tokenize(StringRef Template) {
  // Simple tokenizer that splits the template into tokens
  // the mustache spec allows {{{ }}} to unescape variables
  // but we don't support that here unescape variable
  // is represented only by {{& variable}}
  std::vector<Token> Tokens;
  SmallString<128> Open("{{");
  SmallString<128> Close("}}");
  std::size_t Start = 0;
  std::size_t DelimiterStart = Template.find(Open);
  if (DelimiterStart == StringRef::npos) {
    Tokens.push_back(Token(Template));
    return Tokens;
  }
  while (DelimiterStart != StringRef::npos) {
    if (DelimiterStart != Start) {
      Token TextToken = Token(Template.substr(Start, DelimiterStart - Start));
      Tokens.push_back(TextToken);
    }

    std::size_t DelimiterEnd = Template.find(Close, DelimiterStart);
    if (DelimiterEnd == StringRef::npos) {
      break;
    }

    SmallString<128> Interpolated =
        Template.substr(DelimiterStart + Open.size(),
                        DelimiterEnd - DelimiterStart - Close.size());
    SmallString<128> RawBody;
    RawBody += Open;
    RawBody += Interpolated;
    RawBody += Close;

    Tokens.push_back(Token(RawBody, Interpolated, Interpolated[0]));
    Start = DelimiterEnd + Close.size();
    DelimiterStart = Template.find(Open, Start);
  }

  if (Start < Template.size())
    Tokens.push_back(Token(Template.substr(Start)));

  // fix up white spaces for
  // open sections/inverted sections/close section/comment
  for (std::size_t I = 0; I < Tokens.size(); I++) {
    Token::Type CurrentType = Tokens[I].getType();
    bool RequiresCleanUp = CurrentType == Token::Type::SectionOpen ||
                           CurrentType == Token::Type::InvertSectionOpen ||
                           CurrentType == Token::Type::SectionClose ||
                           CurrentType == Token::Type::Comment ||
                           CurrentType == Token::Type::Partial;

    bool NoTextBehind = false;
    bool NoTextAhead = false;
    if (I > 0 && Tokens[I - 1].getType() == Token::Type::Text &&
        RequiresCleanUp) {
      Token &PrevToken = Tokens[I - 1];
      StringRef TokenBody = PrevToken.getRawBody().rtrim(" \t\v\t");
      if (TokenBody.ends_with("\n") || TokenBody.ends_with("\r\n") ||
          (TokenBody.empty() && I == 1))
        NoTextBehind = true;
    }
    if (I < Tokens.size() - 1 && Tokens[I + 1].getType() == Token::Type::Text &&
        RequiresCleanUp) {
      Token &NextToken = Tokens[I + 1];
      StringRef TokenBody = NextToken.getRawBody().ltrim(" ");
      if (TokenBody.starts_with("\r\n") || TokenBody.starts_with("\n"))
        NoTextAhead = true;
    }

    if ((NoTextBehind && NoTextAhead) || (NoTextAhead && I == 0)) {
      Token &NextToken = Tokens[I + 1];
      StringRef NextTokenBody = NextToken.getTokenBody();
      if (NextTokenBody.starts_with("\r\n"))
        NextToken.setTokenBody(NextTokenBody.substr(2));
      else if (NextToken.getTokenBody().starts_with("\n"))
        NextToken.setTokenBody(NextTokenBody.substr(1));
    }
    if ((NoTextBehind && NoTextAhead) ||
        (NoTextBehind && I == Tokens.size() - 1)) {
      Token &PrevToken = Tokens[I - 1];
      StringRef PrevTokenBody = PrevToken.getTokenBody();
      PrevToken.setTokenBody(PrevTokenBody.rtrim(" \t\v\t"));
    }
  }
  return Tokens;
}

class Parser {
public:
  Parser(StringRef TemplateStr) : TemplateStr(TemplateStr) {}

  std::shared_ptr<ASTNode> parse();

private:
  void parseMustache(std::shared_ptr<ASTNode> Parent);

  std::vector<Token> Tokens;
  std::size_t CurrentPtr;
  StringRef TemplateStr;
};

std::shared_ptr<ASTNode> Parser::parse() {
  Tokens = tokenize(TemplateStr);
  CurrentPtr = 0;
  std::shared_ptr<ASTNode> Root = std::make_shared<ASTNode>();
  parseMustache(Root);
  return Root;
}

void Parser::parseMustache(std::shared_ptr<ASTNode> Parent) {

  while (CurrentPtr < Tokens.size()) {
    Token CurrentToken = Tokens[CurrentPtr];
    CurrentPtr++;
    Accessor A = CurrentToken.getAccessor();
    std::shared_ptr<ASTNode> CurrentNode;

    switch (CurrentToken.getType()) {
    case Token::Type::Text: {
      CurrentNode =
          std::make_shared<ASTNode>(CurrentToken.getTokenBody(), Parent);
      Parent->addChild(CurrentNode);
      break;
    }
    case Token::Type::Variable: {
      CurrentNode = std::make_shared<ASTNode>(ASTNode::Variable, A, Parent);
      Parent->addChild(CurrentNode);
      break;
    }
    case Token::Type::UnescapeVariable: {
      CurrentNode =
          std::make_shared<ASTNode>(ASTNode::UnescapeVariable, A, Parent);
      Parent->addChild(CurrentNode);
      break;
    }
    case Token::Type::Partial: {
      CurrentNode = std::make_shared<ASTNode>(ASTNode::Partial, A, Parent);
      Parent->addChild(CurrentNode);
      break;
    }
    case Token::Type::SectionOpen: {
      CurrentNode = std::make_shared<ASTNode>(ASTNode::Section, A, Parent);
      std::size_t Start = CurrentPtr;
      parseMustache(CurrentNode);
      std::size_t End = CurrentPtr;
      SmallString<128> RawBody;
      if (Start + 1 < End - 1)
        for (std::size_t I = Start + 1; I < End - 1; I++)
          RawBody += Tokens[I].getRawBody();
      else if (Start + 1 == End - 1)
        RawBody = Tokens[Start].getRawBody();
      CurrentNode->setRawBody(RawBody);
      Parent->addChild(CurrentNode);
      break;
    }
    case Token::Type::InvertSectionOpen: {
      CurrentNode =
          std::make_shared<ASTNode>(ASTNode::InvertSection, A, Parent);
      std::size_t Start = CurrentPtr;
      parseMustache(CurrentNode);
      std::size_t End = CurrentPtr;
      SmallString<128> RawBody;
      if (Start + 1 < End - 1)
        for (std::size_t I = Start + 1; I < End - 1; I++)
          RawBody += Tokens[I].getRawBody();
      else if (Start + 1 == End - 1)
        RawBody = Tokens[Start].getRawBody();
      CurrentNode->setRawBody(RawBody);
      Parent->addChild(CurrentNode);
      break;
    }
    case Token::Type::SectionClose:
      return;
    default:
      break;
    }
  }
}

Template Template::createTemplate(StringRef TemplateStr) {
  Parser P = Parser(TemplateStr);
  std::shared_ptr<ASTNode> MustacheTree = P.parse();
  Template T = Template(MustacheTree);
  // the default behaviour is to escape html entities
  DenseMap<char, StringRef> HtmlEntities = {{'&', "&amp;"},
                                            {'<', "&lt;"},
                                            {'>', "&gt;"},
                                            {'"', "&quot;"},
                                            {'\'', "&#39;"}};
  T.registerEscape(HtmlEntities);
  return T;
}

SmallString<128> Template::render(Value Data) {
  return Tree->render(Data, Partials, Lambdas, SectionLambdas, Escapes);
}

void Template::registerPartial(StringRef Name, StringRef Partial) {
  Parser P = Parser(Partial);
  std::shared_ptr<ASTNode> PartialTree = P.parse();
  Partials[Name] = PartialTree;
}

void Template::registerLambda(StringRef Name, Lambda L) { Lambdas[Name] = L; }

void Template::registerLambda(StringRef Name, SectionLambda L) {
  SectionLambdas[Name] = L;
}

void Template::registerEscape(DenseMap<char, StringRef> E) { Escapes = E; }

SmallString<128> printJson(Value &Data) {

  SmallString<128> Result;
  if (Data.getAsNull())
    return Result;
  if (auto *Arr = Data.getAsArray())
    if (Arr->empty())
      return Result;
  if (Data.getAsString()) {
    Result += Data.getAsString()->str();
    return Result;
  }
  return llvm::formatv("{0:2}", Data);
}

bool isFalsey(Value &V) {
  return V.getAsNull() || (V.getAsBoolean() && !V.getAsBoolean().value()) ||
         (V.getAsArray() && V.getAsArray()->empty()) ||
         (V.getAsObject() && V.getAsObject()->empty());
}

SmallString<128>
ASTNode::render(Value Data,
                DenseMap<StringRef, std::shared_ptr<ASTNode>> &Partials,
                DenseMap<StringRef, Lambda> &Lambdas,
                DenseMap<StringRef, SectionLambda> &SectionLambdas,
                DenseMap<char, StringRef> &Escapes) {
  LocalContext = Data;
  Value Context = T == Root ? Data : findContext();
  SmallString<128> Result;
  switch (T) {
  case Root: {
    for (std::shared_ptr<ASTNode> Child : Children)
      Result +=
          Child->render(Context, Partials, Lambdas, SectionLambdas, Escapes);
    return Result;
  }
  case Text:
    return Body;
  case Partial: {
    if (Partials.find(Accessor[0]) != Partials.end()) {
      std::shared_ptr<ASTNode> Partial = Partials[Accessor[0]];
      Result +=
          Partial->render(Data, Partials, Lambdas, SectionLambdas, Escapes);
      return Result;
    }
  }
  case Variable: {
    if (Lambdas.find(Accessor[0]) != Lambdas.end()) {
      Lambda &L = Lambdas[Accessor[0]];
      Value LambdaResult = L();
      StringRef LambdaStr = printJson(LambdaResult);
      Parser P = Parser(LambdaStr);
      std::shared_ptr<ASTNode> LambdaNode = P.parse();
      SmallString<128> RenderStr =
          LambdaNode->render(Data, Partials, Lambdas, SectionLambdas, Escapes);
      return escapeString(RenderStr, Escapes);
    }
    return escapeString(printJson(Context), Escapes);
  }
  case UnescapeVariable: {
    if (Lambdas.find(Accessor[0]) != Lambdas.end()) {
      Lambda &L = Lambdas[Accessor[0]];
      Value LambdaResult = L();
      StringRef LambdaStr = printJson(LambdaResult);
      Parser P = Parser(LambdaStr);
      std::shared_ptr<ASTNode> LambdaNode = P.parse();
      DenseMap<char, StringRef> EmptyEscapes;
      return LambdaNode->render(Data, Partials, Lambdas, SectionLambdas,
                                EmptyEscapes);
    }
    return printJson(Context);
  }
  case Section: {
    // Sections are not rendered if the context is falsey
    bool IsLambda = SectionLambdas.find(Accessor[0]) != SectionLambdas.end();

    if (isFalsey(Context) && !IsLambda)
      return Result;

    if (IsLambda) {
      SectionLambda &Lambda = SectionLambdas[Accessor[0]];
      Value Return = Lambda(RawBody);
      if (isFalsey(Return))
        return Result;
      StringRef LambdaStr = printJson(Return);
      Parser P = Parser(LambdaStr);
      std::shared_ptr<ASTNode> LambdaNode = P.parse();
      return LambdaNode->render(Data, Partials, Lambdas, SectionLambdas,
                                Escapes);
    }

    if (Context.getAsArray()) {
      json::Array *Arr = Context.getAsArray();
      for (Value &V : *Arr) {
        for (std::shared_ptr<ASTNode> Child : Children)
          Result +=
              Child->render(V, Partials, Lambdas, SectionLambdas, Escapes);
      }
      return Result;
    }

    for (std::shared_ptr<ASTNode> Child : Children)
      Result +=
          Child->render(Context, Partials, Lambdas, SectionLambdas, Escapes);

    return Result;
  }
  case InvertSection: {
    bool IsLambda = SectionLambdas.find(Accessor[0]) != SectionLambdas.end();
    if (!isFalsey(Context) || IsLambda)
      return Result;
    for (std::shared_ptr<ASTNode> Child : Children)
      Result +=
          Child->render(Context, Partials, Lambdas, SectionLambdas, Escapes);
    return Result;
  }
  }
  llvm_unreachable("Invalid ASTNode type");
}

Value ASTNode::findContext() {
  // The mustache spec allows for dot notation to access nested values
  // a single dot refers to the current context
  // We attempt to find the JSON context in the current node if it is not found
  // we traverse the parent nodes to find the context until we reach the root
  // node or the context is found
  if (Accessor.empty())
    return nullptr;
  if (Accessor[0] == ".")
    return LocalContext;
  json::Object *CurrentContext = LocalContext.getAsObject();
  SmallString<128> CurrentAccessor = Accessor[0];
  std::weak_ptr<ASTNode> CurrentParent = Parent;

  while (!CurrentContext || !CurrentContext->get(CurrentAccessor)) {
    if (auto Ptr = CurrentParent.lock()) {
      CurrentContext = Ptr->LocalContext.getAsObject();
      CurrentParent = Ptr->Parent;
      continue;
    }
    return nullptr;
  }
  Value Context = nullptr;
  for (auto E : enumerate(Accessor)) {
    Value *CurrentValue = CurrentContext->get(E.value());
    if (!CurrentValue)
      return nullptr;
    if (E.index() < Accessor.size() - 1) {
      CurrentContext = CurrentValue->getAsObject();
      if (!CurrentContext)
        return nullptr;
    } else
      Context = *CurrentValue;
  }
  return Context;
}
