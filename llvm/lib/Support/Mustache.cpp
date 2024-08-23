//===-- Mustache.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Mustache.h"
#include "llvm/Support/Error.h"
#include <iostream>
#include <regex>
#include <sstream>

using namespace llvm;
using namespace llvm::json;
using namespace llvm::mustache;

std::string escapeHtml(const std::string &Input) {
  DenseMap<char, std::string> HtmlEntities = {{'&', "&amp;"},
                                              {'<', "&lt;"},
                                              {'>', "&gt;"},
                                              {'"', "&quot;"},
                                              {'"', "&#39;"}};
  std::string EscapedString;
  EscapedString.reserve(Input.size());

  for (char C : Input) {
    if (HtmlEntities.find(C) != HtmlEntities.end()) {
      EscapedString += HtmlEntities[C];
    } else {
      EscapedString += C;
    }
  }

  return EscapedString;
}

std::vector<std::string> split(const std::string &Str, char Delimiter) {
  std::vector<std::string> Tokens;
  std::string Token;
  std::stringstream SS(Str);
  if (Str == ".") {
    Tokens.push_back(Str);
    return Tokens;
  }
  while (std::getline(SS, Token, Delimiter)) {
    Tokens.push_back(Token);
  }
  return Tokens;
}

Token::Token(std::string Str, char Identifier) {
  switch (Identifier) {
  case '#':
    TokenType = Type::SectionOpen;
    break;
  case '/':
    TokenType = Type::SectionClose;
    break;
  case '^':
    TokenType = Type::InvertSectionOpen;
    break;
  case '!':
    TokenType = Type::Comment;
    break;
  case '>':
    TokenType = Type::Partial;
    break;
  case '&':
    TokenType = Type::UnescapeVariable;
    break;
  default:
    TokenType = Type::Variable;
  }
  if (TokenType == Type::Comment)
    return;

  TokenBody = Str;
  std::string AccessorStr = Str;
  if (TokenType != Type::Variable) {
    AccessorStr = Str.substr(1);
  }
  Accessor = split(StringRef(AccessorStr).trim().str(), '.');
}

Token::Token(std::string Str)
    : TokenType(Type::Text), TokenBody(Str), Accessor({}) {}

std::vector<Token> tokenize(std::string Template) {
  std::vector<Token> Tokens;
  std::regex Re(R"(\{\{(.*?)\}\})");
  std::sregex_token_iterator Iter(Template.begin(), Template.end(), Re,
                                  {-1, 0});
  std::sregex_token_iterator End;

  for (; Iter != End; ++Iter) {
    if (!Iter->str().empty()) {
      std::string Token = *Iter;
      std::smatch Match;
      if (std::regex_match(Token, Match, Re)) {
        std::string Group = Match[1];
        Tokens.emplace_back(Group, Group[0]);
      } else {
        Tokens.emplace_back(Token);
      }
    }
  }

  return Tokens;
}

class Parser {
public:
  Parser(std::string TemplateStr) : TemplateStr(TemplateStr) {}

  std::shared_ptr<ASTNode> parse();

private:
  void parseMustache(std::shared_ptr<ASTNode> Parent);

  std::vector<Token> Tokens;
  std::size_t CurrentPtr;
  std::string TemplateStr;
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
      parseMustache(CurrentNode);
      Parent->addChild(CurrentNode);
      break;
    }
    case Token::Type::InvertSectionOpen: {
      CurrentNode =
          std::make_shared<ASTNode>(ASTNode::InvertSection, A, Parent);
      parseMustache(CurrentNode);
      Parent->addChild(CurrentNode);
      break;
    }
    case Token::Type::SectionClose: {
      return;
    }
    default:
      break;
    }
  }
}

Expected<Template> Template::createTemplate(std::string TemplateStr) {
  Parser P = Parser(TemplateStr);
  Expected<std::shared_ptr<ASTNode>> MustacheTree = P.parse();
  if (!MustacheTree)
    return MustacheTree.takeError();
  return Template(MustacheTree.get());
}
std::string Template::render(Value Data) { return Tree->render(Data); }

std::string printJson(Value &Data) {
  if (Data.getAsNull().has_value()) {
    return "";
  }
  if (auto *Arr = Data.getAsArray()) {
    if (Arr->empty()) {
      return "";
    }
  }
  if (Data.getAsString().has_value()) {
    return Data.getAsString()->str();
  }
  return llvm::formatv("{0:2}", Data);
}

std::string ASTNode::render(Value Data) {
  LocalContext = Data;
  Value Context = T == Root ? Data : findContext();
  switch (T) {
  case Root: {
    std::string Result = "";
    for (std::shared_ptr<ASTNode> Child : Children) {
      Result += Child->render(Context);
    }
    return Result;
  }
  case Text:
    return escapeHtml(Body);
  case Partial:
    break;
  case Variable:
    return escapeHtml(printJson(Context));
  case UnescapeVariable:
    return printJson(Context);
  case Section:
    break;
  case InvertSection:
    break;
  }

  return std::string();
}

Value ASTNode::findContext() {
  if (Accessor.empty()) {
    return nullptr;
  }
  if (Accessor[0] == ".") {
    return LocalContext;
  }
  json::Object *CurrentContext = LocalContext.getAsObject();
  std::string &CurrentAccessor = Accessor[0];
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
  for (std::size_t i = 0; i < Accessor.size(); i++) {
    CurrentAccessor = Accessor[i];
    Value *CurrentValue = CurrentContext->get(CurrentAccessor);
    if (!CurrentValue) {
      return nullptr;
    }
    if (i < Accessor.size() - 1) {
      CurrentContext = CurrentValue->getAsObject();
      if (!CurrentContext) {
        return nullptr;
      }
    } else {
      Context = *CurrentValue;
    }
  }
  return Context;
}
