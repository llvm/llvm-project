//===--- Mustache.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the Mustache templating language supports version 1.4.2
// (https://mustache.github.io/mustache.5.html).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_MUSTACHE
#define LLVM_SUPPORT_MUSTACHE

#include "Error.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include <string>
#include <variant>
#include <vector>

namespace llvm {
namespace mustache {

using Accessor = std::vector<std::string>;

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

  Token(std::string Str);

  Token(std::string Str, char Identifier);

  std::string getTokenBody() const { return TokenBody; };

  Accessor getAccessor() const { return Accessor; };

  Type getType() const { return TokenType; };

private:
  Type TokenType;
  Accessor Accessor;
  std::string TokenBody;
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

  ASTNode() : T(Type::Root), LocalContext(nullptr){};

  ASTNode(std::string Body, std::shared_ptr<ASTNode> Parent)
      : T(Type::Text), Body(Body), Parent(Parent), LocalContext(nullptr){};

  // Constructor for Section/InvertSection/Variable/UnescapeVariable
  ASTNode(Type T, Accessor Accessor, std::shared_ptr<ASTNode> Parent)
      : T(T), Accessor(Accessor), Parent(Parent), LocalContext(nullptr),
        Children({}){};

  void addChild(std::shared_ptr<ASTNode> Child) {
    Children.emplace_back(Child);
  };

  std::string render(llvm::json::Value Data);

  llvm::json::Value findContext();

  Type T;
  std::string Body;
  std::weak_ptr<ASTNode> Parent;
  std::vector<std::shared_ptr<ASTNode>> Children;
  Accessor Accessor;
  llvm::json::Value LocalContext;
};

class Template {
public:
  static Expected<Template> createTemplate(std::string TemplateStr);

  std::string render(llvm::json::Value Data);

private:
  Template(std::shared_ptr<ASTNode> Tree) : Tree(Tree){};
  std::shared_ptr<ASTNode> Tree;
};

} // namespace mustache
} // end namespace llvm
#endif // LLVM_SUPPORT_MUSTACHE