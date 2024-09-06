//===--- Mustache.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the Mustache templating language supports version 1.4.2
// currently relies on llvm::json::Value for data input
// see the Mustache spec for more information
// (https://mustache.github.io/mustache.5.html).
//
// Current Features Supported:
// - Variables
// - Sections
// - Inverted Sections
// - Partials
// - Comments
// - Lambdas
// - Unescaped Variables
//
// Features Not Supported:
// - Set Delimiter
// - Blocks
// - Parents
// - Dynamic Names
//
// Usage:
// - Creating a simple template and rendering it:
// \code
//   auto Template = Template::createTemplate("Hello, {{name}}!");
//   Value Data = {{"name", "World"}};
//   StringRef Rendered = Template.render(Data);
//   // Rendered == "Hello, World!"
// \endcode
// - Creating a template with a partial and rendering it:
// \code
//   auto Template = Template::createTemplate("{{>partial}}");
//   Template.registerPartial("partial", "Hello, {{name}}!");
//   Value Data = {{"name", "World"}};
//   StringRef Rendered = Template.render(Data);
//   // Rendered == "Hello, World!"
// \endcode
// - Creating a template with a lambda and rendering it:
// \code
//   auto Template = Template::createTemplate("{{#lambda}}Hello,
//                                             {{name}}!{{/lambda}}");
//   Template.registerLambda("lambda", []() { return true; });
//   Value Data = {{"name", "World"}};
//   StringRef Rendered = Template.render(Data);
//   // Rendered == "Hello, World!"
// \endcode
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_MUSTACHE
#define LLVM_SUPPORT_MUSTACHE

#include "Error.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include <vector>

namespace llvm {
namespace mustache {

using Accessor = std::vector<SmallString<128>>;
using Lambda = std::function<llvm::json::Value()>;
using SectionLambda = std::function<llvm::json::Value(StringRef)>;

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

  void setTokenBody(SmallString<128> NewBody) { TokenBody = NewBody; };

  Accessor getAccessor() const { return Accessor; };

  Type getType() const { return TokenType; };

  static Type getTokenType(char Identifier);

private:
  Type TokenType;
  SmallString<128> RawBody;
  Accessor Accessor;
  SmallString<128> TokenBody;
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

  ASTNode(StringRef Body, std::shared_ptr<ASTNode> Parent)
      : T(Type::Text), Body(Body), Parent(Parent), LocalContext(nullptr){};

  // Constructor for Section/InvertSection/Variable/UnescapeVariable
  ASTNode(Type T, Accessor Accessor, std::shared_ptr<ASTNode> Parent)
      : T(T), Accessor(Accessor), Parent(Parent), LocalContext(nullptr),
        Children({}){};

  void addChild(std::shared_ptr<ASTNode> Child) {
    Children.emplace_back(Child);
  };

  SmallString<128> getBody() const { return Body; };

  void setBody(StringRef NewBody) { Body = NewBody; };

  void setRawBody(StringRef NewBody) { RawBody = NewBody; };

  std::shared_ptr<ASTNode> getLastChild() const {
    return Children.empty() ? nullptr : Children.back();
  };

  SmallString<128>
  render(llvm::json::Value Data,
         DenseMap<StringRef, std::shared_ptr<ASTNode>> &Partials,
         DenseMap<StringRef, Lambda> &Lambdas,
         DenseMap<StringRef, SectionLambda> &SectionLambdas,
         DenseMap<char, StringRef> &Escapes);

private:
  llvm::json::Value findContext();
  Type T;
  SmallString<128> RawBody;
  SmallString<128> Body;
  std::weak_ptr<ASTNode> Parent;
  std::vector<std::shared_ptr<ASTNode>> Children;
  const Accessor Accessor;
  llvm::json::Value LocalContext;
};

// A Template represents the container for the AST and the partials
// and Lambdas that are registered with it.
class Template {
public:
  static Template createTemplate(StringRef TemplateStr);

  SmallString<128> render(llvm::json::Value Data);

  void registerPartial(StringRef Name, StringRef Partial);

  void registerLambda(StringRef Name, Lambda Lambda);

  void registerLambda(StringRef Name, SectionLambda Lambda);

  // By default the Mustache Spec Specifies that HTML special characters
  // should be escaped. This function allows the user to specify which
  // characters should be escaped.
  void registerEscape(DenseMap<char, StringRef> Escapes);

private:
  Template(std::shared_ptr<ASTNode> Tree) : Tree(Tree){};
  DenseMap<StringRef, std::shared_ptr<ASTNode>> Partials;
  DenseMap<StringRef, Lambda> Lambdas;
  DenseMap<StringRef, SectionLambda> SectionLambdas;
  DenseMap<char, StringRef> Escapes;
  std::shared_ptr<ASTNode> Tree;
};

} // namespace mustache
} // end namespace llvm
#endif // LLVM_SUPPORT_MUSTACHE