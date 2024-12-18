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
// See the Mustache spec for more information
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
// The Template class is container class outputs the Mustache template string
// and is main class for users. It stores all the lambdas and the ASTNode Tree.
// When the Template is instantiated it tokenize the Template String and
// creates a vector of Tokens. Then it calls a basic recursive descent parser
// to construct the ASTNode Tree. The ASTNodes are all stored in an arena
// allocator which is freed once the template class goes out of scope
//
// Usage:
// \code
//   // Creating a simple template and rendering it
//   auto Template = Template("Hello, {{name}}!");
//   Value Data = {{"name", "World"}};
//   StringRef Rendered = Template.render(Data);
//   // Rendered == "Hello, World!"
//
//   // Creating a template with a partial and rendering it
//   auto Template = Template("{{>partial}}");
//   Template.registerPartial("partial", "Hello, {{name}}!");
//   Value Data = {{"name", "World"}};
//   StringRef Rendered = Template.render(Data);
//   // Rendered == "Hello, World!"
//
//   // Creating a template with a lambda and rendering it
//   auto Template = Template("{{#lambda}}Hello, {{name}}!{{/lambda}}");
//   Template.registerLambda("lambda", []() { return true; });
//   Value Data = {{"name", "World"}};
//   StringRef Rendered = Template.render(Data);
//   // Rendered == "Hello, World!"
// \endcode
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_MUSTACHE
#define LLVM_SUPPORT_MUSTACHE

#include "Error.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/StringSaver.h"
#include <vector>

namespace llvm {
namespace mustache {

using Accessor = SmallVector<std::string>;
using Lambda = std::function<llvm::json::Value()>;
using SectionLambda = std::function<llvm::json::Value(std::string)>;

class ASTNode;

// A Template represents the container for the AST and the partials
// and Lambdas that are registered with it.
class Template {
public:
  Template(std::string TemplateStr);

  void render(llvm::json::Value &Data, llvm::raw_ostream &OS);

  void registerPartial(std::string Name, std::string Partial);

  void registerLambda(std::string Name, Lambda Lambda);

  void registerLambda(std::string Name, SectionLambda Lambda);

  // By default the Mustache Spec Specifies that HTML special characters
  // should be escaped. This function allows the user to specify which
  // characters should be escaped.
  void registerEscape(DenseMap<char, std::string> Escapes);

private:
  std::string Output;
  StringMap<ASTNode *> Partials;
  StringMap<Lambda> Lambdas;
  StringMap<SectionLambda> SectionLambdas;
  DenseMap<char, std::string> Escapes;
  // The allocator for the ASTNode Tree
  llvm::BumpPtrAllocator AstAllocator;
  // Allocator for each render call resets after each render
  llvm::BumpPtrAllocator RenderAllocator;
  ASTNode *Tree;
};
} // namespace mustache
} // end namespace llvm

#endif // LLVM_SUPPORT_MUSTACHE
