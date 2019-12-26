//===-- AppleObjCTypeEncodingParser.cpp -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AppleObjCTypeEncodingParser.h"

#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangUtil.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/StringLexer.h"

#include <vector>

using namespace lldb_private;

AppleObjCTypeEncodingParser::AppleObjCTypeEncodingParser(
    ObjCLanguageRuntime &runtime)
    : ObjCLanguageRuntime::EncodingToType(), m_runtime(runtime) {
  if (!m_scratch_ast_ctx_up)
    m_scratch_ast_ctx_up.reset(new ClangASTContext(runtime.GetProcess()
                                                       ->GetTarget()
                                                       .GetArchitecture()
                                                       .GetTriple()
                                                       .str()
                                                       .c_str()));
}

std::string AppleObjCTypeEncodingParser::ReadStructName(StringLexer &type) {
  StreamString buffer;
  while (type.HasAtLeast(1) && type.Peek() != '=')
    buffer.Printf("%c", type.Next());
  return buffer.GetString();
}

std::string AppleObjCTypeEncodingParser::ReadQuotedString(StringLexer &type) {
  StreamString buffer;
  while (type.HasAtLeast(1) && type.Peek() != '"')
    buffer.Printf("%c", type.Next());
  StringLexer::Character next = type.Next();
  UNUSED_IF_ASSERT_DISABLED(next);
  assert(next == '"');
  return buffer.GetString();
}

uint32_t AppleObjCTypeEncodingParser::ReadNumber(StringLexer &type) {
  uint32_t total = 0;
  while (type.HasAtLeast(1) && isdigit(type.Peek()))
    total = 10 * total + (type.Next() - '0');
  return total;
}

// as an extension to the published grammar recent runtimes emit structs like
// this:
// "{CGRect=\"origin\"{CGPoint=\"x\"d\"y\"d}\"size\"{CGSize=\"width\"d\"height\"d}}"

AppleObjCTypeEncodingParser::StructElement::StructElement()
    : name(""), type(clang::QualType()), bitfield(0) {}

AppleObjCTypeEncodingParser::StructElement
AppleObjCTypeEncodingParser::ReadStructElement(ClangASTContext &ast_ctx,
                                               StringLexer &type,
                                               bool for_expression) {
  StructElement retval;
  if (type.NextIf('"'))
    retval.name = ReadQuotedString(type);
  if (!type.NextIf('"'))
    return retval;
  uint32_t bitfield_size = 0;
  retval.type = BuildType(ast_ctx, type, for_expression, &bitfield_size);
  retval.bitfield = bitfield_size;
  return retval;
}

clang::QualType AppleObjCTypeEncodingParser::BuildStruct(
    ClangASTContext &ast_ctx, StringLexer &type, bool for_expression) {
  return BuildAggregate(ast_ctx, type, for_expression, '{', '}',
                        clang::TTK_Struct);
}

clang::QualType AppleObjCTypeEncodingParser::BuildUnion(
    ClangASTContext &ast_ctx, StringLexer &type, bool for_expression) {
  return BuildAggregate(ast_ctx, type, for_expression, '(', ')',
                        clang::TTK_Union);
}

clang::QualType AppleObjCTypeEncodingParser::BuildAggregate(
    ClangASTContext &ast_ctx, StringLexer &type, bool for_expression,
    char opener, char closer, uint32_t kind) {
  if (!type.NextIf(opener))
    return clang::QualType();
  std::string name(ReadStructName(type));

  // We do not handle templated classes/structs at the moment. If the name has
  // a < in it, we are going to abandon this. We're still obliged to parse it,
  // so we just set a flag that means "Don't actually build anything."

  const bool is_templated = name.find('<') != std::string::npos;

  if (!type.NextIf('='))
    return clang::QualType();
  bool in_union = true;
  std::vector<StructElement> elements;
  while (in_union && type.HasAtLeast(1)) {
    if (type.NextIf(closer)) {
      in_union = false;
      break;
    } else {
      auto element = ReadStructElement(ast_ctx, type, for_expression);
      if (element.type.isNull())
        break;
      else
        elements.push_back(element);
    }
  }
  if (in_union)
    return clang::QualType();

  if (is_templated)
    return clang::QualType(); // This is where we bail out.  Sorry!

  CompilerType union_type(ast_ctx.CreateRecordType(
      nullptr, lldb::eAccessPublic, name, kind, lldb::eLanguageTypeC));
  if (union_type) {
    ClangASTContext::StartTagDeclarationDefinition(union_type);

    unsigned int count = 0;
    for (auto element : elements) {
      if (element.name.empty()) {
        StreamString elem_name;
        elem_name.Printf("__unnamed_%u", count);
        element.name = elem_name.GetString();
      }
      ClangASTContext::AddFieldToRecordType(
          union_type, element.name.c_str(),
          CompilerType(&ast_ctx, element.type.getAsOpaquePtr()),
          lldb::eAccessPublic, element.bitfield);
      ++count;
    }
    ClangASTContext::CompleteTagDeclarationDefinition(union_type);
  }
  return ClangUtil::GetQualType(union_type);
}

clang::QualType AppleObjCTypeEncodingParser::BuildArray(
    ClangASTContext &ast_ctx, StringLexer &type, bool for_expression) {
  if (!type.NextIf('['))
    return clang::QualType();
  uint32_t size = ReadNumber(type);
  clang::QualType element_type(BuildType(ast_ctx, type, for_expression));
  if (!type.NextIf(']'))
    return clang::QualType();
  CompilerType array_type(ast_ctx.CreateArrayType(
      CompilerType(&ast_ctx, element_type.getAsOpaquePtr()), size, false));
  return ClangUtil::GetQualType(array_type);
}

// the runtime can emit these in the form of @"SomeType", giving more specifics
// this would be interesting for expression parser interop, but since we
// actually try to avoid exposing the ivar info to the expression evaluator,
// consume but ignore the type info and always return an 'id'; if anything,
// dynamic typing will resolve things for us anyway
clang::QualType AppleObjCTypeEncodingParser::BuildObjCObjectPointerType(
    ClangASTContext &clang_ast_ctx, StringLexer &type, bool for_expression) {
  if (!type.NextIf('@'))
    return clang::QualType();

  clang::ASTContext &ast_ctx = clang_ast_ctx.getASTContext();

  std::string name;

  if (type.NextIf('"')) {
    // We have to be careful here.  We're used to seeing
    //   @"NSString"
    // but in records it is possible that the string following an @ is the name
    // of the next field and @ means "id". This is the case if anything
    // unquoted except for "}", the end of the type, or another name follows
    // the quoted string.
    //
    // E.g.
    // - @"NSString"@ means "id, followed by a field named NSString of type id"
    // - @"NSString"} means "a pointer to NSString and the end of the struct" -
    // @"NSString""nextField" means "a pointer to NSString and a field named
    // nextField" - @"NSString" followed by the end of the string means "a
    // pointer to NSString"
    //
    // As a result, the rule is: If we see @ followed by a quoted string, we
    // peek. - If we see }, ), ], the end of the string, or a quote ("), the
    // quoted string is a class name. - If we see anything else, the quoted
    // string is a field name and we push it back onto type.

    name = ReadQuotedString(type);

    if (type.HasAtLeast(1)) {
      switch (type.Peek()) {
      default:
        // roll back
        type.PutBack(name.length() +
                     2); // undo our consumption of the string and of the quotes
        name.clear();
        break;
      case '}':
      case ')':
      case ']':
      case '"':
        // the quoted string is a class name – see the rule
        break;
      }
    } else {
      // the quoted string is a class name – see the rule
    }
  }

  if (for_expression && !name.empty()) {
    size_t less_than_pos = name.find('<');

    if (less_than_pos != std::string::npos) {
      if (less_than_pos == 0)
        return ast_ctx.getObjCIdType();
      else
        name.erase(less_than_pos);
    }

    DeclVendor *decl_vendor = m_runtime.GetDeclVendor();
    if (!decl_vendor)
      return clang::QualType();

    auto types = decl_vendor->FindTypes(ConstString(name), /*max_matches*/ 1);

// The user can forward-declare something that has no definition.  The runtime
// doesn't prohibit this at all. This is a rare and very weird case.  We keep
// this assert in debug builds so we catch other weird cases.
#ifdef LLDB_CONFIGURATION_DEBUG
    assert(!types.empty());
#else
    if (types.empty())
      return ast_ctx.getObjCIdType();
#endif

    return ClangUtil::GetQualType(types.front().GetPointerType());
  } else {
    // We're going to resolve this dynamically anyway, so just smile and wave.
    return ast_ctx.getObjCIdType();
  }
}

clang::QualType
AppleObjCTypeEncodingParser::BuildType(ClangASTContext &clang_ast_ctx,
                                       StringLexer &type, bool for_expression,
                                       uint32_t *bitfield_bit_size) {
  if (!type.HasAtLeast(1))
    return clang::QualType();

  clang::ASTContext &ast_ctx = clang_ast_ctx.getASTContext();

  switch (type.Peek()) {
  default:
    break;
  case '{':
    return BuildStruct(clang_ast_ctx, type, for_expression);
  case '[':
    return BuildArray(clang_ast_ctx, type, for_expression);
  case '(':
    return BuildUnion(clang_ast_ctx, type, for_expression);
  case '@':
    return BuildObjCObjectPointerType(clang_ast_ctx, type, for_expression);
  }

  switch (type.Next()) {
  default:
    type.PutBack(1);
    return clang::QualType();
  case 'c':
    return ast_ctx.CharTy;
  case 'i':
    return ast_ctx.IntTy;
  case 's':
    return ast_ctx.ShortTy;
  case 'l':
    return ast_ctx.getIntTypeForBitwidth(32, true);
  // this used to be done like this:
  //   return clang_ast_ctx->GetIntTypeFromBitSize(32, true).GetQualType();
  // which uses one of the constants if one is available, but we don't think
  // all this work is necessary.
  case 'q':
    return ast_ctx.LongLongTy;
  case 'C':
    return ast_ctx.UnsignedCharTy;
  case 'I':
    return ast_ctx.UnsignedIntTy;
  case 'S':
    return ast_ctx.UnsignedShortTy;
  case 'L':
    return ast_ctx.getIntTypeForBitwidth(32, false);
  // see note for 'l'
  case 'Q':
    return ast_ctx.UnsignedLongLongTy;
  case 'f':
    return ast_ctx.FloatTy;
  case 'd':
    return ast_ctx.DoubleTy;
  case 'B':
    return ast_ctx.BoolTy;
  case 'v':
    return ast_ctx.VoidTy;
  case '*':
    return ast_ctx.getPointerType(ast_ctx.CharTy);
  case '#':
    return ast_ctx.getObjCClassType();
  case ':':
    return ast_ctx.getObjCSelType();
  case 'b': {
    uint32_t size = ReadNumber(type);
    if (bitfield_bit_size) {
      *bitfield_bit_size = size;
      return ast_ctx.UnsignedIntTy; // FIXME: the spec is fairly vague here.
    } else
      return clang::QualType();
  }
  case 'r': {
    clang::QualType target_type =
        BuildType(clang_ast_ctx, type, for_expression);
    if (target_type.isNull())
      return clang::QualType();
    else if (target_type == ast_ctx.UnknownAnyTy)
      return ast_ctx.UnknownAnyTy;
    else
      return ast_ctx.getConstType(target_type);
  }
  case '^': {
    if (!for_expression && type.NextIf('?')) {
      // if we are not supporting the concept of unknownAny, but what is being
      // created here is an unknownAny*, then we can just get away with a void*
      // this is theoretically wrong (in the same sense as 'theoretically
      // nothing exists') but is way better than outright failure in many
      // practical cases
      return ast_ctx.VoidPtrTy;
    } else {
      clang::QualType target_type =
          BuildType(clang_ast_ctx, type, for_expression);
      if (target_type.isNull())
        return clang::QualType();
      else if (target_type == ast_ctx.UnknownAnyTy)
        return ast_ctx.UnknownAnyTy;
      else
        return ast_ctx.getPointerType(target_type);
    }
  }
  case '?':
    return for_expression ? ast_ctx.UnknownAnyTy : clang::QualType();
  }
}

CompilerType AppleObjCTypeEncodingParser::RealizeType(ClangASTContext &ast_ctx,
                                                      const char *name,
                                                      bool for_expression) {
  if (name && name[0]) {
    StringLexer lexer(name);
    clang::QualType qual_type = BuildType(ast_ctx, lexer, for_expression);
    return CompilerType(&ast_ctx, qual_type.getAsOpaquePtr());
  }
  return CompilerType();
}
