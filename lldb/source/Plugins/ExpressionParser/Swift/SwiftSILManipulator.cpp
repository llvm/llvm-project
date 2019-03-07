//===-- SwiftSILManipulator.cpp ---------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "SwiftSILManipulator.h"

#include "SwiftASTManipulator.h"

#include "lldb/Symbol/CompilerType.h"
#include "lldb/Utility/Log.h"

#include "swift/SIL/SILArgument.h"
#include "swift/SIL/SILBasicBlock.h"
#include "swift/SIL/SILBuilder.h"
#include "swift/SIL/SILFunction.h"
#include "swift/SIL/SILLocation.h"
#include "swift/SIL/SILModule.h"
#include "swift/SIL/SILType.h"
#include "swift/SIL/SILValue.h"
#include "swift/SIL/TypeLowering.h"

using namespace lldb_private;

SwiftSILManipulator::SwiftSILManipulator(swift::SILBuilder &builder)
    : m_builder(builder),
      m_log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS)) {}

swift::SILValue SwiftSILManipulator::emitLValueForVariable(
    swift::VarDecl *var, SwiftExpressionParser::SILVariableInfo &info) {
  swift::SILFunction &function = m_builder.getFunction();

  swift::SILBasicBlock &entry_block = *function.getBlocks().begin();

  swift::SILArgument *struct_argument = nullptr;

  for (swift::SILArgument *argument : entry_block.getArguments()) {
    swift::Identifier argument_name = argument->getDecl()->getBaseName()
                                        .getIdentifier();

    if (!strcmp(argument_name.get(), SwiftASTManipulator::GetArgumentName())) {
      struct_argument = argument;
      break;
    }
  }

  if (!struct_argument)
    return swift::SILValue();

  assert(struct_argument->getType().getAsString().find(
             "UnsafeMutablePointer") != std::string::npos);

  swift::CanType unsafe_mutable_pointer_can_type =
      struct_argument->getType().getSwiftRValueType();

  swift::BoundGenericStructType *unsafe_mutable_pointer_struct_type =
      llvm::cast<swift::BoundGenericStructType>(
          unsafe_mutable_pointer_can_type.getPointer());
  swift::StructDecl *unsafe_mutable_pointer_struct_decl =
      unsafe_mutable_pointer_struct_type->getDecl();

  swift::VarDecl *value_member_decl = nullptr;

  for (swift::Decl *member : unsafe_mutable_pointer_struct_decl->getMembers()) {
    if (swift::VarDecl *member_var = llvm::dyn_cast<swift::VarDecl>(member)) {
      if (member_var->getName().str().equals("_rawValue")) {
        value_member_decl = member_var;
        break;
      }
    }
  }

  if (!value_member_decl)
    return swift::SILValue();

  swift::ASTContext &ast_ctx = m_builder.getASTContext();
  swift::Lowering::TypeConverter converter(m_builder.getModule());

  swift::SILLocation null_loc((swift::Decl *)nullptr);
  swift::SILType raw_pointer_type = swift::SILType::getRawPointerType(ast_ctx);

  swift::StructExtractInst *struct_extract = m_builder.createStructExtract(
      null_loc, struct_argument, value_member_decl, raw_pointer_type);
  swift::IntegerLiteralInst *integer_literal = m_builder.createIntegerLiteral(
      null_loc, swift::SILType::getBuiltinIntegerType(64, ast_ctx),
      (intmax_t)info.offset);
  swift::IndexRawPointerInst *index_raw_pointer =
      m_builder.createIndexRawPointer(null_loc, struct_extract,
                                      integer_literal);
  swift::PointerToAddressInst *pointer_to_return_slot =
      m_builder.createPointerToAddress(null_loc, index_raw_pointer,
                                       raw_pointer_type.getAddressType(),
                                       /*isStrict*/ true);
  swift::LoadInst *pointer_to_variable =
      m_builder.createLoad(null_loc, pointer_to_return_slot,
                           swift::LoadOwnershipQualifier::Trivial);
  auto type = var->getDeclContext()->mapTypeIntoContext(
      var->getInterfaceType());
  swift::PointerToAddressInst *address_of_variable =
      m_builder.createPointerToAddress(
          null_loc, pointer_to_variable,
          converter.getLoweredType(type, swift::ResilienceExpansion::Minimal).getAddressType(),
          /*isStrict*/ true);

  if (info.needs_init) {
    info.needs_init = false;
    return swift::SILValue(m_builder.createMarkUninitialized(
        null_loc, address_of_variable, swift::MarkUninitializedInst::Var));
  } else
    return swift::SILValue(address_of_variable);
}
