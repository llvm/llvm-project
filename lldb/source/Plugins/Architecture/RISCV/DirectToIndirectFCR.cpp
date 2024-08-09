//===--- DirectToIndirectFCR.cpp - RISC-V specific pass -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "Plugins/Architecture/RISCV/DirectToIndirectFCR.h"

#include "lldb/Core/Architecture.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/Symtab.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

#include <optional>

using namespace llvm;
using namespace lldb_private;

namespace {
std::string GetValueTypeStr(const llvm::Type *value_ty) {
  assert(value_ty);
  std::string str_type;
  llvm::raw_string_ostream rso(str_type);
  value_ty->print(rso);
  return rso.str();
}

template <typename... Args> void LogMessage(const char *msg, Args &&...args) {
  Log *log = GetLog(LLDBLog::Expressions);
  LLDB_LOG(log, msg, std::forward<Args>(args)...);
}
} // namespace

bool DirectToIndirectFCR::canBeReplaced(const llvm::CallInst *ci) {
  assert(ci);
  auto *return_value_ty = ci->getType();
  if (!(return_value_ty->isIntegerTy() || return_value_ty->isVoidTy())) {
    LogMessage("DirectToIndirectFCR: function {0} has unsupported "
               "return type ({1})\n",
               ci->getCalledFunction()->getName(),
               GetValueTypeStr(return_value_ty));
    return false;
  }

  const auto *arg = llvm::find_if_not(ci->args(), [](const auto &arg) {
    const auto *type = arg->getType();
    return type->isIntegerTy();
  });

  if (arg != ci->arg_end()) {
    LogMessage("DirectToIndirectFCR: argument {0} of {1} function "
               "has unsupported type ({2})\n",
               (*arg)->getName(), ci->getCalledFunction()->getName(),
               GetValueTypeStr((*arg)->getType()));
    return false;
  }
  return true;
}

std::vector<llvm::Value *>
DirectToIndirectFCR::getFunctionArgsAsValues(const llvm::CallInst *ci) {
  assert(ci);
  std::vector<llvm::Value *> args{};
  llvm::transform(ci->args(), std::back_inserter(args),
                  [](const auto &arg) { return arg.get(); });
  return args;
}

std::optional<lldb::addr_t>
DirectToIndirectFCR::getFunctionAddress(const llvm::CallInst *ci) const {
  auto *target = m_exe_ctx.GetTargetPtr();
  const auto &lldb_module_sp = target->GetExecutableModule();
  const auto &symtab = lldb_module_sp->GetSymtab();
  const llvm::StringRef name = ci->getCalledFunction()->getName();

  // eSymbolTypeCode: we try to find function
  // eDebugNo: not a debug symbol
  // eVisibilityExtern: function from extern module
  const auto *symbol = symtab->FindFirstSymbolWithNameAndType(
      ConstString(name), lldb::SymbolType::eSymbolTypeCode,
      Symtab::Debug::eDebugNo, Symtab::Visibility::eVisibilityExtern);
  if (!symbol) {
    LogMessage("DirectToIndirectFCR: can't find {0} in symtab\n", name);
    return std::nullopt;
  }

  lldb::addr_t addr = symbol->GetLoadAddress(target);
  LogMessage("DirectToIndirectFCR: found address ({0}) of symbol {1}\n", addr,
             name);
  return addr;
}

llvm::CallInst *DirectToIndirectFCR::getInstReplace(llvm::CallInst *ci) const {
  assert(ci);

  std::optional<lldb::addr_t> addr_or_null = getFunctionAddress(ci);
  if (!addr_or_null.has_value())
    return nullptr;

  lldb::addr_t addr = addr_or_null.value();

  llvm::IRBuilder<> builder(ci);

  std::vector<llvm::Value *> args = getFunctionArgsAsValues(ci);
  llvm::Constant *func_addr = builder.getInt64(addr);
  llvm::PointerType *ptr_func_ty = builder.getPtrTy();
  auto *cast = builder.CreateIntToPtr(func_addr, ptr_func_ty);
  auto *new_inst =
      builder.CreateCall(ci->getFunctionType(), cast, ArrayRef(args));
  return new_inst;
}

DirectToIndirectFCR::DirectToIndirectFCR(const ExecutionContext &exe_ctx)
    : FunctionPass(ID), m_exe_ctx{exe_ctx} {}

DirectToIndirectFCR::~DirectToIndirectFCR() = default;

bool DirectToIndirectFCR::runOnFunction(llvm::Function &func) {
  bool has_irreplaceable =
      llvm::any_of(instructions(func), [this](llvm::Instruction &inst) {
        llvm::CallInst *ci = dyn_cast<llvm::CallInst>(&inst);
        if (!ci || ci->getCalledFunction()->isIntrinsic() ||
            (DirectToIndirectFCR::canBeReplaced(ci) &&
             getFunctionAddress(ci).has_value()))
          return false;
        return true;
      });

  if (has_irreplaceable) {
    func.getParent()->getOrInsertNamedMetadata(
        Architecture::s_target_incompatibility_marker);
    return false;
  }

  std::vector<std::reference_wrapper<llvm::Instruction>>
      replaceable_function_calls{};
  llvm::copy_if(instructions(func),
                std::back_inserter(replaceable_function_calls),
                [](llvm::Instruction &inst) {
                  llvm::CallInst *ci = dyn_cast<llvm::CallInst>(&inst);
                  if (ci && !ci->getCalledFunction()->isIntrinsic())
                    return true;
                  return false;
                });

  if (replaceable_function_calls.empty())
    return false;

  std::vector<std::pair<llvm::CallInst *, llvm::CallInst *>> replaces;
  llvm::transform(replaceable_function_calls, std::back_inserter(replaces),
                  [this](std::reference_wrapper<llvm::Instruction> inst)
                      -> std::pair<llvm::CallInst *, llvm::CallInst *> {
                    llvm::CallInst *ci = cast<llvm::CallInst>(&(inst.get()));
                    llvm::CallInst *new_inst = getInstReplace(ci);
                    return {ci, new_inst};
                  });

  for (auto &&[from, to] : replaces) {
    from->replaceAllUsesWith(to);
    from->eraseFromParent();
  }

  return true;
}

llvm::StringRef DirectToIndirectFCR::getPassName() const {
  return "Transform function calls to calls by address";
}

char DirectToIndirectFCR::ID = 0;

llvm::FunctionPass *
lldb_private::createDirectToIndirectFCR(const ExecutionContext &exe_ctx) {
  return new DirectToIndirectFCR(exe_ctx);
}
