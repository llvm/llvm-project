//===- Pass manager wrappers for building CodeGen pass pipeline -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Passes/CodeGenPassManager.h"
#include <algorithm>

using namespace llvm;

namespace {
// helper type for the visitor
template <class... Ts> struct overloads : Ts... {
  using Ts::operator()...;
};
template <class... Ts> overloads(Ts...) -> overloads<Ts...>;
} // namespace

LoopPassManager CodeGenLoopPassManager::getLoopPassManager() {
  CodeGenLoopPassManager LPM;
  const auto Visitor = overloads{
      [&LPM](LoopPassConceptT::unique_ptr &P) {
        LPM.LoopPasses.push_back(std::move(P));
        LPM.IsLoopNestPass.push_back(false);
      },
      [&LPM](LoopNestPassConceptT::unique_ptr &P) {
        LPM.LoopNestPasses.push_back(std::move(P));
        LPM.IsLoopNestPass.push_back(true);
      },
  };
  for (auto &P : PassList)
    std::visit(Visitor, P);
  PassList.clear();
  return LPM;
}

FunctionPassManager CodeGenFunctionPassManager::getFunctionPassManager() {
  CodeGenFunctionPassManager FPM;
  const auto Visitor = overloads{
      [&FPM](PassConceptT::unique_ptr &P) {
        FPM.Passes.push_back(std::move(P));
      },
      [&FPM](CodeGenLoopPassManager &CGLPM) {
        FPM.FunctionPassManager::addPass(createFunctionToLoopPassAdaptor(
            CGLPM.getLoopPassManager(), CGLPM.UseMemorySSA));
      },
      [&FPM](CodeGenMachineFunctionPassManager &CGMFPM) {
        FPM.FunctionPassManager::addPass(
            createFunctionToMachineFunctionPassAdaptor(
                CGMFPM.getMachineFunctionPassManager()));
      }};
  for (auto &P : PassList)
    std::visit(Visitor, P);
  PassList.clear();
  return FPM;
}

ModulePassManager CodeGenModulePassManager::getModulePassManager() {
  CodeGenModulePassManager MPM;
  const auto Visitor = overloads{
      [&MPM](PassConceptT::unique_ptr &P) {
        MPM.Passes.push_back(std::move(P));
      },
      [&MPM](CodeGenFunctionPassManager &CGFPM) {
        if (CGFPM.RequireCGSCCOrder) {
          MPM.ModulePassManager::addPass(
              createModuleToPostOrderCGSCCPassAdaptor(
                  createCGSCCToFunctionPassAdaptor(
                      CGFPM.getFunctionPassManager())));
        } else {
          MPM.ModulePassManager::addPass(createModuleToFunctionPassAdaptor(
              CGFPM.getFunctionPassManager(), CGFPM.EagerlyInvalidate));
        }
      }};
  for (auto &P : PassList)
    std::visit(Visitor, P);
  return MPM;
}

void CodeGenLoopPassManager::eraseIf(function_ref<bool(StringRef)> Pred) {
  for (auto I = PassList.begin(), E = PassList.end(); I != E;) {
    std::visit(
        [&](auto &P) {
          if (Pred(P->name()))
            I = PassList.erase(I);
          else
            ++I;
        },
        *I);
  }
}

void CodeGenFunctionPassManager::eraseIf(function_ref<bool(StringRef)> Pred) {
  for (auto I = PassList.begin(), E = PassList.end(); I != E;) {
    const auto Visitor =
        overloads{[&](PassConceptT::unique_ptr &P) {
                    if (Pred(P->name()))
                      I = PassList.erase(I);
                    else
                      ++I;
                  },
                  [&](CodeGenLoopPassManager &CGLPM) {
                    CGLPM.eraseIf(Pred);
                    if (CGLPM.isEmpty())
                      I = PassList.erase(I);
                    else
                      ++I;
                  },
                  [&](CodeGenMachineFunctionPassManager &CGMFPM) {
                    CGMFPM.eraseIf(Pred);
                    if (CGMFPM.isEmpty())
                      I = PassList.erase(I);
                    else
                      ++I;
                  }};
    std::visit(Visitor, *I);
  }
}

void CodeGenModulePassManager::eraseIf(function_ref<bool(StringRef)> Pred) {
  for (auto I = PassList.begin(), E = PassList.end(); I != E;) {
    const auto Visitor = overloads{[&](PassConceptT::unique_ptr &P) {
                                     if (Pred(P->name()))
                                       I = PassList.erase(I);
                                     else
                                       ++I;
                                   },
                                   [&](CodeGenFunctionPassManager &CGFPM) {
                                     CGFPM.eraseIf(Pred);
                                     if (CGFPM.isEmpty())
                                       I = PassList.erase(I);
                                     else
                                       ++I;
                                   }};
    std::visit(Visitor, *I);
  }
}

void CodeGenFunctionPassManager::combineSimilarPassManagers() {
  auto B = PassList.begin(), End = PassList.end();
  auto I = B;
  while (I != End) {
    I = std::find_if(I, End, [](const decltype(PassList)::value_type &V) {
      return std::holds_alternative<CodeGenLoopPassManager>(V) ||
             std::holds_alternative<CodeGenMachineFunctionPassManager>(V);
    });
    auto MergeEnd =
        std::find_if_not(I, End, [&](const decltype(PassList)::value_type &V) {
          if (V.index() != I->index())
            return false;
          if (auto *CGPM = std::get_if<CodeGenLoopPassManager>(&V))
            return CGPM->isSimilarWith(std::get<CodeGenLoopPassManager>(*I));
          else
            return true;
        });

    for (auto J = I; J != MergeEnd;) {
      if (J == I) {
        ++J;
        continue;
      }
      std::visit(
          [&](auto &JPM) {
            if constexpr (!std::is_same_v<
                              std::remove_reference_t<decltype(JPM)>,
                              PassConceptT::unique_ptr>) {
              auto &CGPM = std::get<std::remove_reference_t<decltype(JPM)>>(*I);
              CGPM.addPass(std::move(JPM));
            }
          },
          *J);
      J = PassList.erase(J);
    }
    I = MergeEnd;
  }
}

void CodeGenModulePassManager::handleInsert(
    function_ref<void(StringRef, CodeGenMachineFunctionPassManager &)> F) {
  for (auto &P : PassList) {
    auto *CGFPM = std::get_if<CodeGenFunctionPassManager>(&P);
    if (!CGFPM)
      continue;
    for (auto &PV : CGFPM->PassList) {
      auto *CGMFPM = std::get_if<CodeGenMachineFunctionPassManager>(&PV);
      if (!CGMFPM)
        continue;
      CGMFPM->handleInsert(F);
    }
  }
}

void CodeGenModulePassManager::combineSimilarPassManagers() {
  auto B = PassList.begin(), End = PassList.end();
  auto I = B;
  while (I != End) {
    I = std::find_if(I, End, [](const decltype(PassList)::value_type &V) {
      return std::holds_alternative<CodeGenFunctionPassManager>(V);
    });
    auto MergeEnd =
        std::find_if_not(I, End, [&](const decltype(PassList)::value_type &V) {
          auto *CGPM = std::get_if<CodeGenFunctionPassManager>(&V);
          if (!CGPM)
            return false;
          return CGPM->isSimilarWith(std::get<CodeGenFunctionPassManager>(*I));
        });

    if (I != MergeEnd) {
      auto &CGPM = std::get<CodeGenFunctionPassManager>(*I);
      for (auto J = I; J != MergeEnd;) {
        if (J == I) {
          ++J;
          continue;
        }

        CGPM.addPass(std::move(std::get<CodeGenFunctionPassManager>(*J)));
        J = PassList.erase(J);
      }
      CGPM.combineSimilarPassManagers();
    }

    I = MergeEnd;
  }
}
