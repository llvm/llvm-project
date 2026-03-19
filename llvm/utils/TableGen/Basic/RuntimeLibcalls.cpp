//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RuntimeLibcalls.h"
#include "llvm/TableGen/Error.h"

using namespace llvm;

RuntimeLibcalls::RuntimeLibcalls(const RecordKeeper &Records) {
  ArrayRef<const Record *> AllRuntimeLibcalls =
      Records.getAllDerivedDefinitions("RuntimeLibcall");

  RuntimeLibcallDefList.reserve(AllRuntimeLibcalls.size());

  size_t CallTypeEnumVal = 0;
  for (const Record *RuntimeLibcallDef : AllRuntimeLibcalls) {
    RuntimeLibcallDefList.emplace_back(RuntimeLibcallDef, CallTypeEnumVal++);
    Def2RuntimeLibcall[RuntimeLibcallDef] = &RuntimeLibcallDefList.back();
  }

  for (RuntimeLibcall &LibCall : RuntimeLibcallDefList)
    Def2RuntimeLibcall[LibCall.getDef()] = &LibCall;

  ArrayRef<const Record *> AllRuntimeLibcallImplsRaw =
      Records.getAllDerivedDefinitions("RuntimeLibcallImpl");

  SmallVector<const Record *, 1024> AllRuntimeLibcallImpls(
      AllRuntimeLibcallImplsRaw);

  // Sort by libcall impl name and secondarily by the enum name.
  sort(AllRuntimeLibcallImpls, [](const Record *A, const Record *B) {
    return std::pair(A->getValueAsString("LibCallFuncName"), A->getName()) <
           std::pair(B->getValueAsString("LibCallFuncName"), B->getName());
  });

  RuntimeLibcallImplDefList.reserve(AllRuntimeLibcallImpls.size());

  size_t LibCallImplEnumVal = 1;
  for (const Record *LibCallImplDef : AllRuntimeLibcallImpls) {
    RuntimeLibcallImplDefList.emplace_back(LibCallImplDef, Def2RuntimeLibcall,
                                           LibCallImplEnumVal++);

    const RuntimeLibcallImpl &LibCallImpl = RuntimeLibcallImplDefList.back();
    Def2RuntimeLibcallImpl[LibCallImplDef] = &LibCallImpl;

    if (LibCallImpl.isDefault()) {
      const RuntimeLibcall *Provides = LibCallImpl.getProvides();
      if (!Provides)
        PrintFatalError(LibCallImplDef->getLoc(),
                        "default implementations must provide a libcall");
      LibCallToDefaultImpl[Provides] = &LibCallImpl;
    }
  }
}

void LibcallPredicateExpander::expand(SetTheory &ST, const Record *Def,
                                      SetTheory::RecSet &Elts) {
  assert(Def->isSubClassOf("LibcallImpls"));

  SetTheory::RecSet TmpElts;

  ST.evaluate(Def->getValueInit("MemberList"), TmpElts, Def->getLoc());

  Elts.insert(TmpElts.begin(), TmpElts.end());

  AvailabilityPredicate AP(Def->getValueAsDef("AvailabilityPredicate"));
  const Record *CCClass = Def->getValueAsOptionalDef("CallingConv");

  // This is assuming we aren't conditionally applying a calling convention to
  // some subsets, and not another, but this doesn't appear to be used.

  for (const Record *LibcallImplDef : TmpElts) {
    const RuntimeLibcallImpl *LibcallImpl =
        Libcalls.getRuntimeLibcallImpl(LibcallImplDef);
    if (!AP.isAlwaysAvailable() || CCClass) {
      auto [It, Inserted] = Func2Preds.insert({LibcallImpl, {{}, CCClass}});
      if (!Inserted) {
        PrintError(
            Def,
            "combining nested libcall set predicates currently unhandled: '" +
                LibcallImpl->getLibcallFuncName() + "'");
      }

      It->second.first.push_back(AP.getDef());
      It->second.second = CCClass;
    }
  }
}
