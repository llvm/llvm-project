#include "llvm/Transforms/IPO/DeadRTTIElimination.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LibCXXABI.h"

using namespace llvm;

#define DEBUG_TYPE "dre"

STATISTIC(NumDeadTypeInfo, "Number of dead type info global variable");

void DeadRTTIElimIndex::run() {
  if (!ABI)
    return;

  if (ExportSummary.typeIdCompatibleVtableMap().empty())
    return;

  DenseSet<StringRef> TypeIdSlotMayLiveVTables;

  const auto &UsedTypeIds = ExportSummary.getTypeIdAccessed();
  for (StringRef TypeId : UsedTypeIds) {
    auto Info = ExportSummary.getTypeIdCompatibleVtableSummary(TypeId);

    if (!Info.has_value())
      continue;

    for (auto CompatibleVTable : *Info)
      TypeIdSlotMayLiveVTables.insert(CompatibleVTable.VTableVI.name());
  }

  for (auto &VI : ExportSummary) {
    StringRef GVSName = VI.second.U.Name;
    if (!ABI->isVTable(GVSName) ||
        TypeIdSlotMayLiveVTables.contains(GVSName) ||
        VI.second.SummaryList.empty())
      continue;

    auto *GVS = dyn_cast<GlobalVarSummary>(VI.second.SummaryList[0].get());
    if (GVS &&
        GVS->getVCallVisibility() == llvm::GlobalObject::VCallVisibilityPublic)
      continue;

    ++NumDeadTypeInfo;
    for (auto &SL : VI.second.SummaryList)
      SL->eraseRef(ABI->getTypeInfoFromVTable(GVSName));
  }
}
