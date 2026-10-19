//===-- Shared/TaskGraphDebugDump.h - dump graph and data plan ---- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_SHARED_TASKGRAPHDEBUGDUMP_H
#define OMPTARGET_SHARED_TASKGRAPHDEBUGDUMP_H

#include "TaskGraph.h"
#include "llvm/Support/Debug.h"

namespace llvm {
namespace omp {
namespace target {

class TaskGraphDebugDump {
  TaskGraphTy &G;
  llvm::raw_ostream &OS;

  void dumpRanges(const llvm::ArrayRef<TaskGraphRangeTy> Ranges) {
    for (const auto &R : Ranges) {
      OS << "Id: " << R.getId() << ", ";
      OS << "Host addr: " << R.getHostBegin() << ", ";
      OS << "Device ptr: " << R.getDevicePtr() << ", ";
      OS << "Size: " << R.getSize() << "\n";
    }
  }

  void dumpCopy(const char *Dir, const TaskGraphCopyTy &Copy) {
    if (Copy.Requested) {
      OS << ", " << Dir << " (";
      OS << (Copy.Perimeter ? "perimeter" : "internal");
      OS << (Copy.Enabled ? ", enabled" : ", disabled");
      OS << ")";
    }
  }

  void dumpArg(const TaskGraphDataPlanTy &Plan, size_t I,
               const TaskGraphArgSlotTy &Slot, int Indent = 0) {
    const auto *Arg = &Slot.get();
    // All the args are stored in a single array, and a particular construct
    // uses a slice of that.  The numbering shown is global/local.  Updates
    // use the global index; ops use the local index.
    OS.indent(Indent) << Plan.argIndex(Slot) << "/" << I << ": ";
    switch (Arg->getKind()) {
    case TaskGraphArgBaseTy::KindTy::None:
      OS << "none\n";
      break;
    case TaskGraphArgBaseTy::KindTy::Literal: {
      const auto *L = cast<TaskGraphLiteralArgTy>(Arg);
      OS << "literal (" << L->LiteralValue << ")\n";
      break;
    }
    case TaskGraphArgBaseTy::KindTy::Mapped: {
      const auto *M = cast<TaskGraphMappedArgTy>(Arg);
      OS << "mapped (range: " << M->Range->getId() << ", offset: " << M->Offset
         << ", size: " << M->Size;
      dumpCopy("in", M->In);
      dumpCopy("out", M->Out);
      if (M->OwnsBorrowedOut)
        OS << ", owns-borrowed-out";
      if (M->Always)
        OS << ", always";
      OS << ")\n";
      break;
    }
    case TaskGraphArgBaseTy::KindTy::Attach: {
      const auto *A = cast<TaskGraphAttachArgTy>(Arg);
      OS << "attach (ptr-range: " << A->PtrRange->getId()
         << ", offset: " << A->Offset << ", size: " << A->Size
         << ", ptee-range: " << A->PteeRange->getId()
         << ", ptee-offset: " << A->PteeOffset
         << ", staging-offset: " << A->StagingOffset;
      if (A->Enabled)
        OS << ", enabled)\n";
      else
        OS << ", disabled)\n";
      break;
    }
    case TaskGraphArgBaseTy::KindTy::NoTargetAttach:
      OS << "no-target-attach\n";
      break;
    case TaskGraphArgBaseTy::KindTy::FirstPrivate: {
      const auto *F = cast<TaskGraphFirstPrivateArgTy>(Arg);
      OS << "firstprivate (base-offset: " << F->BaseOffset
         << ", size: " << F->Size;
      if (F->Staged)
        OS << ", staging-offset: " << F->StagingOffset;
      else
        OS << ", not-staged";
      if (F->Saved)
        OS << ", saved";
      if (F->InitPending)
        OS << ", init-pending";
      OS << ")\n";
      break;
    }
    }
  }

  void dumpOp(size_t I, const TaskGraphOpTy &Op, int Indent = 0) {
    OS.indent(Indent) << I << ": ";
    switch (Op.Kind) {
    case TaskGraphOpTy::KindTy::Attach:
      OS << "attach";
      break;
    case TaskGraphOpTy::KindTy::PrivateInit:
      OS << "private-init";
      break;
    case TaskGraphOpTy::KindTy::CopyIn:
      OS << "copy-in";
      break;
    case TaskGraphOpTy::KindTy::CopyOut:
      OS << "copy-out";
      break;
    case TaskGraphOpTy::KindTy::Launch:
      OS << "launch\n";
      break;
    }
    if (Op.isCopyLike())
      OS << " (arg: " << Op.Arg << ")\n";
  }

  void dumpLeaf(const TaskGraphDataPlanTy &Plan, const TaskGraphElement *Elem,
                const TaskGraphRegion *Parent = nullptr, int Indent = 0) {
    auto IrredSuccessors = [&](const TaskGraphRegion *Parent, int Indent) {
      if (const auto *Irr =
              dyn_cast_if_present<TaskGraphIrreducibleRegion>(Parent)) {
        OS.indent(Indent) << "successors {\n";
        const char *Sep = "";
        for (size_t Succ = 0; Succ < Irr->NumEdges; ++Succ) {
          const auto &Edge = Irr->edges()[Succ];
          if (Edge.Src == (int32_t)Elem->ChildIndex) {
            if (Sep[0] == '\0')
              OS.indent(Indent + 2) << "-> ";
            OS << Sep << "[" << Edge.Dst << "]";
            Sep = ", ";
          }
        }
        OS << "\n";
        OS.indent(Indent) << "}\n";
      }
    };
    if (isa_and_present<TaskGraphIrreducibleRegion>(Parent))
      OS.indent(Indent) << "[" << Elem->ChildIndex << "] ";
    else
      OS << indent(Indent);
    if (const auto *R = dyn_cast<const TaskGraphRegion>(Elem)) {
      if (isa<const TaskGraphParRegion>(R)) {
        OS << "parallel {\n";
      } else if (isa<const TaskGraphSeqRegion>(R)) {
        OS << "sequential {\n";
      } else if (isa<const TaskGraphExclRegion>(R)) {
        OS << "exclusive {\n";
      } else if (isa<const TaskGraphIrreducibleRegion>(R)) {
        OS << "irreducible {\n";
      }
      for (const auto *C : R->children()) {
        dumpLeaf(Plan, C, R, Indent + 2);
      }
      IrredSuccessors(Parent, Indent + 2);
      OS.indent(Indent) << "}\n";
    } else if (const auto *L = dyn_cast<const TaskGraphNode>(Elem)) {
      if (isa<const TaskGraphTargetNode>(L)) {
        OS << "target {\n";
      } else if (isa<const TaskGraphTargetEnterDataNode>(L)) {
        OS << "enter data {\n";
      } else if (isa<const TaskGraphTargetExitDataNode>(L)) {
        OS << "exit data {\n";
      } else if (isa<const TaskGraphTargetUpdateDataNode>(L)) {
        OS << "update {\n";
      }
      OS.indent(Indent + 2) << "args {\n";
      for (size_t I = 0; I < L->argPlan().size(); ++I)
        dumpArg(Plan, I, L->argPlan()[I], Indent + 4);
      OS.indent(Indent + 2) << "}\n";
      OS.indent(Indent + 2) << "ops {\n";
      for (size_t I = 0; I < L->ops().size(); ++I)
        dumpOp(I, L->ops()[I], Indent + 4);
      OS.indent(Indent + 2) << "}\n";
      IrredSuccessors(Parent, Indent + 2);
      OS.indent(Indent) << "}\n";
    }
  }

public:
  explicit TaskGraphDebugDump(TaskGraphTy &G, llvm::raw_ostream &OS)
      : G(G), OS(OS) {}

  void dumpDataPlan() {
    OS << "Ranges:\n";
    dumpRanges(G.Plan.ranges());
    OS << "\n";
    OS << "Firstprivate staging buffer size: " << G.Plan.privateStaging().size()
       << ", device ptr: " << G.Plan.getPrivateBlock() << "\n\n";
    OS << "attach staging buffer size: " << G.Plan.attachStaging().size()
       << "\n\n";
    OS << "Graph " << (void *)&G << ":\n";
    dumpLeaf(G.Plan, G.getRoot());
  }

  void dumpUpdates(llvm::ArrayRef<TaskGraphArgUpdateTy> Updates, bool Resync) {
    if (Resync) {
      OS << "Updates all (resync)\n";
      return;
    }
    OS << "Updates:\n";
    for (const auto &U : Updates) {
      OS << "Arg " << U.Arg << " changes:";
      if (U.has(TaskGraphArgUpdateTy::HostAddressChanged))
        OS << " host-addr";
      if (U.has(TaskGraphArgUpdateTy::DeviceAddressChanged))
        OS << " device-addr";
      if (U.has(TaskGraphArgUpdateTy::DataChanged))
        OS << " data";
      if (U.has(TaskGraphArgUpdateTy::EnabledChanged))
        OS << " enable";
      OS << "\n";
    }
    if (Updates.empty())
      OS << "(no updates)\n";
  }
};

} // namespace target
} // namespace omp
} // namespace llvm

#endif // OMPTARGET_SHARED_TASKGRAPHDEBUGDUMP_H
