//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Basic/SequenceToOffsetTable.h"
#include "Common/CodeGenDAGPatterns.h" // For SDNodeInfo.
#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/StringToOffsetTable.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

static cl::OptionCategory SDNodeInfoEmitterCat("Options for -gen-sdnode-info");

static cl::opt<std::string> TargetSDNodeNamespace(
    "sdnode-namespace", cl::cat(SDNodeInfoEmitterCat),
    cl::desc("Specify target SDNode namespace (default=<Target>ISD)"));

static cl::opt<bool> WarnOnSkippedNodes(
    "warn-on-skipped-nodes", cl::cat(SDNodeInfoEmitterCat),
    cl::desc("Explain why a node was skipped (default=true)"), cl::init(true));

namespace {

class SDNodeInfoEmitter {
  const RecordKeeper &RK;
  const CodeGenTarget Target;
  std::map<StringRef, SmallVector<SDNodeInfo, 2>> NodesByName;

public:
  explicit SDNodeInfoEmitter(const RecordKeeper &RK);

  void run(raw_ostream &OS) const;

private:
  void emitEnum(raw_ostream &OS) const;

  std::vector<unsigned> emitNodeNames(raw_ostream &OS) const;

  std::vector<std::pair<unsigned, unsigned>>
  emitTypeConstraints(raw_ostream &OS) const;

  void emitDescs(raw_ostream &OS) const;
};

} // namespace

static bool haveCompatibleDescriptions(const SDNodeInfo &N1,
                                       const SDNodeInfo &N2) {
  // Number of results/operands must match.
  if (N1.getNumResults() != N2.getNumResults() ||
      N1.getNumOperands() != N2.getNumOperands())
    return false;

  // Flags must match.
  if (N1.isStrictFP() != N2.isStrictFP() || N1.getTSFlags() != N2.getTSFlags())
    return false;

  // We're only interested in a subset of node properties. Properties like
  // SDNPAssociative and SDNPCommutative do not impose constraints on nodes,
  // and sometimes differ between nodes sharing the same enum name.
  constexpr unsigned PropMask = (1 << SDNPHasChain) | (1 << SDNPOutGlue) |
                                (1 << SDNPInGlue) | (1 << SDNPOptInGlue) |
                                (1 << SDNPMemOperand) | (1 << SDNPVariadic);

  return (N1.getProperties() & PropMask) == (N2.getProperties() & PropMask);
}

static bool haveCompatibleDescriptions(ArrayRef<SDNodeInfo> Nodes) {
  const SDNodeInfo &N = Nodes.front();
  return all_of(drop_begin(Nodes), [&](const SDNodeInfo &Other) {
    return haveCompatibleDescriptions(Other, N);
  });
}

static void warnOnSkippedNode(const SDNodeInfo &N, const Twine &Reason) {
  PrintWarning(N.getRecord()->getLoc(), "skipped node: " + Reason);
}

SDNodeInfoEmitter::SDNodeInfoEmitter(const RecordKeeper &RK)
    : RK(RK), Target(RK) {
  const CodeGenHwModes &HwModes = Target.getHwModes();

  // Figure out target SDNode namespace.
  if (!TargetSDNodeNamespace.getNumOccurrences())
    TargetSDNodeNamespace = Target.getName().str() + "ISD";

  // Filter nodes by the target SDNode namespace and create a mapping
  // from an enum name to a list of nodes that have that name.
  // The mapping is usually 1:1, but in rare cases it can be 1:N.
  for (const Record *R : RK.getAllDerivedDefinitions("SDNode")) {
    SDNodeInfo Node(R, HwModes);
    auto [NS, EnumName] = Node.getEnumName().split("::");

    if (NS.empty() || EnumName.empty()) {
      if (WarnOnSkippedNodes)
        warnOnSkippedNode(Node, "invalid enum name");
      continue;
    }

    if (NS != TargetSDNodeNamespace)
      continue;

    NodesByName[EnumName].push_back(std::move(Node));
  }

  // Filter out nodes that have different "prototypes" and/or flags.
  // Don't look at type constraints though, we will simply skip emitting
  // the constraints if they differ.
  decltype(NodesByName)::iterator Next;
  for (auto I = NodesByName.begin(), E = NodesByName.end(); I != E; I = Next) {
    Next = std::next(I);

    if (haveCompatibleDescriptions(I->second))
      continue;

    if (WarnOnSkippedNodes)
      for (const SDNodeInfo &N : I->second)
        warnOnSkippedNode(N, "incompatible description");

    NodesByName.erase(I);
  }
}

void SDNodeInfoEmitter::emitEnum(raw_ostream &OS) const {
  OS << "#ifdef GET_SDNODE_ENUM\n";
  OS << "#undef GET_SDNODE_ENUM\n\n";
  OS << "namespace llvm::" << TargetSDNodeNamespace << " {\n\n";

  if (!NodesByName.empty()) {
    StringRef FirstName = NodesByName.begin()->first;
    StringRef LastName = NodesByName.rbegin()->first;

    OS << "enum GenNodeType : unsigned {\n";
    OS << "  " << FirstName << " = ISD::BUILTIN_OP_END,\n";

    for (StringRef EnumName : make_first_range(drop_begin(NodesByName)))
      OS << "  " << EnumName << ",\n";

    OS << "};\n\n";
    OS << "static constexpr unsigned GENERATED_OPCODE_END = " << LastName
       << " + 1;\n\n";
  } else {
    OS << "static constexpr unsigned GENERATED_OPCODE_END = "
          "ISD::BUILTIN_OP_END;\n\n";
  }

  OS << "} // namespace llvm::" << TargetSDNodeNamespace << "\n\n";
  OS << "#endif // GET_SDNODE_ENUM\n\n";
}

std::vector<unsigned> SDNodeInfoEmitter::emitNodeNames(raw_ostream &OS) const {
  StringToOffsetTable NameTable;

  std::vector<unsigned> NameOffsets;
  NameOffsets.reserve(NodesByName.size());

  for (StringRef EnumName : make_first_range(NodesByName)) {
    SmallString<64> DebugName;
    raw_svector_ostream SS(DebugName);
    SS << TargetSDNodeNamespace << "::" << EnumName;
    NameOffsets.push_back(NameTable.GetOrAddStringOffset(DebugName));
  }

  NameTable.EmitStringTableDef(OS, Target.getName() + "SDNodeNames");
  OS << '\n';

  return NameOffsets;
}

static StringRef getTypeConstraintKindName(SDTypeConstraint::KindTy Kind) {
#define CASE(NAME)                                                             \
  case SDTypeConstraint::NAME:                                                 \
    return #NAME

  switch (Kind) {
    CASE(SDTCisVT);
    CASE(SDTCisPtrTy);
    CASE(SDTCisInt);
    CASE(SDTCisFP);
    CASE(SDTCisVec);
    CASE(SDTCisSameAs);
    CASE(SDTCisVTSmallerThanOp);
    CASE(SDTCisOpSmallerThanOp);
    CASE(SDTCisEltOfVec);
    CASE(SDTCisSubVecOfVec);
    CASE(SDTCVecEltisVT);
    CASE(SDTCisSameNumEltsAs);
    CASE(SDTCisSameSizeAs);
  }
  llvm_unreachable("Unknown constraint kind"); // Make MSVC happy.
#undef CASE
}

static void emitTypeConstraint(raw_ostream &OS, SDTypeConstraint C) {
  unsigned OtherOpNo = 0;
  MVT VT;

  switch (C.ConstraintType) {
  case SDTypeConstraint::SDTCisVT:
  case SDTypeConstraint::SDTCVecEltisVT:
    if (C.VVT.isSimple())
      VT = C.VVT.getSimple();
    break;
  case SDTypeConstraint::SDTCisPtrTy:
  case SDTypeConstraint::SDTCisInt:
  case SDTypeConstraint::SDTCisFP:
  case SDTypeConstraint::SDTCisVec:
    break;
  case SDTypeConstraint::SDTCisSameAs:
  case SDTypeConstraint::SDTCisVTSmallerThanOp:
  case SDTypeConstraint::SDTCisOpSmallerThanOp:
  case SDTypeConstraint::SDTCisEltOfVec:
  case SDTypeConstraint::SDTCisSubVecOfVec:
  case SDTypeConstraint::SDTCisSameNumEltsAs:
  case SDTypeConstraint::SDTCisSameSizeAs:
    OtherOpNo = C.OtherOperandNo;
    break;
  }

  StringRef KindName = getTypeConstraintKindName(C.ConstraintType);
  StringRef VTName = VT.SimpleTy == MVT::INVALID_SIMPLE_VALUE_TYPE
                         ? "MVT::INVALID_SIMPLE_VALUE_TYPE"
                         : getEnumName(VT.SimpleTy);
  OS << formatv("{{{}, {}, {}, {}}", KindName, C.OperandNo, OtherOpNo, VTName);
}

std::vector<std::pair<unsigned, unsigned>>
SDNodeInfoEmitter::emitTypeConstraints(raw_ostream &OS) const {
  using ConstraintsVecTy = SmallVector<SDTypeConstraint, 0>;
  SequenceToOffsetTable<ConstraintsVecTy> ConstraintTable(
      /*Terminator=*/std::nullopt);

  std::vector<std::pair<unsigned, unsigned>> ConstraintOffsetsAndCounts;
  ConstraintOffsetsAndCounts.reserve(NodesByName.size());

  SmallVector<StringRef> SkippedNodes;
  for (const auto &[EnumName, Nodes] : NodesByName) {
    ArrayRef<SDTypeConstraint> Constraints = Nodes.front().getTypeConstraints();

    bool IsAmbiguous = any_of(drop_begin(Nodes), [&](const SDNodeInfo &Other) {
      return ArrayRef(Other.getTypeConstraints()) != Constraints;
    });

    // If nodes with the same enum name have different constraints,
    // treat them as if they had no constraints at all.
    if (IsAmbiguous) {
      SkippedNodes.push_back(EnumName);
      continue;
    }

    // Don't add empty sequences to the table. This slightly simplifies
    // the implementation and makes the output less confusing if the table
    // ends up empty.
    if (Constraints.empty())
      continue;

    // SequenceToOffsetTable reuses the storage if a sequence matches another
    // sequence's *suffix*. It is more likely that we have a matching *prefix*,
    // so reverse the order to increase the likelihood of a match.
    ConstraintTable.add(ConstraintsVecTy(reverse(Constraints)));
  }

  ConstraintTable.layout();

  OS << "static const SDTypeConstraint " << Target.getName()
     << "SDTypeConstraints[] = {\n";
  ConstraintTable.emit(OS, emitTypeConstraint);
  OS << "};\n\n";

  for (const auto &[EnumName, Nodes] : NodesByName) {
    ArrayRef<SDTypeConstraint> Constraints = Nodes.front().getTypeConstraints();

    if (Constraints.empty() || is_contained(SkippedNodes, EnumName)) {
      ConstraintOffsetsAndCounts.emplace_back(/*Offset=*/0, /*Size=*/0);
      continue;
    }

    unsigned ConstraintsOffset =
        ConstraintTable.get(ConstraintsVecTy(reverse(Constraints)));
    ConstraintOffsetsAndCounts.emplace_back(ConstraintsOffset,
                                            Constraints.size());
  }

  return ConstraintOffsetsAndCounts;
}

static void emitDesc(raw_ostream &OS, StringRef EnumName,
                     ArrayRef<SDNodeInfo> Nodes, unsigned NameOffset,
                     unsigned ConstraintsOffset, unsigned ConstraintCount) {
  assert(haveCompatibleDescriptions(Nodes));
  const SDNodeInfo &N = Nodes.front();
  OS << "    {" << N.getNumResults() << ", " << N.getNumOperands() << ", 0";

  // Emitted properties must be kept in sync with haveCompatibleDescriptions.
  unsigned Properties = N.getProperties();
  if (Properties & (1 << SDNPHasChain))
    OS << "|1<<SDNPHasChain";
  if (Properties & (1 << SDNPOutGlue))
    OS << "|1<<SDNPOutGlue";
  if (Properties & (1 << SDNPInGlue))
    OS << "|1<<SDNPInGlue";
  if (Properties & (1 << SDNPOptInGlue))
    OS << "|1<<SDNPOptInGlue";
  if (Properties & (1 << SDNPVariadic))
    OS << "|1<<SDNPVariadic";
  if (Properties & (1 << SDNPMemOperand))
    OS << "|1<<SDNPMemOperand";

  OS << ", 0";
  if (N.isStrictFP())
    OS << "|1<<SDNFIsStrictFP";

  OS << formatv(", {}, {}, {}, {}}, // {}\n", N.getTSFlags(), NameOffset,
                ConstraintsOffset, ConstraintCount, EnumName);
}

void SDNodeInfoEmitter::emitDescs(raw_ostream &OS) const {
  StringRef TargetName = Target.getName();

  OS << "#ifdef GET_SDNODE_DESC\n";
  OS << "#undef GET_SDNODE_DESC\n\n";
  OS << "namespace llvm {\n";

  std::vector<unsigned> NameOffsets = emitNodeNames(OS);
  std::vector<std::pair<unsigned, unsigned>> ConstraintOffsetsAndCounts =
      emitTypeConstraints(OS);

  OS << "static const SDNodeDesc " << TargetName << "SDNodeDescs[] = {\n";

  for (const auto &[NameAndNodes, NameOffset, ConstraintOffsetAndCount] :
       zip_equal(NodesByName, NameOffsets, ConstraintOffsetsAndCounts))
    emitDesc(OS, NameAndNodes.first, NameAndNodes.second, NameOffset,
             ConstraintOffsetAndCount.first, ConstraintOffsetAndCount.second);

  OS << "};\n\n";

  OS << formatv("static const SDNodeInfo {0}GenSDNodeInfo(\n"
                "    /*NumOpcodes=*/{1}, {0}SDNodeDescs,\n"
                "    {0}SDNodeNames, {0}SDTypeConstraints);\n\n",
                TargetName, NodesByName.size());

  OS << "} // namespace llvm\n\n";
  OS << "#endif // GET_SDNODE_DESC\n\n";
}

void SDNodeInfoEmitter::run(raw_ostream &OS) const {
  emitSourceFileHeader("Target SDNode descriptions", OS, RK);
  emitEnum(OS);
  emitDescs(OS);
}

static TableGen::Emitter::OptClass<SDNodeInfoEmitter>
    X("gen-sd-node-info", "Generate target SDNode descriptions");
