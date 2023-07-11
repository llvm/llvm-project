//===- GlobalISelMatchTableExecutorEmitter.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GlobalISelMatchTableExecutorEmitter.h"
#include "GlobalISelMatchTable.h"

using namespace llvm;
using namespace llvm::gi;

void GlobalISelMatchTableExecutorEmitter::emitSubtargetFeatureBitsetImpl(
    raw_ostream &OS, ArrayRef<RuleMatcher> Rules) {
  SubtargetFeatureInfo::emitSubtargetFeatureBitEnumeration(SubtargetFeatures,
                                                           OS);

  // Separate subtarget features by how often they must be recomputed.
  SubtargetFeatureInfoMap ModuleFeatures;
  std::copy_if(SubtargetFeatures.begin(), SubtargetFeatures.end(),
               std::inserter(ModuleFeatures, ModuleFeatures.end()),
               [](const SubtargetFeatureInfoMap::value_type &X) {
                 return !X.second.mustRecomputePerFunction();
               });
  SubtargetFeatureInfoMap FunctionFeatures;
  std::copy_if(SubtargetFeatures.begin(), SubtargetFeatures.end(),
               std::inserter(FunctionFeatures, FunctionFeatures.end()),
               [](const SubtargetFeatureInfoMap::value_type &X) {
                 return X.second.mustRecomputePerFunction();
               });

  SubtargetFeatureInfo::emitComputeAvailableFeatures(
      getTarget().getName(), getClassName(), "computeAvailableModuleFeatures",
      ModuleFeatures, OS);

  OS << "void " << getClassName()
     << "::setupGeneratedPerFunctionState(MachineFunction &MF) {\n"
        "  AvailableFunctionFeatures = computeAvailableFunctionFeatures("
        "(const "
     << getTarget().getName()
     << "Subtarget *)&MF.getSubtarget(), &MF);\n"
        "}\n";

  SubtargetFeatureInfo::emitComputeAvailableFeatures(
      getTarget().getName(), getClassName(), "computeAvailableFunctionFeatures",
      FunctionFeatures, OS, "const MachineFunction *MF");

  // Emit a table containing the PredicateBitsets objects needed by the matcher
  // and an enum for the matcher to reference them with.
  std::vector<std::vector<Record *>> FeatureBitsets;
  FeatureBitsets.reserve(Rules.size());
  for (auto &Rule : Rules)
    FeatureBitsets.push_back(Rule.getRequiredFeatures());
  llvm::sort(FeatureBitsets, [&](const std::vector<Record *> &A,
                                 const std::vector<Record *> &B) {
    if (A.size() < B.size())
      return true;
    if (A.size() > B.size())
      return false;
    for (auto [First, Second] : zip(A, B)) {
      if (First->getName() < Second->getName())
        return true;
      if (First->getName() > Second->getName())
        return false;
    }
    return false;
  });
  FeatureBitsets.erase(
      std::unique(FeatureBitsets.begin(), FeatureBitsets.end()),
      FeatureBitsets.end());
  OS << "// Feature bitsets.\n"
     << "enum {\n"
     << "  GIFBS_Invalid,\n";
  for (const auto &FeatureBitset : FeatureBitsets) {
    if (FeatureBitset.empty())
      continue;
    OS << "  " << getNameForFeatureBitset(FeatureBitset) << ",\n";
  }
  OS << "};\n"
     << "const static PredicateBitset FeatureBitsets[] {\n"
     << "  {}, // GIFBS_Invalid\n";
  for (const auto &FeatureBitset : FeatureBitsets) {
    if (FeatureBitset.empty())
      continue;
    OS << "  {";
    for (const auto &Feature : FeatureBitset) {
      const auto &I = SubtargetFeatures.find(Feature);
      assert(I != SubtargetFeatures.end() && "Didn't import predicate?");
      OS << I->second.getEnumBitName() << ", ";
    }
    OS << "},\n";
  }
  OS << "};\n\n";
}

void GlobalISelMatchTableExecutorEmitter::emitComplexPredicates(
    raw_ostream &OS, ArrayRef<Record *> ComplexOperandMatchers) {
  // Emit complex predicate table and an enum to reference them with.
  OS << "// ComplexPattern predicates.\n"
     << "enum {\n"
     << "  GICP_Invalid,\n";
  for (const auto &Record : ComplexOperandMatchers)
    OS << "  GICP_" << Record->getName() << ",\n";
  OS << "};\n"
     << "// See constructor for table contents\n\n";

  OS << getClassName() << "::ComplexMatcherMemFn\n"
     << getClassName() << "::ComplexPredicateFns[] = {\n"
     << "  nullptr, // GICP_Invalid\n";
  for (const auto &Record : ComplexOperandMatchers)
    OS << "  &" << getClassName()
       << "::" << Record->getValueAsString("MatcherFn") << ", // "
       << Record->getName() << "\n";
  OS << "};\n\n";
}

void GlobalISelMatchTableExecutorEmitter::emitCustomOperandRenderers(
    raw_ostream &OS, ArrayRef<StringRef> CustomOperandRenderers) {
  OS << "// Custom renderers.\n"
     << "enum {\n"
     << "  GICR_Invalid,\n";
  for (const auto &Fn : CustomOperandRenderers)
    OS << "  GICR_" << Fn << ",\n";
  OS << "};\n";

  OS << getClassName() << "::CustomRendererFn\n"
     << getClassName() << "::CustomRenderers[] = {\n"
     << "  nullptr, // GICR_Invalid\n";
  for (const auto &Fn : CustomOperandRenderers)
    OS << "  &" << getClassName() << "::" << Fn << ",\n";
  OS << "};\n\n";
}

void GlobalISelMatchTableExecutorEmitter::emitTypeObjects(
    raw_ostream &OS, ArrayRef<LLTCodeGen> TypeObjects) {
  OS << "// LLT Objects.\n"
     << "enum {\n";
  for (const auto &TypeObject : TypeObjects) {
    OS << "  ";
    TypeObject.emitCxxEnumValue(OS);
    OS << ",\n";
  }
  OS << "};\n"
     << "const static size_t NumTypeObjects = " << TypeObjects.size() << ";\n"
     << "const static LLT TypeObjects[] = {\n";
  for (const auto &TypeObject : TypeObjects) {
    OS << "  ";
    TypeObject.emitCxxConstructorCall(OS);
    OS << ",\n";
  }
  OS << "};\n\n";
}

void GlobalISelMatchTableExecutorEmitter::emitMatchTable(
    raw_ostream &OS, const MatchTable &Table) {
  OS << "const int64_t *" << getClassName() << "::getMatchTable() const {\n";
  Table.emitDeclaration(OS);
  OS << "  return ";
  Table.emitUse(OS);
  OS << ";\n}\n";
}

void GlobalISelMatchTableExecutorEmitter::emitExecutorImpl(
    raw_ostream &OS, const MatchTable &Table, ArrayRef<LLTCodeGen> TypeObjects,
    ArrayRef<RuleMatcher> Rules, ArrayRef<Record *> ComplexOperandMatchers,
    ArrayRef<StringRef> CustomOperandRenderers, StringRef IfDefName) {
  OS << "#ifdef " << IfDefName << "\n";
  emitTypeObjects(OS, TypeObjects);
  emitSubtargetFeatureBitsetImpl(OS, Rules);
  emitComplexPredicates(OS, ComplexOperandMatchers);
  emitMIPredicateFns(OS);
  emitI64ImmPredicateFns(OS);
  emitAPFloatImmPredicateFns(OS);
  emitAPIntImmPredicateFns(OS);
  emitTestSimplePredicate(OS);
  emitCustomOperandRenderers(OS, CustomOperandRenderers);
  emitAdditionalImpl(OS);
  emitRunCustomAction(OS);
  emitMatchTable(OS, Table);
  OS << "#endif // ifdef " << IfDefName << "\n\n";
}

void GlobalISelMatchTableExecutorEmitter::emitPredicateBitset(
    raw_ostream &OS, StringRef IfDefName) {
  OS << "#ifdef " << IfDefName << "\n"
     << "const unsigned MAX_SUBTARGET_PREDICATES = " << SubtargetFeatures.size()
     << ";\n"
     << "using PredicateBitset = "
        "llvm::PredicateBitsetImpl<MAX_SUBTARGET_PREDICATES>;\n"
     << "#endif // ifdef " << IfDefName << "\n\n";
}

void GlobalISelMatchTableExecutorEmitter::emitTemporariesDecl(
    raw_ostream &OS, StringRef IfDefName) {
  OS << "#ifdef " << IfDefName << "\n"
     << "  mutable MatcherState State;\n"
     << "  typedef "
        "ComplexRendererFns("
     << getClassName() << "::*ComplexMatcherMemFn)(MachineOperand &) const;\n"

     << "  typedef void(" << getClassName()
     << "::*CustomRendererFn)(MachineInstrBuilder &, const "
        "MachineInstr &, int) "
        "const;\n"
     << "  const ExecInfoTy<PredicateBitset, ComplexMatcherMemFn, "
        "CustomRendererFn> "
        "ExecInfo;\n"
     << "  static " << getClassName()
     << "::ComplexMatcherMemFn ComplexPredicateFns[];\n"
     << "  static " << getClassName()
     << "::CustomRendererFn CustomRenderers[];\n"
     << "  bool testImmPredicate_I64(unsigned PredicateID, int64_t Imm) const "
        "override;\n"
     << "  bool testImmPredicate_APInt(unsigned PredicateID, const APInt &Imm) "
        "const override;\n"
     << "  bool testImmPredicate_APFloat(unsigned PredicateID, const APFloat "
        "&Imm) const override;\n"
     << "  const int64_t *getMatchTable() const override;\n"
     << "  bool testMIPredicate_MI(unsigned PredicateID, const MachineInstr &MI"
        ", const MatcherState &State) "
        "const override;\n"
     << "  bool testSimplePredicate(unsigned PredicateID) const override;\n"
     << "  void runCustomAction(unsigned FnID, const MatcherState &State) "
        "const override;\n";
  emitAdditionalTemporariesDecl(OS, "  ");
  OS << "#endif // ifdef " << IfDefName << "\n\n";
}

void GlobalISelMatchTableExecutorEmitter::emitTemporariesInit(
    raw_ostream &OS, unsigned MaxTemporaries, StringRef IfDefName) {
  OS << "#ifdef " << IfDefName << "\n"
     << ", State(" << MaxTemporaries << "),\n"
     << "ExecInfo(TypeObjects, NumTypeObjects, FeatureBitsets"
     << ", ComplexPredicateFns, CustomRenderers)\n"
     << "#endif // ifdef " << IfDefName << "\n\n";

  emitAdditionalTemporariesInit(OS);
}

void GlobalISelMatchTableExecutorEmitter::emitPredicatesDecl(
    raw_ostream &OS, StringRef IfDefName) {
  OS << "#ifdef " << IfDefName << "\n"
     << "PredicateBitset AvailableModuleFeatures;\n"
     << "mutable PredicateBitset AvailableFunctionFeatures;\n"
     << "PredicateBitset getAvailableFeatures() const {\n"
     << "  return AvailableModuleFeatures | AvailableFunctionFeatures;\n"
     << "}\n"
     << "PredicateBitset\n"
     << "computeAvailableModuleFeatures(const " << getTarget().getName()
     << "Subtarget *Subtarget) const;\n"
     << "PredicateBitset\n"
     << "computeAvailableFunctionFeatures(const " << getTarget().getName()
     << "Subtarget *Subtarget,\n"
     << "                                 const MachineFunction *MF) const;\n"
     << "void setupGeneratedPerFunctionState(MachineFunction &MF) override;\n"
     << "#endif // ifdef " << IfDefName << "\n";
}

void GlobalISelMatchTableExecutorEmitter::emitPredicatesInit(
    raw_ostream &OS, StringRef IfDefName) {
  OS << "#ifdef " << IfDefName << "\n"
     << "AvailableModuleFeatures(computeAvailableModuleFeatures(&STI)),\n"
     << "AvailableFunctionFeatures()\n"
     << "#endif // ifdef " << IfDefName << "\n";
}
