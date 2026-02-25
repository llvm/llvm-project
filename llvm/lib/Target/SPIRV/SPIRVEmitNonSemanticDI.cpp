#include "MCTargetDesc/SPIRVBaseInfo.h"
#include "MCTargetDesc/SPIRVMCTargetDesc.h"
#include "SPIRV.h"
#include "SPIRVGlobalRegistry.h"
#include "SPIRVRegisterInfo.h"
#include "SPIRVTargetMachine.h"
#include "SPIRVUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "spirv-nonsemantic-debug-info"

using namespace llvm;

namespace {
struct SPIRVCodeGenContext {
  MachineIRBuilder &MIRBuilder;
  MachineRegisterInfo &MRI;
  SPIRVGlobalRegistry *GR;
  const SPIRVTypeInst VoidTy;
  const SPIRVTypeInst I32Ty;
  const SPIRVInstrInfo *TII;
  const SPIRVRegisterInfo *TRI;
  const RegisterBankInfo *RBI;
  MachineFunction &MF;
  const Register &I32ZeroReg;
  SPIRVTargetMachine *TM;
  SmallVector<std::pair<const DIFile *const, const Register>, 24>
      &SourceRegPairs;
  SmallVector<std::pair<const DIScope *const, const Register>, 24>
      &ScopeRegPairs;
  SmallVector<std::pair<const DISubroutineType *const, const Register>, 24>
      &SubRoutineTypeRegPairs;
  SmallVector<std::pair<const DIBasicType *const, const Register>, 24>
      &BasicTypeRegPairs;
  SmallVector<std::pair<const DICompositeType *const, const Register>, 24>
      &CompositeTypeRegPairs;
  SmallVector<std::pair<const DIFile *const, const Register>, 24>
      &CompileUnitRegPairs;
  SmallVector<std::pair<const DIDerivedType *const, const Register>, 24>
      &PointerRegPairs;
  SmallVector<std::pair<const DILocation *, Register>, 24> &InlinedAtRegPairs;
  SmallVector<std::pair<const DIExpression *, Register>, 12> &ExprRegPairs;

  SPIRVCodeGenContext(
      MachineIRBuilder &Builder, MachineRegisterInfo &RegInfo,
      SPIRVGlobalRegistry *Registry, SPIRVTypeInst VTy,
      SPIRVTypeInst I32Ty, const SPIRVInstrInfo *TI,
      const SPIRVRegisterInfo *TR, const RegisterBankInfo *RB,
      MachineFunction &Function, const Register &ZeroReg,
      SPIRVTargetMachine *TargetMachine,
      SmallVector<std::pair<const DIFile *const, const Register>, 24>
          &SourceRegisterPairs,
      SmallVector<std::pair<const DIScope *const, const Register>, 24>
          &ScopeRegisterPairs,
      SmallVector<std::pair<const DISubroutineType *const, const Register>, 24>
          &SubRoutineTypeRegisterPairs,
      SmallVector<std::pair<const DIBasicType *const, const Register>, 24>
          &BasicTypePairs,
      SmallVector<std::pair<const DICompositeType *const, const Register>, 24>
          &CompositeTypePairs,
      SmallVector<std::pair<const DIFile *const, const Register>, 24>
          &CompileUnitPairs,
      SmallVector<std::pair<const DIDerivedType *const, const Register>, 24>
          &PointerPairs,
      SmallVector<std::pair<const DILocation *, Register>, 24> &InlinedAtPairs,
      SmallVector<std::pair<const DIExpression *, Register>, 12> &ExprPairs)
      : MIRBuilder(Builder), MRI(RegInfo), GR(Registry), VoidTy(VTy),
        I32Ty(I32Ty), TII(TI), TRI(TR), RBI(RB), MF(Function),
        I32ZeroReg(ZeroReg), TM(TargetMachine),
        SourceRegPairs(SourceRegisterPairs), ScopeRegPairs(ScopeRegisterPairs),
        SubRoutineTypeRegPairs(SubRoutineTypeRegisterPairs),
        BasicTypeRegPairs(BasicTypePairs),
        CompositeTypeRegPairs(CompositeTypePairs),
        CompileUnitRegPairs(CompileUnitPairs), PointerRegPairs(PointerPairs),
        InlinedAtRegPairs(InlinedAtPairs), ExprRegPairs(ExprPairs) {}
};
struct DebugInfoCollector {
  SmallPtrSet<DIBasicType *, 12> BasicTypes;
  SmallPtrSet<DIDerivedType *, 12> PointerDerivedTypes;
  SmallPtrSet<DIDerivedType *, 12> QualifiedDerivedTypes;
  SmallPtrSet<DIDerivedType *, 12> TypedefTypes;
  SmallPtrSet<DIDerivedType *, 12> InheritedTypes;
  SmallPtrSet<DIDerivedType *, 12> PtrToMemberTypes;
  SmallVector<const DIImportedEntity *, 5> ImportedEntities;
  SmallVector<const DIFile *, 4> CUFileNodes;
  SmallVector<const DICompileUnit *, 4> CUNodes;
  SmallPtrSet<DISubprogram *, 12> SubPrograms;
  SmallPtrSet<DISubroutineType *, 12> SubRoutineTypes;
  SmallPtrSet<DIScope *, 12> LexicalScopes;
  SmallPtrSet<DICompositeType *, 12> ArrayTypes;
  SmallPtrSet<const DICompositeType *, 8> CompositeTypesWithTemplates;
  SmallPtrSet<const DICompositeType *, 8> CompositeTypes;
  SmallPtrSet<const DICompositeType *, 8> EnumTypes;
  DenseSet<const DIType *> visitedTypes;
  SmallVector<const DIExpression *, 10> ExprList;
  SmallSet<const DIExpression *, 10> SeenExprs;
  SmallPtrSet<DILocalVariable *, 12> LocalVariables;
  SmallPtrSet<DILocation *, 12> InlinedAtLoc;
};
struct SPIRVEmitNonSemanticDI : public ModulePass {
  static char ID;
  SPIRVTargetMachine *TM;
  SPIRVEmitNonSemanticDI(SPIRVTargetMachine *TM = nullptr)
      : ModulePass(ID), TM(TM) {}

  bool runOnModule(llvm::Module &M) override;

private:
  bool IsGlobalDIEmitted = false;
  bool emitGlobalDI(MachineFunction &MF, SPIRVCodeGenContext &Ctx,
                    DebugInfoCollector &Collector);
  Register EmitOpString(StringRef, SPIRVCodeGenContext &Ctx);
  uint32_t transDebugFlags(const DINode *DN);
  uint32_t mapTagToCompositeEncoding(const DICompositeType *CT);
  uint32_t mapTagToQualifierEncoding(unsigned Tag);
  uint32_t mapDebugFlags(DINode::DIFlags DFlags);
  uint32_t mapImportedTagToEncoding(const DIImportedEntity *Imported);

  Register EmitDIInstruction(SPIRV::NonSemanticExtInst::NonSemanticExtInst Inst,
                             ArrayRef<Register> Operands,
                             SPIRVCodeGenContext &Ctx, bool HasForwardRef,
                             Register DefReg = Register(0));

  Register getOrCreateFileRegister(const DIFile *File,
                                   SPIRVCodeGenContext &Ctx);
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void emitDebugLineAndScopeInstructions(SPIRVCodeGenContext &Ctx);

  Register findEmittedBasicTypeReg(
      const DIType *BaseType,
      const SmallVectorImpl<std::pair<const DIBasicType *const, const Register>>
          &BasicTypeRegPairs);

  Register findEmittedCompositeTypeReg(
      const DIType *BaseType,
      const SmallVectorImpl<std::pair<const DICompositeType *const,
                                      const Register>> &CompositeTypeRegPairs);

  Register emitDebugSource(StringRef FilePath, StringRef FileContents,
                           SPIRVCodeGenContext &Ctx);

  void extractTypeMetadata(DIType *Ty, DebugInfoCollector &Collector);

  void handleCompositeType(DICompositeType *CT, DebugInfoCollector &Collector);
  void handleDerivedType(DIDerivedType *DT, DebugInfoCollector &Collector);

  void emitDebugBuildIdentifier(StringRef BuildIdentifier,
                                SPIRVCodeGenContext &Ctx);

  void emitDebugStoragePath(StringRef BuildStoragePath,
                            SPIRVCodeGenContext &Ctx);

  void emitDebugBasicTypes(const SmallPtrSetImpl<DIBasicType *> &BasicTypes,
                           SPIRVCodeGenContext &Ctx);

  void emitDebugPointerTypes(
      const SmallPtrSetImpl<DIDerivedType *> &PointerDerivedTypes,
      SPIRVCodeGenContext &Ctx);

  void emitDebugValueInstruction(MachineInstr &MI, SPIRVCodeGenContext &Ctx);

  void emitSingleCompilationUnit(
      StringRef FilePath, int64_t SourceLanguage, SPIRVCodeGenContext &Ctx,
      Register DebugInfoVersionReg, Register DwarfVersionReg,
      Register &DebugSourceResIdReg, Register &DebugCompUnitResIdReg,
      const DIFile *FileNode, const DICompileUnit *CUNode);

  void emitLexicalScopes(const SmallPtrSetImpl<DIScope *> &LexicalScopes,
                         SPIRVCodeGenContext &Ctx);

  void
  emitDebugLocalVariables(SPIRVCodeGenContext &Ctx,
                          SmallPtrSet<DILocalVariable *, 12> LocalVariables);

  void emitDebugLineAndScope(MachineInstr &MI, SPIRVCodeGenContext &Ctx,
                             const DIScope *&CurrentScope,
                             const DILocation *&CurrentInlinedAt);

  Register getOrCreateInlinedAt(const DILocation *Loc,
                                SPIRVCodeGenContext &Ctx);

  void emitDebugArrayTypes(const SmallPtrSetImpl<DICompositeType *> &ArrayTypes,
                           SPIRVCodeGenContext &Ctx);

  void emitDebugVectorTypes(DICompositeType *ArrayTy, Register BaseTypeReg,
                            SPIRVCodeGenContext &Ctx);

  void emitDebugTypeComposite(const DICompositeType *CompTy,
                              SPIRVCodeGenContext &Ctx);

  void emitAllTemplateDebugInstructions(
      const SmallPtrSetImpl<const DICompositeType *> &TemplatedTypes,
      SPIRVCodeGenContext &Ctx);

  void emitAllDebugTypeComposites(
      const SmallPtrSetImpl<const DICompositeType *> &CompositeTypes,
      SPIRVCodeGenContext &Ctx);

  void emitSubroutineTypes(
      const SmallPtrSet<DISubroutineType *, 12> &SubRoutineTypes,
      SPIRVCodeGenContext &Ctx);

  Register findBaseTypeRegisterRecursive(const DIType *Ty,
                                         SPIRVCodeGenContext &Ctx,
                                         bool &IsForwardRef);

  void emitSubprograms(const SmallPtrSet<DISubprogram *, 12> &SubPrograms,
                       SPIRVCodeGenContext &Ctx);

  void emitDebugTypeMember(const DIDerivedType *Member,
                           SPIRVCodeGenContext &Ctx,
                           const Register &CompositeReg,
                           SmallVectorImpl<Register> &MemberRegs,
                           Register DebugSourceReg);

  void emitDebugMacroDefs(MachineFunction &MF, SPIRVCodeGenContext &Ctx);

  void emitDebugMacroUndef(const DIMacro *MacroUndef, StringRef FileName,
                           SPIRVCodeGenContext &Ctx,
                           const DenseMap<StringRef, Register> &MacroDefRegs);

  void
  emitDebugExpressions(const SmallVectorImpl<const DIExpression *> &ExprList,
                       SPIRVCodeGenContext &Ctx);

  void
  emitDebugTypeEnum(const SmallPtrSetImpl<const DICompositeType *> &EnumTypes,
                    SPIRVCodeGenContext &Ctx);

  void emitDebugQualifiedTypes(
      const SmallPtrSetImpl<DIDerivedType *> &QualifiedDerivedTypes,
      SPIRVCodeGenContext &Ctx);

  void emitDebugTypedefs(const SmallPtrSetImpl<DIDerivedType *> &TypedefTypes,
                         SPIRVCodeGenContext &Ctx);

  void emitDebugImportedEntities(
      const SmallVectorImpl<const DIImportedEntity *> &ImportedEntities,
      SPIRVCodeGenContext &Ctx);

  Register emitDebugGlobalVariable(const DIGlobalVariableExpression *GVE,
                                   SPIRVCodeGenContext &Ctx);

  void emitAllDebugGlobalVariables(MachineFunction &MF,
                                   SPIRVCodeGenContext &Ctx);

  void emitDebugTypePtrToMember(
      const SmallPtrSetImpl<DIDerivedType *> &PtrToMemberTypes,
      SPIRVCodeGenContext &Ctx);

  void emitDebugEntryPoint(Register DebugFunctionReg, Register DebugCompUnitReg,
                           SPIRVCodeGenContext &Ctx);
};
} // anonymous namespace

INITIALIZE_PASS(SPIRVEmitNonSemanticDI, DEBUG_TYPE,
                "SPIRV NonSemantic.Shader.DebugInfo.100 emitter", false, false)

char SPIRVEmitNonSemanticDI::ID = 0;

ModulePass *llvm::createSPIRVEmitNonSemanticDIPass(SPIRVTargetMachine *TM) {
  return new SPIRVEmitNonSemanticDI(TM);
}

enum BaseTypeAttributeEncoding {
  Unspecified = 0,
  Address = 1,
  Boolean = 2,
  Float = 3,
  Signed = 4,
  SignedChar = 5,
  Unsigned = 6,
  UnsignedChar = 7
};

enum CompositeTypeAttributeEncoding { Class = 0, Struct = 1, Union = 2 };
enum ImportedEnityAttributeEncoding {
  ImportedModule = 0,
  ImportedDeclaration = 1
};

enum SourceLanguage {
  Unknown = 0,
  ESSL = 1,
  GLSL = 2,
  OpenCL_C = 3,
  OpenCL_CPP = 4,
  HLSL = 5,
  CPP_for_OpenCL = 6,
  SYCL = 7,
  HERO_C = 8,
  NZSL = 9,
  WGSL = 10,
  Slang = 11,
  Zig = 12,
  CPP = 13
};

enum QualifierTypeAttributeEncoding {
  ConstType = 0,
  VolatileType = 1,
  RestrictType = 2,
  AtomicType = 3
};

enum DebugOperation {
  Deref = 0,
  Plus = 1,
  Minus = 2,
  PlusUconst = 3,
  BitPiece = 4,
  Swap = 5,
  Xderef = 6,
  StackValue = 7,
  Constu = 8,
  Fragment = 9
};

enum Flag {
  FlagIsProtected = 1 << 0,
  FlagIsPrivate = 1 << 1,
  FlagIsPublic = FlagIsPrivate | FlagIsProtected,
  FlagAccess = FlagIsPublic,
  FlagIsLocal = 1 << 2,
  FlagIsDefinition = 1 << 3,
  FlagIsFwdDecl = 1 << 4,
  FlagIsArtificial = 1 << 5,
  FlagIsExplicit = 1 << 6,
  FlagIsPrototyped = 1 << 7,
  FlagIsObjectPointer = 1 << 8,
  FlagIsStaticMember = 1 << 9,
  FlagIsIndirectVariable = 1 << 10,
  FlagIsLValueReference = 1 << 11,
  FlagIsRValueReference = 1 << 12,
  FlagIsOptimized = 1 << 13,
  FlagIsEnumClass = 1 << 14,
  FlagTypePassByValue = 1 << 15,
  FlagTypePassByReference = 1 << 16,
  FlagUnknownPhysicalLayout = 1 << 17,
  FlagBitField = 1 << 18
};

struct DebugOpKey {
  DebugOperation Kind;
  SmallVector<uint64_t, 2> Operands;

  bool operator==(const DebugOpKey &Other) const {
    if (Kind != Other.Kind || Operands.size() != Other.Operands.size())
      return false;
    for (size_t i = 0; i < Operands.size(); ++i)
      if (Operands[i] != Other.Operands[i])
        return false;
    return true;
  }
  bool operator<(const DebugOpKey &Other) const {
    if (Kind != Other.Kind)
      return Kind < Other.Kind;
    if (Operands.size() != Other.Operands.size())
      return Operands.size() < Other.Operands.size();
    for (size_t i = 0; i < Operands.size(); ++i) {
      if (Operands[i] != Other.Operands[i])
        return Operands[i] < Other.Operands[i];
    }
    return false;
  }
};

template <typename T, typename Container>
Register findRegisterFromMap(const T *DIType, const Container &RegPairs,
                             Register DefaultReg = Register()) {
  if (!DIType)
    return DefaultReg;
  for (const auto &[DefinedType, Reg] : RegPairs) {
    if (DefinedType == DIType)
      return Reg;
  }
  return DefaultReg;
}

bool SPIRVEmitNonSemanticDI::emitGlobalDI(MachineFunction &MF,
                                          SPIRVCodeGenContext &Ctx,
                                          DebugInfoCollector &Collector) {
  // If this MachineFunction doesn't have any BB repeat procedure
  // for the next
  if (MF.begin() == MF.end()) {
    IsGlobalDIEmitted = false;
    return false;
  }

  [[maybe_unused]] LLVMContext *Context;
  SmallVector<SmallString<128>> FilePaths;
  SmallVector<int64_t> LLVMSourceLanguages;
  int64_t DwarfVersion = 0;
  int64_t DebugInfoVersion = 0;
  SmallString<128> BuildIdentifier;
  SmallString<128> BuildStoragePath;
  Register DebugCompUnitResIdReg;
  Register DebugSourceResIdReg;

  {
    const MachineModuleInfo &MMI =
        getAnalysis<MachineModuleInfoWrapperPass>().getMMI();
    const Module *M = MMI.getModule();
    Context = &M->getContext();
    const NamedMDNode *DbgCu = M->getNamedMetadata("llvm.dbg.cu");
    if (!DbgCu)
      return false;
    for (const auto *Op : DbgCu->operands()) {
      if (const auto *CompileUnit = dyn_cast<DICompileUnit>(Op)) {
        if (CompileUnit->getDWOId())
          BuildIdentifier = std::to_string(CompileUnit->getDWOId());
        if (!CompileUnit->getSplitDebugFilename().empty())
          BuildStoragePath = CompileUnit->getSplitDebugFilename();

        for (auto *GVE : CompileUnit->getGlobalVariables()) {
          if (auto *DIGV = dyn_cast<DIGlobalVariable>(GVE->getVariable())) {
            extractTypeMetadata(DIGV->getType(), Collector);
          }
        }
        for (const auto *IE : CompileUnit->getImportedEntities()) {
          if (const auto *Imported = dyn_cast<DIImportedEntity>(IE)) {
            Collector.ImportedEntities.push_back(Imported);
          }
        }

        DIFile *File = CompileUnit->getFile();
        Collector.CUFileNodes.push_back(File);
        Collector.CUNodes.push_back(CompileUnit);
        FilePaths.emplace_back();
        sys::path::append(FilePaths.back(), File->getDirectory(),
                          File->getFilename());
        LLVMSourceLanguages.push_back(
            CompileUnit->getSourceLanguage().getUnversionedName());
      }
    }
    const NamedMDNode *ModuleFlags = M->getNamedMetadata("llvm.module.flags");
    assert(ModuleFlags && "Expected llvm.module.flags metadata to be present");
    for (const auto *Op : ModuleFlags->operands()) {
      const MDOperand &MaybeStrOp = Op->getOperand(1);
      if (MaybeStrOp.equalsStr("Dwarf Version"))
        DwarfVersion =
            cast<ConstantInt>(
                cast<ConstantAsMetadata>(Op->getOperand(2))->getValue())
                ->getSExtValue();
      else if (MaybeStrOp.equalsStr("Debug Info Version"))
        DebugInfoVersion =
            cast<ConstantInt>(
                cast<ConstantAsMetadata>(Op->getOperand(2))->getValue())
                ->getSExtValue();
    }

    // This traversal is the only supported way to access
    // instruction related DI metadata like DIBasicType
    for (auto &F : *M) {
      if (DISubprogram *SP = F.getSubprogram()) {
        Collector.SubPrograms.insert(SP);
        if (auto *SubType = dyn_cast<DISubroutineType>(SP->getType()))
          Collector.SubRoutineTypes.insert(SubType);
      }
      for (auto &BB : F) {
        for (auto &I : BB) {
          for (DbgVariableRecord &DVR : filterDbgVars(I.getDbgRecordRange())) {
            if (DILocalVariable *LocalVariable = DVR.getVariable()) {
              Collector.LocalVariables.insert(LocalVariable);
              extractTypeMetadata(LocalVariable->getType(), Collector);
            }
            if (const DIExpression *Expr = DVR.getExpression()) {
              if (Collector.SeenExprs.insert(Expr).second) {
                Collector.ExprList.push_back(Expr);
              }
            }
          }
          if (const DebugLoc &DL = I.getDebugLoc()) {
            if (const DILocation *Loc = DL.get()) {
              DIScope *Scope = Loc->getScope();
              if (auto *SP = dyn_cast<DISubprogram>(Scope))
                Collector.SubPrograms.insert(SP);
              else if (isa<DILexicalBlock>(Scope) ||
                       isa<DILexicalBlockFile>(Scope))
                Collector.LexicalScopes.insert(Scope);
            }
            if (DILocation *InlinedAt = DL.getInlinedAt()) {
              Collector.InlinedAtLoc.insert(InlinedAt);
            }
          }
        }
      }
    }
  }
  // NonSemantic.Shader.DebugInfo.100 global DI instruction emitting
  {
    // Required LLVM variables for emitting logic

    const Register DwarfVersionReg = Ctx.GR->buildConstantInt(
        DwarfVersion, Ctx.MIRBuilder, Ctx.I32Ty, false);

    const Register DebugInfoVersionReg = Ctx.GR->buildConstantInt(
        DebugInfoVersion, Ctx.MIRBuilder, Ctx.I32Ty, false);

    for (unsigned Idx = 0; Idx < LLVMSourceLanguages.size(); ++Idx) {
      emitSingleCompilationUnit(
          FilePaths[Idx], LLVMSourceLanguages[Idx], Ctx, DebugInfoVersionReg,
          DwarfVersionReg, DebugSourceResIdReg, DebugCompUnitResIdReg,
          Collector.CUFileNodes[Idx], Collector.CUNodes[Idx]);
    }
    if (const DISubprogram *SP = Ctx.MF.getFunction().getSubprogram()) {
      if (const DIFile *File = SP->getFile())
        Ctx.GR->addDebugValue(File, DebugCompUnitResIdReg);
      if (const DICompileUnit *Unit = SP->getUnit())
        Ctx.GR->addDebugValue(Unit, DebugCompUnitResIdReg);
    }

    emitDebugMacroDefs(MF, Ctx);
    emitDebugBuildIdentifier(BuildIdentifier, Ctx);
    emitDebugStoragePath(BuildStoragePath, Ctx);
    emitDebugBasicTypes(Collector.BasicTypes, Ctx);

    emitDebugPointerTypes(Collector.PointerDerivedTypes, Ctx);
    emitDebugQualifiedTypes(Collector.QualifiedDerivedTypes, Ctx);
    emitDebugTypedefs(Collector.TypedefTypes, Ctx);

    emitDebugArrayTypes(Collector.ArrayTypes, Ctx);
    emitAllDebugTypeComposites(Collector.CompositeTypes, Ctx);
    emitDebugTypeEnum(Collector.EnumTypes, Ctx);
    emitAllTemplateDebugInstructions(Collector.CompositeTypesWithTemplates,
                                     Ctx);
    emitSubroutineTypes(Collector.SubRoutineTypes, Ctx);
    emitSubprograms(Collector.SubPrograms, Ctx);
    emitLexicalScopes(Collector.LexicalScopes, Ctx);
    emitDebugTypePtrToMember(Collector.PtrToMemberTypes, Ctx);
    emitAllDebugGlobalVariables(MF, Ctx);
    emitDebugImportedEntities(Collector.ImportedEntities, Ctx);
    emitDebugExpressions(Collector.ExprList, Ctx);
  }
  return true;
}

bool SPIRVEmitNonSemanticDI::runOnModule(llvm::Module &M) {
  bool Changed = false;
  IsGlobalDIEmitted = false;
  auto *MMIWrapper = getAnalysisIfAvailable<MachineModuleInfoWrapperPass>();
  if (!MMIWrapper) {
    return Changed;
  }
  auto &MMI = MMIWrapper->getMMI();
  if (MMI.getModule() != &M) {
    return Changed;
  }
  if (M.begin() == M.end())
    return false;
  IsGlobalDIEmitted = false;
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    MachineFunction *MF = MMI.getMachineFunction(F);
    if (!MF)
      continue;

    DebugInfoCollector Collector;
    MachineRegisterInfo &MRI = MF->getRegInfo();
    MachineBasicBlock &MBB = *MF->begin();
    MachineIRBuilder MIRBuilder(MBB, MBB.getFirstTerminator());

    const SPIRVInstrInfo *TII = TM->getSubtargetImpl()->getInstrInfo();
    const SPIRVRegisterInfo *TRI = TM->getSubtargetImpl()->getRegisterInfo();
    const RegisterBankInfo *RBI = TM->getSubtargetImpl()->getRegBankInfo();
    SPIRVGlobalRegistry *GR = TM->getSubtargetImpl()->getSPIRVGlobalRegistry();

    GR->setCurrentFunc(*MF);

    SPIRVTypeInst VoidTy =
        GR->getOrCreateSPIRVType(Type::getVoidTy(M.getContext()), MIRBuilder,
                                 SPIRV::AccessQualifier::ReadWrite, false);
    SPIRVTypeInst I32Ty =
        GR->getOrCreateSPIRVType(Type::getInt32Ty(M.getContext()), MIRBuilder,
                                 SPIRV::AccessQualifier::ReadWrite, false);
    const Register I32ZeroReg =
        GR->buildConstantInt(0, MIRBuilder, I32Ty, false, false);

    SmallVector<std::pair<const DIBasicType *const, const Register>, 24>
        BasicTypeRegPairs;
    SmallVector<std::pair<const DICompositeType *const, const Register>, 24>
        CompositeTypeRegPairs;
    SmallVector<std::pair<const DIFile *const, const Register>, 24>
        SourceRegPairs;
    SmallVector<std::pair<const DIScope *const, const Register>, 24>
        ScopeRegPairs;
    SmallVector<std::pair<const DISubroutineType *const, const Register>, 24>
        SubRoutineTypeRegPairs;
    SmallVector<std::pair<const DIFile *const, const Register>, 24>
        CompileUnitRegPairs;
    SmallVector<std::pair<const DIDerivedType *const, const Register>, 24>
        PointerRegPairs;
    SmallVector<std::pair<const DILocation *, Register>, 24> InlinedAtRegPairs;
    SmallVector<std::pair<const DIExpression *, Register>, 12> ExprRegPairs;

    SPIRVCodeGenContext Ctx(MIRBuilder, MRI, GR, VoidTy, I32Ty, TII, TRI, RBI,
                            *MF, I32ZeroReg, TM, SourceRegPairs, ScopeRegPairs,
                            SubRoutineTypeRegPairs, BasicTypeRegPairs,
                            CompositeTypeRegPairs, CompileUnitRegPairs,
                            PointerRegPairs, InlinedAtRegPairs, ExprRegPairs);
    emitGlobalDI(*MF, Ctx, Collector);
    emitDebugLocalVariables(Ctx, Collector.LocalVariables);
    emitDebugLineAndScopeInstructions(Ctx);
  }
  return Changed;
}

void SPIRVEmitNonSemanticDI::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineModuleInfoWrapperPass>();
  AU.setPreservesAll();
}

Register SPIRVEmitNonSemanticDI::EmitOpString(StringRef SR,
                                              SPIRVCodeGenContext &Ctx) {
  const Register StrReg = Ctx.MRI.createVirtualRegister(&SPIRV::IDRegClass);
  Ctx.MRI.setType(StrReg, LLT::scalar(32));
  MachineInstrBuilder MIB = Ctx.MIRBuilder.buildInstr(SPIRV::OpString);
  MIB.addDef(StrReg);
  addStringImm(SR, MIB);
  return StrReg;
}

Register SPIRVEmitNonSemanticDI::EmitDIInstruction(
    SPIRV::NonSemanticExtInst::NonSemanticExtInst Inst,
    ArrayRef<Register> Operands, SPIRVCodeGenContext &Ctx, bool HasForwardRef,
    Register DefReg) {

  Register InstReg = DefReg;
  if (!InstReg.isValid()) {
    InstReg = Ctx.MRI.createVirtualRegister(&SPIRV::IDRegClass);
    Ctx.MRI.setType(InstReg, LLT::scalar(32));
  }

  unsigned Opcode =
      HasForwardRef ? SPIRV::OpExtInstWithForwardRefsKHR : SPIRV::OpExtInst;
  MachineInstrBuilder MIB =
      Ctx.MIRBuilder.buildInstr(Opcode)
          .addDef(InstReg)
          .addUse(Ctx.GR->getSPIRVTypeID(Ctx.VoidTy))
          .addImm(static_cast<int64_t>(
              SPIRV::InstructionSet::NonSemantic_Shader_DebugInfo_100))
          .addImm(Inst);
  for (auto Reg : Operands) {
    MIB.addUse(Reg);
  }
  MIB.constrainAllUses(*Ctx.TII, *Ctx.TRI, *Ctx.RBI);
  Ctx.GR->assignSPIRVTypeToVReg(Ctx.VoidTy, InstReg, Ctx.MF);

  if (DefReg.isValid() && Ctx.GR->isForwardPlaceholder(DefReg))
    Ctx.GR->resolveForwardPlaceholder(DefReg);

  return InstReg;
}

uint32_t SPIRVEmitNonSemanticDI::transDebugFlags(const DINode *DN) {
  uint32_t Flags = 0;
  if (const DIGlobalVariable *GV = dyn_cast<DIGlobalVariable>(DN)) {
    if (GV->isLocalToUnit())
      Flags |= Flag::FlagIsLocal;
    if (GV->isDefinition())
      Flags |= Flag::FlagIsDefinition;
  }
  if (const DISubprogram *DS = dyn_cast<DISubprogram>(DN)) {
    if (DS->isLocalToUnit())
      Flags |= Flag::FlagIsLocal;
    if (DS->isOptimized())
      Flags |= Flag::FlagIsOptimized;
    if (DS->isDefinition())
      Flags |= Flag::FlagIsDefinition;
    Flags |= mapDebugFlags(DS->getFlags());
  }
  if (DN->getTag() == dwarf::DW_TAG_reference_type)
    Flags |= Flag::FlagIsLValueReference;
  if (DN->getTag() == dwarf::DW_TAG_rvalue_reference_type)
    Flags |= Flag::FlagIsRValueReference;
  if (const DIType *DT = dyn_cast<DIType>(DN))
    Flags |= mapDebugFlags(DT->getFlags());
  if (const DILocalVariable *DLocVar = dyn_cast<DILocalVariable>(DN))
    Flags |= mapDebugFlags(DLocVar->getFlags());

  return Flags;
}

uint32_t SPIRVEmitNonSemanticDI::mapDebugFlags(DINode::DIFlags DFlags) {
  uint32_t Flags = 0;
  if ((DFlags & DINode::FlagAccessibility) == DINode::FlagPublic)
    Flags |= Flag::FlagIsPublic;
  if ((DFlags & DINode::FlagAccessibility) == DINode::FlagProtected)
    Flags |= Flag::FlagIsProtected;
  if ((DFlags & DINode::FlagAccessibility) == DINode::FlagPrivate)
    Flags |= Flag::FlagIsPrivate;
  if (DFlags & DINode::FlagFwdDecl)
    Flags |= Flag::FlagIsFwdDecl;
  if (DFlags & DINode::FlagArtificial)
    Flags |= Flag::FlagIsArtificial;
  if (DFlags & DINode::FlagExplicit)
    Flags |= Flag::FlagIsExplicit;
  if (DFlags & DINode::FlagPrototyped)
    Flags |= Flag::FlagIsPrototyped;
  if (DFlags & DINode::FlagObjectPointer)
    Flags |= Flag::FlagIsObjectPointer;
  if (DFlags & DINode::FlagStaticMember)
    Flags |= Flag::FlagIsStaticMember;
  if (DFlags & DINode::FlagLValueReference)
    Flags |= Flag::FlagIsLValueReference;
  if (DFlags & DINode::FlagRValueReference)
    Flags |= Flag::FlagIsRValueReference;
  if (DFlags & DINode::FlagTypePassByValue)
    Flags |= Flag::FlagTypePassByValue;
  if (DFlags & DINode::FlagTypePassByReference)
    Flags |= Flag::FlagTypePassByReference;
  if (DFlags & DINode::FlagEnumClass)
    Flags |= Flag::FlagIsEnumClass;
  return Flags;
}

Register SPIRVEmitNonSemanticDI::emitDebugSource(StringRef FilePath,
                                                 StringRef FileContents,
                                                 SPIRVCodeGenContext &Ctx) {
  const Register FilePathStrReg = EmitOpString(FilePath, Ctx);

  if (FileContents.empty()) {
    return EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugSource,
                             {FilePathStrReg}, Ctx, false);
  }
  constexpr size_t MaxNumWords = UINT16_MAX - 2;
  constexpr size_t MaxStrSize = MaxNumWords * 4 - 1;
  std::string RemainingSource = FileContents.str();
  std::string FirstChunk = RemainingSource.substr(0, MaxStrSize);
  const Register FirstTextStrReg = EmitOpString(FirstChunk, Ctx);

  Register DebugSourceResIdReg =
      EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugSource,
                        {FilePathStrReg, FirstTextStrReg}, Ctx, false);

  RemainingSource.erase(0, FirstChunk.size());

  while (!RemainingSource.empty()) {
    std::string NextChunk = RemainingSource.substr(0, MaxStrSize);
    const Register ContinuedStrReg = EmitOpString(NextChunk, Ctx);
    EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugSourceContinued,
                      {ContinuedStrReg}, Ctx, false);
    RemainingSource.erase(0, NextChunk.size());
  }

  return DebugSourceResIdReg;
}

void SPIRVEmitNonSemanticDI::emitDebugTypeEnum(
    const SmallPtrSetImpl<const DICompositeType *> &EnumTypes,
    SPIRVCodeGenContext &Ctx) {
  for (auto *EnumTy : EnumTypes) {
    if (!EnumTy || EnumTy->getTag() != dwarf::DW_TAG_enumeration_type)
      continue;
    if (Register Existing = Ctx.GR->getDebugValue(EnumTy); Existing.isValid())
      continue;
    Register NameStr = EmitOpString(EnumTy->getName(), Ctx);
    bool UnderlyingTypeIsFwd = false;
    Register UnderlyingTypeReg = findBaseTypeRegisterRecursive(
        EnumTy->getBaseType(), Ctx, UnderlyingTypeIsFwd);
    if (!UnderlyingTypeReg.isValid()) {
      UnderlyingTypeReg = EmitDIInstruction(
          SPIRV::NonSemanticExtInst::DebugInfoNone, {}, Ctx, false);
      UnderlyingTypeIsFwd = false;
    }
    Register SourceReg =
        findRegisterFromMap(EnumTy->getFile(), Ctx.SourceRegPairs);
    Register Line = Ctx.GR->buildConstantInt(EnumTy->getLine(), Ctx.MIRBuilder,
                                             Ctx.I32Ty, false);
    Register Column =
        Ctx.GR->buildConstantInt(1, Ctx.MIRBuilder, Ctx.I32Ty, false);

    Register ParentReg = Ctx.GR->getDebugValue(EnumTy->getScope());
    if (!ParentReg.isValid()) {
      llvm::errs() << "Warning: Could not find Parent scope register for Enum: "
                   << EnumTy->getName() << "\n";
      ParentReg = Ctx.GR->getDebugValue(EnumTy->getFile());
    }
    Register Size = Ctx.GR->buildConstantInt(EnumTy->getSizeInBits(),
                                             Ctx.MIRBuilder, Ctx.I32Ty, false);
    uint32_t Flags = transDebugFlags(EnumTy);
    Register FlagsReg =
        Ctx.GR->buildConstantInt(Flags, Ctx.MIRBuilder, Ctx.I32Ty, false);
    SmallVector<Register, 16> EnumOperands;
    for (Metadata *MD : EnumTy->getElements()) {
      if (auto *E = dyn_cast<DIEnumerator>(MD)) {
        Register Val = Ctx.GR->buildConstantInt(
            E->getValue().getZExtValue(), Ctx.MIRBuilder, Ctx.I32Ty, false);
        Register Name = EmitOpString(E->getName(), Ctx);
        EnumOperands.push_back(Val);
        EnumOperands.push_back(Name);
      }
    }
    SmallVector<Register, 12> Ops = {
        NameStr, UnderlyingTypeReg, SourceReg, Line,
        Column,  ParentReg,         Size,      FlagsReg};
    Ops.append(EnumOperands);
    bool HasForwardRef = UnderlyingTypeIsFwd;
    Register DefReg;
    Register Res = EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugTypeEnum,
                                     Ops, Ctx, HasForwardRef, DefReg);
    Ctx.GR->addDebugValue(EnumTy, Res);
    Ctx.CompositeTypeRegPairs.emplace_back(EnumTy, Res);
  }
}

void SPIRVEmitNonSemanticDI::emitSingleCompilationUnit(
    StringRef FilePath, int64_t Language, SPIRVCodeGenContext &Ctx,
    Register DebugInfoVersionReg, Register DwarfVersionReg,
    Register &DebugSourceResIdReg, Register &DebugCompUnitResIdReg,
    const DIFile *FileMDNode, const DICompileUnit *CUNode) {

  std::string FileMDContents;
  if (FileMDNode && FileMDNode->getRawFile() &&
      FileMDNode->getSource().has_value())
    FileMDContents = FileMDNode->getSource().value().str();

  DebugSourceResIdReg = emitDebugSource(FilePath, FileMDContents, Ctx);
  Ctx.SourceRegPairs.emplace_back(FileMDNode, DebugSourceResIdReg);

  SourceLanguage SpirvSourceLanguage = SourceLanguage::Unknown;
  switch (Language) {
  case dwarf::DW_LANG_OpenCL:
    SpirvSourceLanguage = SourceLanguage::OpenCL_C;
    break;
  case dwarf::DW_LANG_OpenCL_CPP:
    SpirvSourceLanguage = SourceLanguage::OpenCL_CPP;
    break;
  case dwarf::DW_LANG_CPP_for_OpenCL:
    SpirvSourceLanguage = SourceLanguage::CPP_for_OpenCL;
    break;
  case dwarf::DW_LANG_GLSL:
    SpirvSourceLanguage = SourceLanguage::GLSL;
    break;
  case dwarf::DW_LANG_HLSL:
    SpirvSourceLanguage = SourceLanguage::HLSL;
    break;
  case dwarf::DW_LANG_SYCL:
    SpirvSourceLanguage = SourceLanguage::SYCL;
    break;
  case dwarf::DW_LANG_Zig:
    SpirvSourceLanguage = SourceLanguage::Zig;
    break;
  case dwarf::DW_LANG_C_plus_plus_14:
    SpirvSourceLanguage = SourceLanguage::CPP;
    break;
  default:
    SpirvSourceLanguage = SourceLanguage::Unknown;
    break;
  }

  const Register SourceLanguageReg = Ctx.GR->buildConstantInt(
      SpirvSourceLanguage, Ctx.MIRBuilder, Ctx.I32Ty, false);

  DebugCompUnitResIdReg =
      EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugCompilationUnit,
                        {DebugInfoVersionReg, DwarfVersionReg,
                         DebugSourceResIdReg, SourceLanguageReg},
                        Ctx, false);
  if (CUNode)
    Ctx.GR->addDebugValue(CUNode, DebugCompUnitResIdReg);
  Ctx.CompileUnitRegPairs.emplace_back(FileMDNode, DebugCompUnitResIdReg);
}

Register SPIRVEmitNonSemanticDI::findEmittedBasicTypeReg(
    const DIType *BaseType,
    const SmallVectorImpl<std::pair<const DIBasicType *const, const Register>>
        &BasicTypeRegPairs) {
  const DIType *Ty = BaseType;

  while (Ty && !isa<DIBasicType>(Ty)) {
    if (auto *Derived = dyn_cast<DIDerivedType>(Ty))
      if (Derived->getBaseType())
        Ty = Derived->getBaseType();
      else
        return Register();
    else
      return Register();
  }

  if (const auto *BT = dyn_cast<DIBasicType>(Ty)) {
    StringRef Name = BT->getName();
    uint64_t Size = BT->getSizeInBits();

    for (const auto &[DefinedBT, Reg] : BasicTypeRegPairs) {
      if (DefinedBT->getName() == Name && DefinedBT->getSizeInBits() == Size)
        return Reg;
    }
  }
  return Register();
}

Register SPIRVEmitNonSemanticDI::findEmittedCompositeTypeReg(
    const DIType *BaseType,
    const SmallVectorImpl<std::pair<const DICompositeType *const,
                                    const Register>> &CompositeTypeRegPairs) {

  StringRef Name = BaseType->getName();
  uint64_t Size = BaseType->getSizeInBits();
  unsigned Tag = BaseType->getTag();

  for (const auto &[DefinedCT, Reg] : CompositeTypeRegPairs) {
    if (DefinedCT->getName() == Name && DefinedCT->getSizeInBits() == Size &&
        DefinedCT->getTag() == Tag)
      return Reg;
  }

  return Register();
}

void SPIRVEmitNonSemanticDI::emitDebugBuildIdentifier(
    StringRef BuildIdentifier, SPIRVCodeGenContext &Ctx) {
  if (BuildIdentifier.empty())
    return;

  const Register BuildIdStrReg = EmitOpString(BuildIdentifier, Ctx);
  uint32_t Flags = 0;
  if (BuildIdentifier.contains("IdentifierPossibleDuplicates"))
    Flags |= (1 << 0);
  const Register FlagsReg =
      Ctx.GR->buildConstantInt(Flags, Ctx.MIRBuilder, Ctx.I32Ty, false, false);

  EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugBuildIdentifier,
                    {BuildIdStrReg, FlagsReg}, Ctx, false);
}

void SPIRVEmitNonSemanticDI::emitDebugStoragePath(StringRef BuildStoragePath,
                                                  SPIRVCodeGenContext &Ctx) {
  if (!BuildStoragePath.empty()) {
    const Register PathStrReg = EmitOpString(BuildStoragePath, Ctx);
    EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugStoragePath, {PathStrReg},
                      Ctx, false);
  }
}

void SPIRVEmitNonSemanticDI::emitDebugBasicTypes(
    const SmallPtrSetImpl<DIBasicType *> &BasicTypes,
    SPIRVCodeGenContext &Ctx) {
  // We need to store pairs because further instructions reference
  // the DIBasicTypes and size will be always small so there isn't
  // need for any kind of map
  for (auto *BasicType : BasicTypes) {
    if (!BasicType)
      continue;
    const Register BasicTypeStrReg = EmitOpString(BasicType->getName(), Ctx);

    const Register ConstIntBitwidthReg = Ctx.GR->buildConstantInt(
        BasicType->getSizeInBits(), Ctx.MIRBuilder, Ctx.I32Ty, false, false);

    uint64_t AttributeEncoding = BaseTypeAttributeEncoding::Unspecified;
    switch (BasicType->getEncoding()) {
    case dwarf::DW_ATE_signed:
      AttributeEncoding = BaseTypeAttributeEncoding::Signed;
      break;
    case dwarf::DW_ATE_unsigned:
      AttributeEncoding = BaseTypeAttributeEncoding::Unsigned;
      break;
    case dwarf::DW_ATE_unsigned_char:
      AttributeEncoding = BaseTypeAttributeEncoding::UnsignedChar;
      break;
    case dwarf::DW_ATE_signed_char:
      AttributeEncoding = BaseTypeAttributeEncoding::SignedChar;
      break;
    case dwarf::DW_ATE_float:
      AttributeEncoding = BaseTypeAttributeEncoding::Float;
      break;
    case dwarf::DW_ATE_boolean:
      AttributeEncoding = BaseTypeAttributeEncoding::Boolean;
      break;
    case dwarf::DW_ATE_address:
      AttributeEncoding = BaseTypeAttributeEncoding::Address;
    }

    const Register AttributeEncodingReg = Ctx.GR->buildConstantInt(
        AttributeEncoding, Ctx.MIRBuilder, Ctx.I32Ty, false, false);

    const Register FlagsReg = Ctx.GR->buildConstantInt(
        transDebugFlags(BasicType), Ctx.MIRBuilder, Ctx.I32Ty, false, false);

    [[maybe_unused]]
    const Register BasicTypeReg = EmitDIInstruction(
        SPIRV::NonSemanticExtInst::DebugTypeBasic,
        {BasicTypeStrReg, ConstIntBitwidthReg, AttributeEncodingReg, FlagsReg},
        Ctx, false);
    Ctx.GR->addDebugValue(BasicType, BasicTypeReg);
    Ctx.BasicTypeRegPairs.emplace_back(BasicType, BasicTypeReg);
  }
}

void SPIRVEmitNonSemanticDI::emitDebugPointerTypes(
    const SmallPtrSetImpl<DIDerivedType *> &PointerDerivedTypes,
    SPIRVCodeGenContext &Ctx) {

  if (PointerDerivedTypes.empty())
    return;

  for (const auto *PointerDerivedType : PointerDerivedTypes) {
    uint64_t DWARFAddrSpace = 0;
    if (PointerDerivedType->getDWARFAddressSpace().has_value())
      DWARFAddrSpace = PointerDerivedType->getDWARFAddressSpace().value();
    else {
      LLVM_DEBUG(dbgs() << "Warning: pointer DI has no DWARF address space; "
                        << "falling back to 0 (generic)\n");
    }
    const Register StorageClassReg = Ctx.GR->buildConstantInt(
        addressSpaceToStorageClass(DWARFAddrSpace, *Ctx.TM->getSubtargetImpl()),
        Ctx.MIRBuilder, Ctx.I32Ty, false);
    const uint32_t Flags = transDebugFlags(PointerDerivedType);
    const Register FlagsReg = Ctx.GR->buildConstantInt(Flags, Ctx.MIRBuilder,
                                                       Ctx.I32Ty, false, false);

    const DIType *BaseTy = PointerDerivedType->getBaseType();
    bool HasForwardRef = false;
    Register BaseTypeReg;

    if (!BaseTy) {
      BaseTypeReg = EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugInfoNone,
                                      {}, Ctx, false);
      HasForwardRef = false;
    } else {
      BaseTypeReg = findBaseTypeRegisterRecursive(BaseTy, Ctx, HasForwardRef);
    }
    if (!BaseTypeReg.isValid()) {
      llvm::errs() << "Warning: Failed to find base type for pointer: "
                   << PointerDerivedType->getName() << "\n";
      BaseTypeReg = EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugInfoNone,
                                      {}, Ctx, false);
      HasForwardRef = false;
    }

    const Register DebugPointerTypeReg = EmitDIInstruction(
        SPIRV::NonSemanticExtInst::DebugTypePointer,
        {BaseTypeReg, StorageClassReg, FlagsReg}, Ctx, HasForwardRef);

    Ctx.GR->addDebugValue(PointerDerivedType, DebugPointerTypeReg);
    Ctx.PointerRegPairs.emplace_back(PointerDerivedType, DebugPointerTypeReg);
  }
}

void SPIRVEmitNonSemanticDI::emitLexicalScopes(
    const SmallPtrSetImpl<DIScope *> &LexicalScopes, SPIRVCodeGenContext &Ctx) {
  for (DIScope *Scope : LexicalScopes) {
    if (!Scope)
      continue;

    SmallVector<Register, 4> Operands;
    Register ScopeSourceReg =
        findRegisterFromMap(Scope->getFile(), Ctx.SourceRegPairs);
    Register ScopeParentReg =
        findRegisterFromMap(Scope->getScope(), Ctx.ScopeRegPairs);

    Operands.push_back(ScopeSourceReg);

    if (auto *LBF = dyn_cast<DILexicalBlockFile>(Scope)) {
      Register Discriminator = Ctx.GR->buildConstantInt(
          LBF->getDiscriminator(), Ctx.MIRBuilder, Ctx.I32Ty, false, false);

      Operands.push_back(Discriminator);
      Operands.push_back(ScopeParentReg);

      EmitDIInstruction(
          SPIRV::NonSemanticExtInst::DebugLexicalBlockDiscriminator, Operands,
          Ctx, false);

    } else if (auto *LB = dyn_cast<DILexicalBlock>(Scope)) {
      Operands.push_back(Ctx.GR->buildConstantInt(LB->getLine(), Ctx.MIRBuilder,
                                                  Ctx.I32Ty, false, false));
      Operands.push_back(Ctx.GR->buildConstantInt(
          LB->getColumn(), Ctx.MIRBuilder, Ctx.I32Ty, false, false));
      Operands.push_back(ScopeParentReg);

      Register LexicalScopeReg = EmitDIInstruction(
          SPIRV::NonSemanticExtInst::DebugLexicalBlock, Operands, Ctx, false);

      Ctx.ScopeRegPairs.emplace_back(Scope, LexicalScopeReg);
    }
  }
}

void SPIRVEmitNonSemanticDI::emitSubroutineTypes(
    const SmallPtrSet<DISubroutineType *, 12> &SubRoutineTypes,
    SPIRVCodeGenContext &Ctx) {

  for (const auto *SubroutineType : SubRoutineTypes) {
    SmallVector<Register, 6> TypeRegs;
    bool HasForwardRef = false;

    const Register FlagsReg =
        Ctx.GR->buildConstantInt(transDebugFlags(SubroutineType),
                                 Ctx.MIRBuilder, Ctx.I32Ty, false, false);

    auto Types = SubroutineType->getTypeArray();
    const DIType *RetTy =
        Types.size() > 0 ? dyn_cast_or_null<DIType>(Types[0]) : nullptr;
    Register RetTypeReg;
    bool RetTyIsFwd = false;
    if (RetTy) {
      RetTypeReg = findBaseTypeRegisterRecursive(RetTy, Ctx, RetTyIsFwd);
    }
    if (!RetTypeReg.isValid()) {
      SPIRVTypeInst VoidTy = Ctx.GR->getOrCreateSPIRVType(
          Type::getVoidTy(Ctx.MF.getFunction().getContext()), Ctx.MIRBuilder,
          SPIRV::AccessQualifier::ReadWrite, false);
      RetTypeReg = VoidTy->getOperand(0).getReg();
      RetTyIsFwd = false;
    }
    TypeRegs.push_back(RetTypeReg);
    if (RetTyIsFwd)
      HasForwardRef = true;
    for (unsigned I = 1; I < Types.size(); ++I) {
      if (const DIType *ParamTy = dyn_cast_or_null<DIType>(Types[I])) {
        bool ParamTyIsFwd = false;
        Register ParamTypeReg =
            findBaseTypeRegisterRecursive(ParamTy, Ctx, ParamTyIsFwd);

        if (!ParamTypeReg.isValid()) {
          llvm::errs()
              << "Warning: Could not find type for function parameter.\n";
          SPIRVTypeInst VoidTy = Ctx.GR->getOrCreateSPIRVType(
              Type::getVoidTy(Ctx.MF.getFunction().getContext()),
              Ctx.MIRBuilder, SPIRV::AccessQualifier::ReadWrite, false);
          ParamTypeReg = VoidTy->getOperand(0).getReg();
          ParamTyIsFwd = false;
        }
        TypeRegs.push_back(ParamTypeReg);
        if (ParamTyIsFwd)
          HasForwardRef = true;
      } else {
        SPIRVTypeInst VoidTy = Ctx.GR->getOrCreateSPIRVType(
            Type::getVoidTy(Ctx.MF.getFunction().getContext()), Ctx.MIRBuilder,
            SPIRV::AccessQualifier::ReadWrite, false);
        Register TypeReg = VoidTy->getOperand(0).getReg();
        TypeRegs.push_back(TypeReg);
      }
    }
    Register DefReg = Ctx.GR->getDebugValue(SubroutineType);
    SmallVector<Register, 6> Operands;
    Operands.push_back(FlagsReg);
    Operands.append(TypeRegs);
    const Register FuncTypeReg =
        EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugTypeFunction,
                          Operands, Ctx, HasForwardRef, DefReg);
    Ctx.SubRoutineTypeRegPairs.emplace_back(SubroutineType, FuncTypeReg);
  }
}

void SPIRVEmitNonSemanticDI::emitSubprograms(
    const SmallPtrSet<DISubprogram *, 12> &SubPrograms,
    SPIRVCodeGenContext &Ctx) {

  for (const auto *SubProgram : SubPrograms) {
    SmallVector<Register, 10> Operands;

    Operands.push_back(EmitOpString(SubProgram->getName(), Ctx));
    Operands.push_back(
        findRegisterFromMap(SubProgram->getType(), Ctx.SubRoutineTypeRegPairs));
    Operands.push_back(
        findRegisterFromMap(SubProgram->getFile(), Ctx.SourceRegPairs));
    Operands.push_back(Ctx.GR->buildConstantInt(
        SubProgram->getLine(), Ctx.MIRBuilder, Ctx.I32Ty, false, false));
    Operands.push_back(
        Ctx.GR->buildConstantInt(1, Ctx.MIRBuilder, Ctx.I32Ty, false, false));
    Register CUReg =
        findRegisterFromMap(SubProgram->getFile(), Ctx.CompileUnitRegPairs);
    Operands.push_back(CUReg);
    Operands.push_back(EmitOpString(SubProgram->getLinkageName(), Ctx));
    Operands.push_back(Ctx.GR->buildConstantInt(
        SubProgram->getFlags(), Ctx.MIRBuilder, Ctx.I32Ty, false, false));

    if (!SubProgram->isDefinition()) {
      EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugFunctionDeclaration,
                        Operands, Ctx, false);
    } else {
      Operands.push_back(Ctx.GR->buildConstantInt(
          SubProgram->getScopeLine(), Ctx.MIRBuilder, Ctx.I32Ty, false, false));
      const Register FuncReg = EmitDIInstruction(
          SPIRV::NonSemanticExtInst::DebugFunction, Operands, Ctx, false);

      Ctx.ScopeRegPairs.emplace_back(dynamic_cast<const DIScope *>(SubProgram),
                                     FuncReg);

      const Function &MF_Func = Ctx.MF.getFunction();

      if (SubProgram->describes(&MF_Func) &&
          MF_Func.getCallingConv() == CallingConv::SPIR_KERNEL) {
        emitDebugEntryPoint(FuncReg, CUReg, Ctx);
      }
    }
  }
}

uint32_t SPIRVEmitNonSemanticDI::mapTagToQualifierEncoding(unsigned Tag) {
  switch (Tag) {
  case dwarf::DW_TAG_const_type:
    return QualifierTypeAttributeEncoding::ConstType;
  case dwarf::DW_TAG_volatile_type:
    return QualifierTypeAttributeEncoding::VolatileType;
  case dwarf::DW_TAG_restrict_type:
    return QualifierTypeAttributeEncoding::RestrictType;
  case dwarf::DW_TAG_atomic_type:
    return QualifierTypeAttributeEncoding::AtomicType;
  default:
    llvm_unreachable("Unknown DWARF tag for DebugTypeQualifier");
  }
}

uint32_t SPIRVEmitNonSemanticDI::mapImportedTagToEncoding(
    const DIImportedEntity *Imported) {
  switch (Imported->getTag()) {
  case dwarf::DW_TAG_imported_module:
    return ImportedEnityAttributeEncoding::ImportedModule;
  case dwarf::DW_TAG_imported_declaration:
    return ImportedEnityAttributeEncoding::ImportedDeclaration;
  default:
    llvm_unreachable("Unknown DWARF tag for DebugImportedEntity");
  }
}

void SPIRVEmitNonSemanticDI::emitDebugQualifiedTypes(
    const SmallPtrSetImpl<DIDerivedType *> &QualifiedDerivedTypes,
    SPIRVCodeGenContext &Ctx) {
  if (!QualifiedDerivedTypes.empty()) {
    for (const auto *QualifiedDT : QualifiedDerivedTypes) {
      bool IsForwardRef = false;
      Register BaseTypeReg = findBaseTypeRegisterRecursive(
          QualifiedDT->getBaseType(), Ctx, IsForwardRef);
      if (!BaseTypeReg.isValid()) {
        llvm::errs()
            << "Warning: Could not find base type for DebugTypeQualifier.\n";
        BaseTypeReg = EmitDIInstruction(
            SPIRV::NonSemanticExtInst::DebugInfoNone, {}, Ctx, false);
        IsForwardRef = false;
      }
      const uint32_t QualifierValue =
          mapTagToQualifierEncoding(QualifiedDT->getTag());
      const Register QualifierConstReg = Ctx.GR->buildConstantInt(
          QualifierValue, Ctx.MIRBuilder, Ctx.I32Ty, false, false);

      Register DefReg;

      const Register DebugQualifiedTypeReg = EmitDIInstruction(
          SPIRV::NonSemanticExtInst::DebugTypeQualifier,
          {BaseTypeReg, QualifierConstReg}, Ctx, IsForwardRef, DefReg);

      Ctx.GR->addDebugValue(QualifiedDT, DebugQualifiedTypeReg);
    }
  }
}

void SPIRVEmitNonSemanticDI::emitDebugTypedefs(
    const SmallPtrSetImpl<DIDerivedType *> &TypedefTypes,
    SPIRVCodeGenContext &Ctx) {
  for (const auto *TypedefDT : TypedefTypes) {
    bool HasForwardRef = false;
    Register BaseTypeReg = findBaseTypeRegisterRecursive(
        TypedefDT->getBaseType(), Ctx, HasForwardRef);
    if (!BaseTypeReg.isValid()) {
      llvm::errs() << "Warning: Could not find base type for Typedef: "
                   << TypedefDT->getName() << "\n";
      BaseTypeReg = EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugInfoNone,
                                      {}, Ctx, false);
      HasForwardRef = false;
    }
    Register DefReg;
    Register DebugSourceReg =
        findRegisterFromMap(TypedefDT->getFile(), Ctx.SourceRegPairs);
    const Register TypedefNameReg = EmitOpString(TypedefDT->getName(), Ctx);
    const Register LineReg = Ctx.GR->buildConstantInt(
        TypedefDT->getLine(), Ctx.MIRBuilder, Ctx.I32Ty, false, false);
    const Register ColumnReg =
        Ctx.GR->buildConstantInt(1, Ctx.MIRBuilder, Ctx.I32Ty, false, false);
    Register ScopeReg = Ctx.GR->getDebugValue(TypedefDT->getFile());
    const Register DebugTypedefReg =
        EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugTypedef,
                          {TypedefNameReg, BaseTypeReg, DebugSourceReg, LineReg,
                           ColumnReg, ScopeReg},
                          Ctx, HasForwardRef, DefReg);
    Ctx.GR->addDebugValue(TypedefDT, DebugTypedefReg);
  }
}

void SPIRVEmitNonSemanticDI::emitDebugImportedEntities(
    const SmallVectorImpl<const DIImportedEntity *> &ImportedEntities,
    SPIRVCodeGenContext &Ctx) {
  for (const auto *Imported : ImportedEntities) {
    if (!Imported->getEntity())
      continue;

    const Register NameStrReg = EmitOpString(Imported->getName(), Ctx);
    const Register DebugSourceReg =
        findRegisterFromMap(Imported->getFile(), Ctx.SourceRegPairs);
    // TODO: Handle Entity as there are no current instructions for DINamespace,
    // so replaced by DebugInfoNone
    const DINode *Entity = Imported->getEntity();
    bool HasForwardRef = false;
    Register EntityReg;
    if (auto *Ty = dyn_cast<DIType>(Entity))
      EntityReg = findBaseTypeRegisterRecursive(Ty, Ctx, HasForwardRef);
    else
      EntityReg = EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugInfoNone,
                                    {}, Ctx, false);
    const Register LineReg = Ctx.GR->buildConstantInt(
        Imported->getLine(), Ctx.MIRBuilder, Ctx.I32Ty, false, false);
    const Register ColumnReg =
        Ctx.GR->buildConstantInt(1, Ctx.MIRBuilder, Ctx.I32Ty, false, false);
    Register ParentReg = Ctx.GR->getDebugValue(Imported->getScope());
    if (!ParentReg.isValid()) {
      llvm::errs() << "Warning: Could not find Parent scope register for Enum: "
                   << Imported->getName() << "\n";
      ParentReg = Ctx.GR->getDebugValue(Imported->getFile());
    }
    uint32_t Tag = mapImportedTagToEncoding(Imported);
    Register TagReg =
        Ctx.GR->buildConstantInt(Tag, Ctx.MIRBuilder, Ctx.I32Ty, false, false);

    [[maybe_unused]]
    const Register DebugImportedEntityReg =
        EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugImportedEntity,
                          {NameStrReg, TagReg, DebugSourceReg, EntityReg,
                           LineReg, ColumnReg, ParentReg},
                          Ctx, false);
  }
}

void SPIRVEmitNonSemanticDI::emitAllDebugGlobalVariables(
    MachineFunction &MF, SPIRVCodeGenContext &Ctx) {

  const DISubprogram *SP = MF.getFunction().getSubprogram();
  if (!SP)
    return;

  const DICompileUnit *CU = SP->getUnit();
  if (!CU)
    return;

  auto GlobalVars = CU->getGlobalVariables();

  for (auto *GVE : GlobalVars) {
    if (GVE)
      emitDebugGlobalVariable(GVE, Ctx);
  }
}

Register SPIRVEmitNonSemanticDI::emitDebugGlobalVariable(
    const DIGlobalVariableExpression *GVE, SPIRVCodeGenContext &Ctx) {

  const DIGlobalVariable *DIGV = GVE->getVariable();
  StringRef Name = DIGV->getName();
  StringRef LinkageName = DIGV->getLinkageName();
  unsigned Line = DIGV->getLine();
  unsigned Column = 1;
  const DIScope *ParentScope = DIGV->getFile();
  uint32_t Flags = transDebugFlags(DIGV);
  Register NameStrReg = EmitOpString(Name, Ctx);
  Register LinkageStrReg = EmitOpString(LinkageName, Ctx);
  Register LineReg =
      Ctx.GR->buildConstantInt(Line, Ctx.MIRBuilder, Ctx.I32Ty, false, false);
  Register ColumnReg =
      Ctx.GR->buildConstantInt(Column, Ctx.MIRBuilder, Ctx.I32Ty, false, false);
  Register FlagsReg =
      Ctx.GR->buildConstantInt(Flags, Ctx.MIRBuilder, Ctx.I32Ty, false, false);

  const DIType *Ty = DIGV->getType();
  bool HasForwardRef = false;
  Register TypeReg = findBaseTypeRegisterRecursive(Ty, Ctx, HasForwardRef);
  if (!TypeReg.isValid()) {
    llvm::errs() << "Warning: Could not find type for Global Variable: " << Name
                 << "\n";
    TypeReg = EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugInfoNone, {},
                                Ctx, false);
    HasForwardRef = false;
  }
  Register DebugSourceReg =
      findRegisterFromMap(DIGV->getFile(), Ctx.SourceRegPairs);
  Register ParentReg;
  if (ParentScope) {
    ParentReg = Ctx.GR->getDebugValue(ParentScope);
  }
  // TODO: Handle Variable Location operand
  Register VariableReg = EmitDIInstruction(
      SPIRV::NonSemanticExtInst::DebugInfoNone, {}, Ctx, false);

  SmallVector<Register, 9> Ops = {NameStrReg,    TypeReg,     DebugSourceReg,
                                  LineReg,       ColumnReg,   ParentReg,
                                  LinkageStrReg, VariableReg, FlagsReg};

  Register DefReg;

  Register GlobalVarReg =
      EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugGlobalVariable, Ops,
                        Ctx, HasForwardRef, DefReg);

  Ctx.GR->addDebugValue(DIGV, GlobalVarReg);
  return GlobalVarReg;
}

void SPIRVEmitNonSemanticDI::emitDebugArrayTypes(
    const SmallPtrSetImpl<DICompositeType *> &ArrayTypes,
    SPIRVCodeGenContext &Ctx) {
  for (auto *ArrayTy : ArrayTypes) {
    DIType *ElementType = ArrayTy->getBaseType();
    bool HasForwardRef = false;
    Register BaseTypeReg =
        findBaseTypeRegisterRecursive(ElementType, Ctx, HasForwardRef);
    if (!BaseTypeReg.isValid()) {
      llvm::errs()
          << "Warning: Could not find element type for Array/Vector.\n";
      BaseTypeReg = EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugInfoNone,
                                      {}, Ctx, false);
      HasForwardRef = false;
    }

    DINodeArray Subranges = ArrayTy->getElements();
    if (ArrayTy->isVector()) {
      assert(Subranges.size() == 1 && "Only 1D vectors supported!");
      emitDebugVectorTypes(ArrayTy, BaseTypeReg, Ctx);
      continue;
    }

    SmallVector<Register, 4> ComponentCountRegs;
    for (Metadata *M : Subranges) {
      if (auto *SR = dyn_cast<DISubrange>(M)) {
        auto CountValUnion = SR->getCount();
        if (auto *CountCI = CountValUnion.dyn_cast<ConstantInt *>()) {
          uint64_t CountVal = CountCI->getZExtValue();
          Register ConstCountReg = Ctx.GR->buildConstantInt(
              CountVal, Ctx.MIRBuilder, Ctx.I32Ty, false);
          ComponentCountRegs.push_back(ConstCountReg);
        } else {
          Register ConstZero = Ctx.GR->buildConstantInt(
              0, Ctx.MIRBuilder, Ctx.I32Ty, false, false);
          ComponentCountRegs.push_back(ConstZero);
        }
      }
    }

    SmallVector<Register, 6> Ops;
    Ops.push_back(BaseTypeReg);
    llvm::append_range(Ops, ComponentCountRegs);

    Register DefReg;
    Register LocalFound =
        findEmittedCompositeTypeReg(ArrayTy, Ctx.CompositeTypeRegPairs);

    // If we found it locally, and it's a placeholder, we use it to define it.
    // If we didn't find it locally, we create a new one (DefReg stays 0).
    if (LocalFound.isValid() && Ctx.GR->isForwardPlaceholder(LocalFound)) {
      DefReg = LocalFound;
    }

    Register DebugArrayTypeReg =
        EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugTypeArray, Ops, Ctx,
                          HasForwardRef, DefReg); // Pass corrected DefReg

    Ctx.GR->addDebugValue(ArrayTy, DebugArrayTypeReg);
    Ctx.CompositeTypeRegPairs.emplace_back(ArrayTy, DebugArrayTypeReg);
  }
}

void SPIRVEmitNonSemanticDI::emitDebugVectorTypes(DICompositeType *ArrayTy,
                                                  Register BaseTypeReg,
                                                  SPIRVCodeGenContext &Ctx) {
  DINodeArray Subranges = ArrayTy->getElements();
  Register ComponentCountReg;
  if (auto *SR = dyn_cast<DISubrange>(Subranges[0])) {
    auto CountValUnion = SR->getCount();
    if (auto *CountCI = CountValUnion.dyn_cast<ConstantInt *>()) {
      uint64_t CountVal = CountCI->getZExtValue();
      ComponentCountReg = Ctx.GR->buildConstantInt(CountVal, Ctx.MIRBuilder,
                                                   Ctx.I32Ty, false, false);
    } else {
      ComponentCountReg =
          Ctx.GR->buildConstantInt(0, Ctx.MIRBuilder, Ctx.I32Ty, false, false);
    }
  }

  SmallVector<Register, 4> Ops;
  Ops.push_back(BaseTypeReg);
  Ops.push_back(ComponentCountReg);

  [[maybe_unused]]
  Register DebugVectorTypeReg = EmitDIInstruction(
      SPIRV::NonSemanticExtInst::DebugTypeVector, Ops, Ctx, false);
}

void SPIRVEmitNonSemanticDI::emitAllTemplateDebugInstructions(
    const SmallPtrSetImpl<const DICompositeType *> &TemplatedTypes,
    SPIRVCodeGenContext &Ctx) {
  for (const DICompositeType *CompTy : TemplatedTypes) {
    const DINodeArray TemplateParams = CompTy->getTemplateParams();
    if (TemplateParams.empty())
      continue;
    Register DebugSourceReg =
        findRegisterFromMap(CompTy->getFile(), Ctx.SourceRegPairs);
    Register LineReg = Ctx.GR->buildConstantInt(
        CompTy->getLine(), Ctx.MIRBuilder, Ctx.I32Ty, false, false);
    Register ColumnReg =
        Ctx.GR->buildConstantInt(1, Ctx.MIRBuilder, Ctx.I32Ty, false, false);

    SmallVector<Register, 4> ParamRegs;
    bool HasForwardRef = false;

    for (const auto *MD : TemplateParams) {
      Register TypeReg;
      bool ParamHasForwardRef = false;

      if (auto *TTP = dyn_cast<DITemplateTypeParameter>(MD)) {
        TypeReg = findBaseTypeRegisterRecursive(TTP->getType(), Ctx,
                                                ParamHasForwardRef);
        if (!TypeReg.isValid()) {
          llvm::errs()
              << "Warning: Could not find type for DITemplateTypeParameter: "
              << TTP->getName() << "\n";
          TypeReg = EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugInfoNone,
                                      {}, Ctx, false);
          ParamHasForwardRef = false;
        }

        if (ParamHasForwardRef)
          HasForwardRef = true;

        Register NameStr = EmitOpString(TTP->getName(), Ctx);
        Register NoneReg = EmitDIInstruction(
            SPIRV::NonSemanticExtInst::DebugInfoNone, {}, Ctx, false);

        ParamRegs.push_back(EmitDIInstruction(
            SPIRV::NonSemanticExtInst::DebugTypeTemplateParameter,
            {NameStr, TypeReg, NoneReg, DebugSourceReg, LineReg, ColumnReg},
            Ctx, ParamHasForwardRef));

      } else if (auto *TVP = dyn_cast<DITemplateValueParameter>(MD)) {
        Register NameStr = EmitOpString(TVP->getName(), Ctx);
        TypeReg =
            findEmittedBasicTypeReg(TVP->getType(), Ctx.BasicTypeRegPairs);
        if (!TypeReg.isValid()) {
          bool TVPTypeForwardRef = false;
          TypeReg = findBaseTypeRegisterRecursive(TVP->getType(), Ctx,
                                                  TVPTypeForwardRef);
          if (TVPTypeForwardRef)
            HasForwardRef = true;
        }
        if (!TypeReg.isValid()) {
          llvm::errs()
              << "Warning: Could not find type for DITemplateValueParameter: "
              << TVP->getName() << "\n";
          TypeReg = EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugInfoNone,
                                      {}, Ctx, false);
        }
        int64_t ActualValue = 0;
        if (auto *CAM = dyn_cast_or_null<ConstantAsMetadata>(TVP->getValue())) {
          if (auto *CI = dyn_cast<ConstantInt>(CAM->getValue())) {
            ActualValue = CI->getSExtValue();
          }
        }
        Register ValueReg = Ctx.GR->buildConstantInt(
            ActualValue, Ctx.MIRBuilder, Ctx.I32Ty, false);
        ParamRegs.push_back(EmitDIInstruction(
            SPIRV::NonSemanticExtInst::DebugTypeTemplateParameter,
            {NameStr, TypeReg, ValueReg, DebugSourceReg, LineReg, ColumnReg},
            Ctx, false));
      }
    }

    Register CompositeReg =
        findEmittedCompositeTypeReg(CompTy, Ctx.CompositeTypeRegPairs);
    if (!CompositeReg.isValid()) {
      CompositeReg = Ctx.MRI.createVirtualRegister(&SPIRV::IDRegClass);
      Ctx.MRI.setType(CompositeReg, LLT::scalar(32));
      Ctx.GR->markAsForwardPlaceholder(CompositeReg);
      Ctx.GR->addDebugValue(CompTy, CompositeReg);
      Ctx.CompositeTypeRegPairs.emplace_back(CompTy, CompositeReg);
    } else if (Ctx.GR->isForwardPlaceholder(CompositeReg)) {
      HasForwardRef = true;
    }
    ParamRegs.insert(ParamRegs.begin(), CompositeReg);
    EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugTypeTemplate, ParamRegs,
                      Ctx, HasForwardRef);
  }
}

void SPIRVEmitNonSemanticDI::emitAllDebugTypeComposites(
    const SmallPtrSetImpl<const DICompositeType *> &CompositeTypes,
    SPIRVCodeGenContext &Ctx) {

  for (auto *CT : CompositeTypes) {
    emitDebugTypeComposite(CT, Ctx);
  }
}

void SPIRVEmitNonSemanticDI::emitDebugTypeComposite(
    const DICompositeType *CompTy, SPIRVCodeGenContext &Ctx) {

  if (!CompTy)
    return;

  Register NameStr = EmitOpString(CompTy->getName(), Ctx);
  Register LinkageNameStr = EmitOpString(CompTy->getIdentifier(), Ctx);
  uint32_t Tag = mapTagToCompositeEncoding(CompTy);
  Register Tags =
      Ctx.GR->buildConstantInt(Tag, Ctx.MIRBuilder, Ctx.I32Ty, false, false);
  Register DebugSourceReg =
      findRegisterFromMap(CompTy->getFile(), Ctx.SourceRegPairs);
  Register CURegLocal = Ctx.GR->getDebugValue(CompTy->getFile());
  Register Line = Ctx.GR->buildConstantInt(CompTy->getLine(), Ctx.MIRBuilder,
                                           Ctx.I32Ty, false, false);
  Register Column =
      Ctx.GR->buildConstantInt(1, Ctx.MIRBuilder, Ctx.I32Ty, false, false);
  Register SizeReg = Ctx.GR->buildConstantInt(
      CompTy->getSizeInBits(), Ctx.MIRBuilder, Ctx.I32Ty, false, false);
  uint32_t Flags = transDebugFlags(CompTy);
  Register FlagsReg =
      Ctx.GR->buildConstantInt(Flags, Ctx.MIRBuilder, Ctx.I32Ty, false, false);

  Register DefReg;
  Register LocalFound =
      findEmittedCompositeTypeReg(CompTy, Ctx.CompositeTypeRegPairs);
  if (LocalFound.isValid() && Ctx.GR->isForwardPlaceholder(LocalFound)) {
    DefReg = LocalFound;
  }
  SmallVector<Register, 4> MemberRegs;
  bool HasForwardRef = false;

  for (Metadata *El : CompTy->getElements()) {
    if (auto *DTM = dyn_cast<DIDerivedType>(El)) {
      emitDebugTypeMember(DTM, Ctx, DefReg, MemberRegs, DebugSourceReg);

      if (Ctx.GR->isForwardPlaceholder(MemberRegs.back()))
        HasForwardRef = true;
    }
  }

  SmallVector<Register, 12> Ops = {NameStr,        Tags,    DebugSourceReg,
                                   Line,           Column,  CURegLocal,
                                   LinkageNameStr, SizeReg, FlagsReg};
  Ops.append(MemberRegs);
  Register Res =
      EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugTypeComposite, Ops, Ctx,
                        HasForwardRef, DefReg);
  Ctx.GR->addDebugValue(CompTy, Res);
  Ctx.CompositeTypeRegPairs.emplace_back(CompTy, Res);
}

void SPIRVEmitNonSemanticDI::emitDebugTypeMember(
    const DIDerivedType *Member, SPIRVCodeGenContext &Ctx,
    const Register &CompositeReg, SmallVectorImpl<Register> &MemberRegs,
    Register DebugSourceReg) {

  if (!Member || Member->getTag() != dwarf::DW_TAG_member)
    return;

  Register NameStr = EmitOpString(Member->getName(), Ctx);
  const DIType *Ty = Member->getBaseType();
  Register TypeReg;
  if (isa<DICompositeType>(Ty)) {
    TypeReg = findEmittedCompositeTypeReg(Ty, Ctx.CompositeTypeRegPairs);
  } else {
    TypeReg = findEmittedBasicTypeReg(Ty, Ctx.BasicTypeRegPairs);
  }

  Register LineReg = Ctx.GR->buildConstantInt(Member->getLine(), Ctx.MIRBuilder,
                                              Ctx.I32Ty, false, false);
  Register ColumnReg =
      Ctx.GR->buildConstantInt(1, Ctx.MIRBuilder, Ctx.I32Ty, false, false);
  Register OffsetReg =
      Ctx.GR->buildConstantInt(1
                               /*Member->getOffsetInBits()*/,
                               Ctx.MIRBuilder, Ctx.I32Ty, false, false);
  Register SizeReg = Ctx.GR->buildConstantInt(
      Member->getSizeInBits(), Ctx.MIRBuilder, Ctx.I32Ty, false, false);
  uint32_t Flags = transDebugFlags(Member);
  Register FlagsReg =
      Ctx.GR->buildConstantInt(Flags, Ctx.MIRBuilder, Ctx.I32Ty, false, false);

  SmallVector<Register, 10> Ops = {NameStr, TypeReg,   DebugSourceReg,
                                   LineReg, ColumnReg, OffsetReg,
                                   SizeReg, FlagsReg};

  Register MemberReg = EmitDIInstruction(
      SPIRV::NonSemanticExtInst::DebugTypeMember, Ops, Ctx, false);

  MemberRegs.push_back(MemberReg);
}

uint32_t
SPIRVEmitNonSemanticDI::mapTagToCompositeEncoding(const DICompositeType *CT) {
  switch (CT->getTag()) {
  case dwarf::DW_TAG_structure_type:
    return CompositeTypeAttributeEncoding::Struct;
  case dwarf::DW_TAG_class_type:
    return CompositeTypeAttributeEncoding::Class;
  case dwarf::DW_TAG_union_type:
    return CompositeTypeAttributeEncoding::Union;
  default:
    llvm_unreachable("Unknown DWARF tag for DebugTypeComposite");
  }
}

void SPIRVEmitNonSemanticDI::emitDebugMacroDefs(MachineFunction &MF,
                                                SPIRVCodeGenContext &Ctx) {

  DenseMap<StringRef, Register> MacroDefRegs;
  const DISubprogram *SP = MF.getFunction().getSubprogram();
  if (!SP)
    return;

  const DICompileUnit *CU = SP->getUnit();
  if (!CU)
    return;

  if (!CU || !CU->getMacros())
    return;
  const StringRef FileName =
      CU->getFile() ? CU->getFile()->getFilename() : "<unknown>";

  std::function<void(const MDNode *)> WalkMacroTree;
  WalkMacroTree = [&](const MDNode *Node) {
    if (const auto *Macro = dyn_cast<DIMacro>(Node)) {
      if (Macro->getMacinfoType() == dwarf::DW_MACINFO_define) {
        if (Macro->getLine() == 0)
          return;

        const StringRef Name = Macro->getName();
        const StringRef Value = Macro->getValue();
        const unsigned Line = Macro->getLine();
        const Register SourceStrReg = EmitOpString(FileName, Ctx);
        const Register LineConstReg =
            Ctx.GR->buildConstantInt(Line, Ctx.MIRBuilder, Ctx.I32Ty, false);
        const Register NameStrReg = EmitOpString(Name, Ctx);
        const Register ValueStrReg = EmitOpString(Value, Ctx);

        [[maybe_unused]] const Register DebugMacroDefReg = EmitDIInstruction(
            SPIRV::NonSemanticExtInst::DebugMacroDef,
            {SourceStrReg, LineConstReg, NameStrReg, ValueStrReg}, Ctx, false);
        MacroDefRegs[Macro->getName()] = DebugMacroDefReg;
      } else if (Macro->getMacinfoType() == dwarf::DW_MACINFO_undef) {
        emitDebugMacroUndef(Macro, FileName, Ctx, MacroDefRegs);
      }
    } else if (const auto *MacroFile = dyn_cast<DIMacroFile>(Node)) {
      for (const auto &Child : MacroFile->getElements())
        WalkMacroTree(Child);
    }
  };

  for (const auto &MacroNode : CU->getMacros()->operands()) {
    if (const auto *MD = dyn_cast<MDNode>(MacroNode.get()))
      WalkMacroTree(MD);
  }
}
void SPIRVEmitNonSemanticDI::emitDebugMacroUndef(
    const DIMacro *MacroUndef, StringRef FileName, SPIRVCodeGenContext &Ctx,
    const DenseMap<StringRef, Register> &MacroDefRegs) {

  const StringRef Name = MacroUndef->getName();
  const unsigned Line = MacroUndef->getLine();
  auto It = MacroDefRegs.find(Name);
  if (It == MacroDefRegs.end())
    return;

  Register MacroDefReg = It->second;
  Register SourceStrReg = EmitOpString(FileName, Ctx);
  Register LineConstReg =
      Ctx.GR->buildConstantInt(Line, Ctx.MIRBuilder, Ctx.I32Ty, false);

  [[maybe_unused]] Register MacroUndefReg =
      EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugMacroUndef,
                        {SourceStrReg, LineConstReg, MacroDefReg}, Ctx, false);
}

void SPIRVEmitNonSemanticDI::emitDebugTypePtrToMember(
    const SmallPtrSetImpl<DIDerivedType *> &PtrToMemberTypes,
    SPIRVCodeGenContext &Ctx) {
  if (!PtrToMemberTypes.empty()) {
    for (const auto *PtrToMemberType : PtrToMemberTypes) {
      assert(PtrToMemberType->getTag() == dwarf::DW_TAG_ptr_to_member_type &&
             "emitDebugTypePtrToMember expects DW_TAG_ptr_to_member_type");
      bool OpHasForwardRef = false;
      const DIType *BaseTy = PtrToMemberType->getBaseType();
      bool MemberTypeIsFwd = false;
      Register MemberTypeReg =
          findBaseTypeRegisterRecursive(BaseTy, Ctx, MemberTypeIsFwd);
      if (!MemberTypeReg.isValid()) {
        llvm::errs()
            << "Warning: Could not find Member Type for PtrToMember.\n";
        MemberTypeReg = EmitDIInstruction(
            SPIRV::NonSemanticExtInst::DebugInfoNone, {}, Ctx, false);
        MemberTypeIsFwd = false;
      }
      if (MemberTypeIsFwd)
        OpHasForwardRef = true;
      const DIType *ClassTy = PtrToMemberType->getClassType();
      bool ParentTypeIsFwd = false;
      Register ParentReg =
          findBaseTypeRegisterRecursive(ClassTy, Ctx, ParentTypeIsFwd);
      if (ParentTypeIsFwd)
        OpHasForwardRef = true;
      SmallVector<Register, 3> Ops;
      Ops.push_back(MemberTypeReg);
      Ops.push_back(ParentReg);
      Register DefReg;
      Register PtrToMemberTypeReg =
          EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugTypePtrToMember,
                            Ops, Ctx, OpHasForwardRef, DefReg);

      Ctx.GR->addDebugValue(PtrToMemberType, PtrToMemberTypeReg);
    }
  }
}

void SPIRVEmitNonSemanticDI::emitDebugExpressions(
    const SmallVectorImpl<const DIExpression *> &ExprList,
    SPIRVCodeGenContext &Ctx) {

  for (const DIExpression *Expr : ExprList) {

    SmallVector<Register, 10> OpRegs;
    SmallSet<DebugOpKey, 10> SeenOperations;

    for (unsigned i = 0; i < Expr->getNumElements();) {
      uint64_t Op = Expr->getElement(i++);

      DebugOperation Kind;
      DebugOpKey CurrentOp;
      bool ValidOp = false;

      switch (Op) {
      case dwarf::DW_OP_deref:
        Kind = DebugOperation::Deref;
        ValidOp = true;
        break;

      case dwarf::DW_OP_plus:
        Kind = DebugOperation::Plus;
        ValidOp = true;
        break;

      case dwarf::DW_OP_minus:
        Kind = DebugOperation::Minus;
        ValidOp = true;
        break;

      case dwarf::DW_OP_plus_uconst:
        Kind = DebugOperation::PlusUconst;
        ValidOp = true;
        if (i < Expr->getNumElements())
          CurrentOp.Operands.push_back(Expr->getElement(i++));
        break;

      case dwarf::DW_OP_bit_piece:
        Kind = DebugOperation::BitPiece;
        ValidOp = true;
        if (i + 1 < Expr->getNumElements()) {
          CurrentOp.Operands.push_back(Expr->getElement(i++));
          CurrentOp.Operands.push_back(Expr->getElement(i++));
        }
        break;

      case dwarf::DW_OP_swap:
        Kind = DebugOperation::Swap;
        ValidOp = true;
        break;

      case dwarf::DW_OP_xderef:
        Kind = DebugOperation::Xderef;
        ValidOp = true;
        break;

      case dwarf::DW_OP_stack_value:
        Kind = DebugOperation::StackValue;
        ValidOp = true;
        break;

      case dwarf::DW_OP_constu:
        Kind = DebugOperation::Constu;
        ValidOp = true;
        if (i < Expr->getNumElements())
          CurrentOp.Operands.push_back(Expr->getElement(i++));
        break;

      case dwarf::DW_OP_LLVM_fragment:
        Kind = DebugOperation::Fragment;
        ValidOp = true;
        if (i + 1 < Expr->getNumElements()) {
          CurrentOp.Operands.push_back(Expr->getElement(i++));
          CurrentOp.Operands.push_back(Expr->getElement(i++));
        }
        break;

      default:
        continue;
      }

      if (ValidOp) {
        CurrentOp.Kind = Kind;
        if (SeenOperations.insert(CurrentOp).second) {
          const Register OpKindReg =
              Ctx.GR->buildConstantInt(static_cast<uint64_t>(CurrentOp.Kind),
                                       Ctx.MIRBuilder, Ctx.I32Ty, false);

          SmallVector<Register, 4> Args = {OpKindReg};

          for (const uint64_t Operand : CurrentOp.Operands) {
            const Register OperandReg = Ctx.GR->buildConstantInt(
                Operand, Ctx.MIRBuilder, Ctx.I32Ty, false);
            Args.push_back(OperandReg);
          }

          const Register DebugOpReg = EmitDIInstruction(
              SPIRV::NonSemanticExtInst::DebugOperation, Args, Ctx, false);

          OpRegs.push_back(DebugOpReg);
        }
      }
    }
    [[maybe_unused]] const Register DebugExpressionReg = EmitDIInstruction(
        SPIRV::NonSemanticExtInst::DebugExpression, OpRegs, Ctx, false);

    Ctx.ExprRegPairs.emplace_back(Expr, DebugExpressionReg);
  }
}

void SPIRVEmitNonSemanticDI::handleDerivedType(DIDerivedType *DT,
                                               DebugInfoCollector &Collector) {
  switch (DT->getTag()) {
  case dwarf::DW_TAG_pointer_type:
    Collector.PointerDerivedTypes.insert(DT);
    break;
  case dwarf::DW_TAG_const_type:
  case dwarf::DW_TAG_volatile_type:
  case dwarf::DW_TAG_restrict_type:
  case dwarf::DW_TAG_atomic_type:
    Collector.QualifiedDerivedTypes.insert(DT);
    break;
  case dwarf::DW_TAG_typedef:
    Collector.TypedefTypes.insert(DT);
    break;
  case dwarf::DW_TAG_inheritance:
    Collector.InheritedTypes.insert(DT);
    break;
  case dwarf::DW_TAG_ptr_to_member_type:
    Collector.PtrToMemberTypes.insert(DT);
    break;
  case dwarf::DW_TAG_member:
    break;
  default:
    break;
  }
  extractTypeMetadata(DT->getBaseType(), Collector);
}

void SPIRVEmitNonSemanticDI::handleCompositeType(
    DICompositeType *CT, DebugInfoCollector &Collector) {
  if (!CT->getTemplateParams().empty()) {
    Collector.CompositeTypesWithTemplates.insert(CT);
    for (const auto *MD : CT->getTemplateParams()) {
      if (const auto *TTP = dyn_cast<DITemplateTypeParameter>(MD)) {
        extractTypeMetadata(TTP->getType(), Collector);
      } else if (const auto *TVP = dyn_cast<DITemplateValueParameter>(MD)) {
        extractTypeMetadata(TVP->getType(), Collector);
      }
    }
  }

  if (CT->getTag() == dwarf::DW_TAG_array_type) {
    Collector.ArrayTypes.insert(CT);
  } else if (CT->getTag() == dwarf::DW_TAG_structure_type ||
             CT->getTag() == dwarf::DW_TAG_class_type ||
             CT->getTag() == dwarf::DW_TAG_union_type) {
    Collector.CompositeTypes.insert(CT);
  } else if (CT->getTag() == dwarf::DW_TAG_enumeration_type) {
    Collector.EnumTypes.insert(CT);
  }

  for (Metadata *Element : CT->getElements()) {
    if (auto *Member = dyn_cast<DIDerivedType>(Element)) {
      extractTypeMetadata(Member, Collector);
    } else if (auto *SR = dyn_cast<DISubrange>(Element)) {
      if (auto *CountVar = SR->getCount().dyn_cast<DIVariable *>()) {
        extractTypeMetadata(CountVar->getType(), Collector);
      }
    }
  }
  extractTypeMetadata(CT->getBaseType(), Collector);
}

void SPIRVEmitNonSemanticDI::extractTypeMetadata(
    DIType *Ty, DebugInfoCollector &Collector) {
  if (!Ty)
    return;
  if (!Collector.visitedTypes.insert(Ty).second)
    return;
  if (auto *CT = dyn_cast<DICompositeType>(Ty)) {
    handleCompositeType(CT, Collector);
  } else if (auto *BT = dyn_cast<DIBasicType>(Ty)) {
    Collector.BasicTypes.insert(BT);
  } else if (auto *DT = dyn_cast<DIDerivedType>(Ty)) {
    handleDerivedType(DT, Collector);
  }
}

Register SPIRVEmitNonSemanticDI::findBaseTypeRegisterRecursive(
    const DIType *Ty, SPIRVCodeGenContext &Ctx, bool &IsForwardRef) {

  if (!Ty) {
    IsForwardRef = false;
    return Register(0);
  }
  if (isa<DIBasicType>(Ty)) {
    Register Found = findEmittedBasicTypeReg(Ty, Ctx.BasicTypeRegPairs);
    if (Found.isValid()) {
      IsForwardRef = false;
      return Found;
    }
  }

  if (const auto *DT = dyn_cast<DIDerivedType>(Ty)) {
    for (const auto &[DefinedPtr, Reg] : Ctx.PointerRegPairs) {
      if (DefinedPtr == DT) {
        IsForwardRef = false;
        return Reg;
      }
    }
    return findBaseTypeRegisterRecursive(DT->getBaseType(), Ctx, IsForwardRef);
  }
  if (isa<DICompositeType>(Ty)) {
    Register Found = findEmittedCompositeTypeReg(Ty, Ctx.CompositeTypeRegPairs);
    if (Found.isValid()) {
      IsForwardRef = Ctx.GR->isForwardPlaceholder(Found);
      return Found;
    }
    Register PlaceholderReg = Ctx.MRI.createVirtualRegister(&SPIRV::IDRegClass);
    Ctx.MRI.setType(PlaceholderReg, LLT::scalar(32));
    Ctx.GR->markAsForwardPlaceholder(PlaceholderReg);
    Ctx.GR->addDebugValue(Ty, PlaceholderReg);
    IsForwardRef = true;
    return PlaceholderReg;
  }

  IsForwardRef = false;
  return Register(0);
}

void SPIRVEmitNonSemanticDI::emitDebugLineAndScopeInstructions(
    SPIRVCodeGenContext &Ctx) {
  for (auto &MBB : Ctx.MF) {
    const DIScope *CurrentScope = nullptr;
    const DILocation *CurrentInlinedAt = nullptr;
    MachineInstr *LastHeaderInstr = nullptr;
    for (auto &MI : MBB) {
      unsigned OpCode = MI.getOpcode();
      if (OpCode == SPIRV::OpFunction || OpCode == SPIRV::OpFunctionParameter ||
          OpCode == SPIRV::OpLabel || OpCode == SPIRV::OpVariable) {
        LastHeaderInstr = &MI;
      }
    }
    bool PastHeader = (LastHeaderInstr == nullptr);
    for (auto &MI : MBB) {
      if (!PastHeader) {
        if (&MI == LastHeaderInstr) {
          PastHeader = true;
        }
        continue;
      }
      emitDebugLineAndScope(MI, Ctx, CurrentScope, CurrentInlinedAt);
      if (MI.getOpcode() == TargetOpcode::DBG_VALUE) {
        emitDebugValueInstruction(MI, Ctx);
        continue;
      }
    }
  }
}

Register
SPIRVEmitNonSemanticDI::getOrCreateFileRegister(const DIFile *File,
                                                SPIRVCodeGenContext &Ctx) {
  if (!File)
    return Register();

  for (const auto &Pair : Ctx.SourceRegPairs) {
    if (Pair.first == File)
      return Pair.second;
  }
  MachineBasicBlock &CurrentMBB = Ctx.MIRBuilder.getMBB();
  MachineBasicBlock::iterator CurrentPos = Ctx.MIRBuilder.getInsertPt();
  MachineBasicBlock &EntryMBB = *Ctx.MF.begin();
  if (EntryMBB.empty()) {
    Ctx.MIRBuilder.setInsertPt(EntryMBB, EntryMBB.end());
  } else {
    auto It = EntryMBB.begin();
    while (It != EntryMBB.end() &&
           (It->getOpcode() == SPIRV::OpFunction ||
            It->getOpcode() == SPIRV::OpFunctionParameter)) {
      ++It;
    }
    Ctx.MIRBuilder.setInsertPt(EntryMBB, It);
  }
  SmallString<128> FilePath;
  sys::path::append(FilePath, File->getDirectory(), File->getFilename());
  std::string FileMDContents;
  if (File->getSource().has_value())
    FileMDContents = File->getSource().value().str();
  Register DebugSourceResIdReg = emitDebugSource(FilePath, FileMDContents, Ctx);
  Ctx.SourceRegPairs.emplace_back(File, DebugSourceResIdReg);

  if (CurrentPos != CurrentMBB.end()) {
    Ctx.MIRBuilder.setInsertPt(CurrentMBB, CurrentPos);
  } else {
    Ctx.MIRBuilder.setInsertPt(CurrentMBB, CurrentMBB.end());
  }

  return DebugSourceResIdReg;
}

void SPIRVEmitNonSemanticDI::emitDebugLocalVariables(
    SPIRVCodeGenContext &Ctx,
    SmallPtrSet<DILocalVariable *, 12> LocalVariables) {
  for (auto Variable : LocalVariables) {
    SmallVector<Register, 8> Operands;
    Register NameStr = EmitOpString(Variable->getName(), Ctx);
    Operands.push_back(NameStr);
    Register TypeReg = Ctx.GR->getDebugValue(Variable->getType());
    if (!TypeReg.isValid()) {
      bool IsFwd = false;
      TypeReg = findBaseTypeRegisterRecursive(Variable->getType(), Ctx, IsFwd);
    }
    Operands.push_back(TypeReg);
    Operands.push_back(
        findRegisterFromMap(Variable->getFile(), Ctx.SourceRegPairs));
    Operands.push_back(Ctx.GR->buildConstantInt(
        Variable->getLine(), Ctx.MIRBuilder, Ctx.I32Ty, false, false));
    Operands.push_back(
        Ctx.GR->buildConstantInt(1, Ctx.MIRBuilder, Ctx.I32Ty, false, false));
    Register ScopeReg =
        findRegisterFromMap(Variable->getScope(), Ctx.ScopeRegPairs);
    // 2. Fallback to Global Registry (for scopes emitted in previous
    // functions)
    if (!ScopeReg.isValid()) {
      ScopeReg = Ctx.GR->getDebugValue(Variable->getScope());
    }
    // 3. Fallback for Subprograms if direct scope lookup failed
    if (!ScopeReg.isValid()) {
      if (auto *SP = dyn_cast<DISubprogram>(Variable->getScope())) {
        ScopeReg = Ctx.GR->getDebugValue(SP);
      }
    }
    Operands.push_back(ScopeReg);
    Operands.push_back(Ctx.GR->buildConstantInt(
        transDebugFlags(Variable), Ctx.MIRBuilder, Ctx.I32Ty, false, false));
    if (Variable->getArg()) {
      Operands.push_back(Ctx.GR->buildConstantInt(
          Variable->getArg(), Ctx.MIRBuilder, Ctx.I32Ty, false, false));
    }

    Register VarReg = EmitDIInstruction(
        SPIRV::NonSemanticExtInst::DebugLocalVariable, Operands, Ctx, false);
    Ctx.GR->addDebugValue(Variable, VarReg);
  }
}

Register
SPIRVEmitNonSemanticDI::getOrCreateInlinedAt(const DILocation *Loc,
                                             SPIRVCodeGenContext &Ctx) {
  if (!Loc)
    return Register(0);
  Register ExistingReg = findRegisterFromMap(Loc, Ctx.InlinedAtRegPairs);
  if (ExistingReg.isValid())
    return ExistingReg;
  Register ParentInlinedAt = Register(0);
  if (const DILocation *ParentLoc = Loc->getInlinedAt()) {
    ParentInlinedAt = getOrCreateInlinedAt(ParentLoc, Ctx);
  }
  SmallVector<Register, 4> Operands;
  Operands.push_back(Ctx.GR->buildConstantInt(Loc->getLine(), Ctx.MIRBuilder,
                                              Ctx.I32Ty, false, false));
  Register ScopeReg = findRegisterFromMap(Loc->getScope(), Ctx.ScopeRegPairs);
  if (!ScopeReg.isValid()) {
    ScopeReg = Ctx.GR->getDebugValue(Loc->getScope());
  }
  if (!ScopeReg.isValid()) {
    return Register(0);
  }
  Operands.push_back(ScopeReg);
  if (ParentInlinedAt.isValid()) {
    Operands.push_back(ParentInlinedAt);
  }
  Register LineReg = EmitDIInstruction(
      SPIRV::NonSemanticExtInst::DebugInlinedAt, Operands, Ctx, false);

  Ctx.InlinedAtRegPairs.emplace_back(Loc, LineReg);
  return LineReg;
}

void SPIRVEmitNonSemanticDI::emitDebugLineAndScope(
    MachineInstr &MI, SPIRVCodeGenContext &Ctx, const DIScope *&CurrentScope,
    const DILocation *&CurrentInlinedAt) {
  if (MI.getOpcode() == SPIRV::OpExtInst &&
      MI.getOperand(2).getImm() ==
          SPIRV::InstructionSet::NonSemantic_Shader_DebugInfo_100)
    return;

  Ctx.MIRBuilder.setInsertPt(*MI.getParent(), MI.getIterator());
  DebugLoc DL = MI.getDebugLoc();
  if (!DL) {
    EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugNoLine, {}, Ctx, false);
  } else {
    const DILocation *DIL = DL.get();
    Register Source = getOrCreateFileRegister(DIL->getFile(), Ctx);
    Register LineReg = Ctx.GR->buildConstantInt(DIL->getLine(), Ctx.MIRBuilder,
                                                Ctx.I32Ty, false);
    Register ColReg = Ctx.GR->buildConstantInt(DIL->getColumn(), Ctx.MIRBuilder,
                                               Ctx.I32Ty, false);

    SmallVector<Register, 5> Ops = {Source, LineReg, LineReg, ColReg, ColReg};
    EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugLine, Ops, Ctx, false);
  }

  if (!DL) {
    if (CurrentScope != nullptr || CurrentInlinedAt != nullptr) {
      EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugNoScope, {}, Ctx,
                        false);
      CurrentScope = nullptr;
      CurrentInlinedAt = nullptr;
    }
  } else {
    DIScope *NewScope = DL->getScope();
    DILocation *NewInlinedAt = DL->getInlinedAt();
    if (NewScope != CurrentScope || NewInlinedAt != CurrentInlinedAt) {
      CurrentScope = NewScope;
      CurrentInlinedAt = NewInlinedAt;
      SmallVector<Register, 2> ScopeOps;
      Register DIScopeReg =
          findRegisterFromMap(CurrentScope, Ctx.ScopeRegPairs);
      if (!DIScopeReg.isValid()) {
        DIScopeReg = Ctx.GR->getDebugValue(CurrentScope);
        if (!DIScopeReg.isValid()) {
          if (auto *SP = dyn_cast<DISubprogram>(CurrentScope)) {

            DIScopeReg = Ctx.GR->getDebugValue(SP);
          }
        }
      }
      if (DIScopeReg.isValid()) {
        ScopeOps.push_back(DIScopeReg);
        if (CurrentInlinedAt) {
          Register InlineReg = getOrCreateInlinedAt(CurrentInlinedAt, Ctx);

          if (InlineReg.isValid())
            ScopeOps.push_back(InlineReg);
        }

        EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugScope, ScopeOps, Ctx,
                          false);
      } else {
        EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugNoScope, {}, Ctx,
                          false);
      }
    }
  }
}

void SPIRVEmitNonSemanticDI::emitDebugValueInstruction(
    MachineInstr &MI, SPIRVCodeGenContext &Ctx) {
  if (MI.getOpcode() != TargetOpcode::DBG_VALUE)
    return;

  const DILocalVariable *Var = MI.getDebugVariable();
  const DIExpression *Expr = MI.getDebugExpression();
  if (!Var || !Expr)
    return;

  Register LocalVarReg = Ctx.GR->getDebugValue(Var);
  if (!LocalVarReg.isValid()) {
    return;
  }

  Register ExprReg = EmitDIInstruction(
      SPIRV::NonSemanticExtInst::DebugExpression, {}, Ctx, false);

  const MachineOperand &LocOp = MI.getOperand(0);

  if (LocOp.isImm() || LocOp.isFPImm() || LocOp.isCImm()) {
    Register ConstReg;
    if (LocOp.isImm()) {
      ConstReg = Ctx.GR->buildConstantInt(LocOp.getImm(), Ctx.MIRBuilder,
                                          Ctx.I32Ty, false);
    } else {
      ConstReg = Ctx.GR->buildConstantInt(0, Ctx.MIRBuilder, Ctx.I32Ty, false);
    }
    EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugValue,
                      {LocalVarReg, ConstReg, ExprReg}, Ctx, false);
    return;
  }
  if (LocOp.isReg()) {
    Register Reg = LocOp.getReg();
    if (!Reg.isValid())
      return;

    MachineInstr *DefMI = Ctx.MRI.getVRegDef(Reg);
    bool IsVariable = (DefMI && DefMI->getOpcode() == SPIRV::OpVariable);

    MachineBasicBlock &MBB = Ctx.MIRBuilder.getMBB();
    MachineBasicBlock::iterator OldInsertPt = Ctx.MIRBuilder.getInsertPt();
    if (DefMI && DefMI->getParent() == &MBB) {
      bool DefIsAfter = false;
      for (auto It = MI.getIterator(); It != MBB.end(); ++It) {
        if (&*It == DefMI) {
          DefIsAfter = true;
          break;
        }
      }
      if (DefIsAfter) {
        auto NextIt = std::next(DefMI->getIterator());
        if (NextIt != MBB.end()) {
          Ctx.MIRBuilder.setInsertPt(MBB, NextIt);
        } else {
          Ctx.MIRBuilder.setInsertPt(MBB, MBB.end());
        }
      }
    }

    if (IsVariable) {
      EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugDeclare,
                        {LocalVarReg, Reg, ExprReg}, Ctx, false);
    } else {
      EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugValue,
                        {LocalVarReg, Reg, ExprReg}, Ctx, false);
    }
    Ctx.MIRBuilder.setInsertPt(MBB, OldInsertPt);
  }
}

void SPIRVEmitNonSemanticDI::emitDebugEntryPoint(Register DebugFunctionReg,
                                                 Register DebugCompUnitReg,
                                                 SPIRVCodeGenContext &Ctx) {

  std::string CompilerVersion = "LLVM " LLVM_VERSION_STRING;
  Register CompilerSigReg = EmitOpString(CompilerVersion, Ctx);
  std::string CommandLineArgs = "";
  if (const NamedMDNode *CmdLine =
          Ctx.MF.getFunction().getParent()->getNamedMetadata(
              "llvm.commandline")) {
    for (const auto *Op : CmdLine->operands()) {
      if (Op->getNumOperands() > 0) {
        if (const auto *S = dyn_cast<MDString>(Op->getOperand(0))) {
          if (!CommandLineArgs.empty())
            CommandLineArgs += " ";
          CommandLineArgs += S->getString();
        }
      }
    }
  }
  Register CommandLineArgsReg = EmitOpString(CommandLineArgs, Ctx);
  EmitDIInstruction(
      SPIRV::NonSemanticExtInst::DebugEntryPoint,
      {DebugFunctionReg, DebugCompUnitReg, CompilerSigReg, CommandLineArgsReg},
      Ctx, false);
}
