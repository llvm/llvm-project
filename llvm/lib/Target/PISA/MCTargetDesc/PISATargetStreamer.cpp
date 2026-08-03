//===-- PISATargetStreamer.cpp - PISATargetStreamer class -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/PISATargetStreamer.h"
#include "MCTargetDesc/PISAMCTargetDesc.h"
#include "MCTargetDesc/PISARegEncoder.h"
#include "PISAInstPrinter.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGenTypes/LowLevelType.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/PISAAddrSpace.h"

using namespace llvm;

PISA::StorageSpace PISA::mapAddrSpaceToStorageSpace(unsigned AS) {
  switch (PISAAS::AddressSpace(AS)) {
  case PISAAS::AddressSpace::PRIVATE:
    return PISA::StorageSpace::PRIVATE;
  case PISAAS::AddressSpace::GLOBAL:
    return PISA::StorageSpace::GLOBAL;
  case PISAAS::AddressSpace::CONSTANT:
    return PISA::StorageSpace::CONSTANT;
  case PISAAS::AddressSpace::SHARED:
    return PISA::StorageSpace::SHARED;
  case PISAAS::AddressSpace::GENERIC:
    return PISA::StorageSpace::GENERIC;
  }
  llvm_unreachable("Unexpected address space!");
}

static StringRef getStorageSpaceRepr(PISA::StorageSpace SS) {
  switch (SS) {
  case PISA::StorageSpace::PRIVATE:
    return "private";
  case PISA::StorageSpace::GLOBAL:
    return "global";
  case PISA::StorageSpace::CONSTANT:
    return "const";
  case PISA::StorageSpace::SHARED:
    return "shared";
  case PISA::StorageSpace::GENERIC:
    return "generic";
  }
  llvm_unreachable("Invalid storage space!");
}

static StringRef getParamASRepr(PISAAS::AddressSpace AS) {
  switch (AS) {
  case PISAAS::AddressSpace::GLOBAL:
    return "global";
  case PISAAS::AddressSpace::CONSTANT:
    return "const";
  case PISAAS::AddressSpace::SHARED:
    return "shared";
  case PISAAS::AddressSpace::GENERIC:
    return "generic";
  default:
    break;
  }
  llvm_unreachable("Invalid parameter address space!");
}

StringRef PISA::getLinkageTyName(PISA::LinkageTy Linkage) {
  switch (Linkage) {
  case DEFAULT:
    return "default";
  case EXPORT:
    return "export";
  case IMPORT:
    return "import";
  }
  llvm_unreachable("Invalid linkage type!");
}

static StringRef getCallingConvRepr(CallingConv::ID CC) {
  switch (CC) {
  case CallingConv::PISA_KERNEL:
    return ".kernel";
  default:
    return ".function";
  }
}

static std::string getKernelAttributeRepr(const KernelAttribute &Attr) {

  DenseMap<PISA::KernelAttributeType, StringRef> AvailableMetadataNodeTypes = {
      {PISA::KernelAttributeType::REQD_WORK_GROUP_SIZE,
       ".reqd_work_group_size"},
      {PISA::KernelAttributeType::VEC_TYPE_HINT, ".vec_type_hint"}};

  std::string Result;
  auto It = AvailableMetadataNodeTypes.find(Attr.KernelAttrType);
  if (It != AvailableMetadataNodeTypes.end()) {
    Result = It->second;
  }

  switch (Attr.KernelAttrType) {
  case llvm::PISA::KernelAttributeType::REQD_WORK_GROUP_SIZE: {
    Result += "(";
    const auto &Values = std::get<std::vector<uint32_t>>(Attr.KernelAttrValues);
    if (!Values.empty()) {
      Result +=
          std::accumulate(std::next(Values.begin()), Values.end(),
                          std::to_string(Values.front()),
                          [](const std::string &Acc, uint32_t El) {
                            return Acc + std::string(", ") + std::to_string(El);
                          });
    }
    Result += ")";
  } break;
  case llvm::PISA::KernelAttributeType::VEC_TYPE_HINT: {
    Result += "(";
    const auto &Arg = std::get<std::string>(Attr.KernelAttrValues);
    Result += Arg;
    Result += ")";
  } break;
  }
  return Result;
}

static void emitTypeString(uint32_t TypeSizeInBits, uint32_t NumElts,
                           raw_ostream &OS) {
  assert(NumElts > 0);
  if (NumElts == 1 && TypeSizeInBits == 1) {
    OS << ".pred";
    return;
  }

  if (NumElts > 1)
    OS << ".v" << NumElts;
  OS << "." << TypeSizeInBits << "b";
}

static void emitTypeString(const LLT &T, raw_ostream &OS) {
  auto TypeBitSize = T.getScalarType().getSizeInBits();
  auto NumElts = T.isVector() ? T.getNumElements() : 1;
  emitTypeString(TypeBitSize, NumElts, OS);
}

namespace {

class PISATargetAsmStreamer final : public PISATargetStreamer {
  formatted_raw_ostream &OS;
  const MCAsmInfo *MAI = nullptr;
  void printName(raw_ostream &OS, StringRef Name) {
    PISAInstPrinter::printSymbolName(OS, Name, MAI);
  }

public:
  PISATargetAsmStreamer(MCStreamer &S, formatted_raw_ostream &OS)
      : PISATargetStreamer(S), OS(OS), MAI(&S.getContext().getAsmInfo()) {}
  ~PISATargetAsmStreamer() override;

  StringRef emitHeader(const PISA::HeaderDcl &HD) override {
    if (HD.Version != 0)
      OS << ".version " << (HD.Version / 100) << "." << (HD.Version % 100)
         << ";\n";
    if (!HD.Target.empty())
      OS << ".target " << HD.Target << ";\n";
    return HD.Target;
  }

  void emitGlobalVariable(const PISA::GlobalVariableDcl &GV) override {
    if (GV.Dcl.Linkage != PISA::LinkageTy::DEFAULT)
      OS << "." << getLinkageTyName(GV.Dcl.Linkage) << " ";

    // globals must have .const or .global addrspace
    switch (GV.Dcl.SS) {
    case PISA::StorageSpace::GLOBAL:
      OS << ".global ";
      break;
    case PISA::StorageSpace::CONSTANT:
      OS << ".const ";
      break;
    default:
      llvm_unreachable("Invalid storage space for global variables");
    }

    auto PrintOffset = [](int64_t Val, raw_ostream &OS) {
      // print imm offset only when it's not zero
      if (Val > 0) {
        OS << "+" << format("%" PRId64, Val);
      } else if (Val < 0) {
        if (Val == std::numeric_limits<int64_t>::min())
          OS << "-" << format("%" PRIu64, Val);
        else
          OS << "-" << format("%" PRId64, -Val);
      }
    };

    OS << " .align " << GV.Dcl.Alignment.value() << " ";

    if (!GV.Dcl.Section.empty())
      OS << ".section(\"" << GV.Dcl.Section << "\") ";

    if (!GV.Dcl.HostAccessName.empty())
      OS << ".host_access(\"" << GV.Dcl.HostAccessName << "\") ";

    OS << "@";
    printName(OS, GV.Dcl.Name);

    if (GV.Init.Initializer.empty()) {
      OS << "[" << GV.Dcl.Size << "]";
    } else {
      OS << " = { ";
      for (auto [i, X] : llvm::enumerate(GV.Init.Initializer)) {
        OS << ((i == 0) ? "" : ", ");
        if (auto Iter = GV.Init.Exprs.find(i); Iter != GV.Init.Exprs.end()) {
          auto &Entry = Iter->second;
          if (auto *G = std::get_if<PISA::VariableInit::GlobalExpr>(&Entry)) {
            OS << "." << X.Type.getSizeInBits() << "b ";
            OS << "@" << G->Name;
            PrintOffset(G->Offset, OS);
          } else if (auto *Z = std::get_if<PISA::VariableInit::Zeros>(&Entry)) {
            OS << ".zero " << Z->N;
          } else {
            llvm_unreachable("unknown construct!");
          }
        } else {
          TypeSize SizeInBits = X.Type.getSizeInBits();
          OS << "." << SizeInBits << "b ";
          OS << format_hex(X.Value & maskTrailingOnes<uint64_t>(SizeInBits), 1);
        }
      }
      OS << " }";
    }

    OS << ";\n";
  }

  void emitFunctionSignature(const PISA::FunctionSignature &Sig) override {
    if (Sig.DN.CC != CallingConv::PISA_KERNEL) {
      switch (Sig.DN.Linkage) {
      case PISA::LinkageTy::EXPORT:
        OS << ".export ";
        break;
      case PISA::LinkageTy::IMPORT:
        OS << ".import ";
        break;
      default:
        break;
      }
    }

    OS << getCallingConvRepr(Sig.DN.CC) << " ";

    for (auto KernelAttr : Sig.DN.KernelAttrs) {
      OS << getKernelAttributeRepr(KernelAttr) << " ";
    }

    if (Sig.DN.CC != CallingConv::PISA_KERNEL) {
      if (Sig.DN.RetLLT.isValid())
        emitTypeString(Sig.DN.RetLLT, OS);
      else
        OS << "void";
      OS << " ";
    }

    OS << "@";
    printName(OS, Sig.DN.Name);

    OS << "(";
    if (Sig.DN.CC != CallingConv::PISA_KERNEL) {
      const char *Sep = "";
      for (auto &Param : Sig.FunctionParams) {
        OS << Sep << ".reg ";
        emitTypeString(Param.Ty, OS);
        OS << " " << Param.Prefix << Param.Idx;
        Sep = ", ";
      }
    } else {
      const char *Sep = "";
      unsigned I = 0;
      for (auto &Param : Sig.KernelParams) {
        OS << Sep << ".param[" << Param.Size << "] ";
        if (Param.hasAlign())
          OS << ".align(" << Param.Align << ") ";
        if (Param.hasAS())
          OS << ".addrspace("
              << getParamASRepr(static_cast<PISAAS::AddressSpace>(Param.AS))
              << ") ";
        if (Param.hasPtrAlign())
          OS << ".ptr_align(" << Param.PtrAlign << ") ";
        if (!Param.ArgName.empty())
          OS << "%" << Param.ArgName;
        else
          OS << "%arg" << I;
        I++;
        Sep = ", ";
      }
    }

    OS << ")";

    OS << "\n";
  }

  void emitFunctionDeclaration(const PISA::FunctionDeclaration &Dcl) override {
    assert(Dcl.DN.CC != CallingConv::PISA_KERNEL);

    if (Dcl.DN.Linkage != PISA::LinkageTy::IMPORT) {
      return;
    }

    OS << ".import ";
    OS << getCallingConvRepr(Dcl.DN.CC) << " ";

    if (Dcl.DN.RetLLT.isValid()) {
      emitTypeString(Dcl.DN.RetLLT, OS);
    } else
      OS << "void";

    OS << " @";
    printName(OS, Dcl.DN.Name);

    OS << "(";
    const char *Sep = "";
    for (auto &Param : Dcl.FunctionParams) {
      OS << Sep << ".reg ";
      emitTypeString(Param.Ty, OS);
      Sep = ", ";
    }
    OS << ");";
  }

  void addRegDcl(unsigned NumElts, unsigned EltSize, unsigned RegType,
                 RegNamesRange RegNames, bool MustAdd) override {
    llvm_unreachable("Never used.");
  }

  void emitRegDcls(const PISA::RegDcls &Dcls,
                   const PISA::DataTypes &DTs) override {
    for (auto &[Key, TI] : DTs) {
      // Skip if 0 registers of this type were declared
      if (TI.RegStart == TI.RegCounter)
        continue;

      auto [NumElts, BitWidth, RegType] = Key;

      switch (RegType) {
      case PISA::RegEncoder::PRED:
        OS << "\t.pred ";
        break;
      case PISA::RegEncoder::REG:
        OS << "\t.reg ";
        emitTypeString(BitWidth, NumElts, OS);
        OS << " ";
        break;
      default:
        llvm_unreachable("unknown reg type!");
      }

      // Do not use range syntax if only 5 or fewer registers were declared
      if (TI.RegCounter - TI.RegStart <= 5) {
        for (unsigned I = TI.RegStart; I < TI.RegCounter; ++I) {
          if (I != TI.RegStart)
            OS << ", ";
          OS << TI.Prefix << I;
        }
        OS << ";\n";
      } else {
        OS << TI.Prefix << "<" << TI.RegStart << "~" << (TI.RegCounter)
           << ">;\n";
      }
    }
  }

  void addLocalVariableDecl(const PISA::LocalVariableDcl &V) override {
    OS << "\t." << getStorageSpaceRepr(V.SS) << " ";
    OS << ".align " << V.Alignment.value() << " ";
    OS << "@";
    // If PISA comes from PISA reader StackIndex won't be set, so we fall
    // back and use already parsed name.
    if (V.StackIndex == -1) {
      OS << V.Name;
    } else {
      OS << "R" << V.StackIndex;
    }
    OS << '[' << V.Size << "];\n";
  }

  void emitLocalVariableDcls(const PISA::LocalVariableDcls &Dcls) override {
    for (auto &V : Dcls.Vars)
      addLocalVariableDecl(V);
  }

  void emitFuncBodyStart() override { OS << "{\n"; }
  void emitFuncBodyEnd() override { OS << "}\n"; }

  bool isAsmStreamer() const override { return true; }
  int getCurLine() const override {
    // getLine() in OS is 0 based
    return OS.getLine() + 1;
  }
};

class PISATargetELFStreamer final : public PISATargetStreamer {
public:
  PISATargetELFStreamer(MCStreamer &S) : PISATargetStreamer(S) {}
  ~PISATargetELFStreamer() override;

  StringRef emitHeader(const PISA::HeaderDcl &) override {
    llvm_unreachable("Not implemented yet!");
  }
  void emitGlobalVariable(const PISA::GlobalVariableDcl &) override {
    llvm_unreachable("Not implemented yet!");
  }
  void emitFunctionSignature(const PISA::FunctionSignature &) override {
    llvm_unreachable("Not implemented yet!");
  }
  void emitFunctionDeclaration(const PISA::FunctionDeclaration &) override {
    llvm_unreachable("Not implemented yet!");
  }
  void emitRegDcls(const PISA::RegDcls &, const PISA::DataTypes &) override {
    llvm_unreachable("Not implemented yet!");
  }
  void emitLocalVariableDcls(const PISA::LocalVariableDcls &) override {
    llvm_unreachable("Not implemented yet!");
  }
  void emitFuncBodyStart() override {
    llvm_unreachable("Not implemented yet!");
  }
  void emitFuncBodyEnd() override { llvm_unreachable("Not implemented yet!"); }

  void addRegDcl(unsigned NumElts, unsigned EltSize, unsigned RegType,
                 RegNamesRange RegNames, bool MustAdd = true) override {
    llvm_unreachable("Not implemented yet!");
  }
  void addLocalVariableDecl(const PISA::LocalVariableDcl &) override {
    llvm_unreachable("Not implemented yet!");
  }
};

class PISATargetNullStreamer final : public PISATargetStreamer {
public:
  PISATargetNullStreamer(MCStreamer &S) : PISATargetStreamer(S) {}
  ~PISATargetNullStreamer() override;

  StringRef emitHeader(const PISA::HeaderDcl &) override { return ""; }
  void emitGlobalVariable(const PISA::GlobalVariableDcl &) override {}
  void emitFunctionSignature(const PISA::FunctionSignature &) override {}
  void emitFunctionDeclaration(const PISA::FunctionDeclaration &) override {}
  void emitRegDcls(const PISA::RegDcls &, const PISA::DataTypes &) override {}
  void emitLocalVariableDcls(const PISA::LocalVariableDcls &) override {}
  void emitFuncBodyStart() override {}
  void emitFuncBodyEnd() override {}

  void addRegDcl(unsigned NumElts, unsigned EltSize, unsigned RegType,
                 RegNamesRange RegNames, bool MustAdd = true) override {}
  void addLocalVariableDecl(const PISA::LocalVariableDcl &) override {}
};

} // namespace

PISATargetStreamer::~PISATargetStreamer() = default;
PISATargetAsmStreamer::~PISATargetAsmStreamer() = default;
PISATargetELFStreamer::~PISATargetELFStreamer() = default;
PISATargetNullStreamer::~PISATargetNullStreamer() = default;

MCTargetStreamer *
llvm::createPISAAsmTargetStreamer(MCStreamer &S, formatted_raw_ostream &OS,
                                  MCInstPrinter *InstPrinter) {
  return new PISATargetAsmStreamer(S, OS);
}

MCTargetStreamer *
llvm::createPISAObjectTargetStreamer(MCStreamer &S,
                                     const MCSubtargetInfo &STI) {
  const Triple &TT = STI.getTargetTriple();
  if (TT.isOSBinFormatELF())
    return new PISATargetELFStreamer(S);
  return nullptr;
}

MCTargetStreamer *llvm::createPISANullTargetStreamer(MCStreamer &S) {
  return new PISATargetNullStreamer(S);
}

DataType DataTypes::getTypeFromLLT(LLT Ty) {
  unsigned NumElts = Ty.isScalar() ? 1 : Ty.getNumElements();
  unsigned EltSize = Ty.getScalarSizeInBits();
  unsigned RegType =
      EltSize == 1 ? PISA::RegEncoder::PRED : PISA::RegEncoder::REG;
  return DataType{NumElts, EltSize, RegType};
}

LLT DataTypes::getLLTFromType(const DataType &Ty) {
  if (Ty.NumElts == 1) {
    return LLT::integer(Ty.EltSize);
  }
  return LLT::vector(ElementCount::getFixed(Ty.NumElts), Ty.EltSize);
}

std::string DataTypes::getPrefixFromLLT(LLT Ty) {
  unsigned NumElts = Ty.isScalar() ? 1 : Ty.getNumElements();
  unsigned EltSize = Ty.getScalarSizeInBits();
  std::string Prefix = "%";

  const DenseMap<unsigned, StringRef> VectorPrefixes = {
      {1, ""},   {2, "v2"}, {3, "v3"},   {4, "v4"},   {5, "v5"},  {6, "v6"},
      {7, "v7"}, {8, "v8"}, {16, "v16"}, {32, "v32"}, {64, "v64"}};

  if (auto It = VectorPrefixes.find(NumElts); It != VectorPrefixes.end())
    Prefix += It->second;
  else
    llvm_unreachable("Unsupported PISA vector size");

  const DenseMap<unsigned, StringRef> ScalarPrefixes = {
      {1, "p"},  // Predicate
      {8, "b"},  // Byte
      {16, "h"}, // Half-word
      {32, "w"}, // Word
      {64, "d"}, // Double-word
      {128, "q"} // Quad-word
  };

  if (auto It = ScalarPrefixes.find(EltSize); It != ScalarPrefixes.end())
    Prefix += It->second;
  else
    llvm_unreachable("Unsupported PISA scalar size");
  return Prefix;
}

DataType DataTypes::getTypeFromPrefix(std::string Prefix) {
  auto Result =
      llvm::StringSwitch<std::tuple<unsigned, unsigned, unsigned>>(Prefix)
          .Case("%p", {1, 1, PISA::RegEncoder::PRED})
          .Case("%b", {1, 8, PISA::RegEncoder::REG})
          .Case("%h", {1, 16, PISA::RegEncoder::REG})
          .Case("%w", {1, 32, PISA::RegEncoder::REG})
          .Case("%d", {1, 64, PISA::RegEncoder::REG})
          .Case("%q", {1, 128, PISA::RegEncoder::REG})
          .Case("%v2b", {2, 8, PISA::RegEncoder::REG})
          .Case("%v3b", {3, 8, PISA::RegEncoder::REG})
          .Case("%v4b", {4, 8, PISA::RegEncoder::REG})
          .Case("%v2h", {2, 16, PISA::RegEncoder::REG})
          .Case("%v3h", {3, 16, PISA::RegEncoder::REG})
          .Case("%v4h", {4, 16, PISA::RegEncoder::REG})
          .Case("%v2w", {2, 32, PISA::RegEncoder::REG})
          .Case("%v3w", {3, 32, PISA::RegEncoder::REG})
          .Case("%v4w", {4, 32, PISA::RegEncoder::REG})
          .Case("%v5w", {5, 32, PISA::RegEncoder::REG})
          .Case("%v6w", {6, 32, PISA::RegEncoder::REG})
          .Case("%v7w", {7, 32, PISA::RegEncoder::REG})
          .Case("%v8w", {8, 32, PISA::RegEncoder::REG})
          .Case("%v16w", {16, 32, PISA::RegEncoder::REG})
          .Case("%v32w", {32, 32, PISA::RegEncoder::REG})
          .Case("%v64w", {64, 32, PISA::RegEncoder::REG})
          .Case("%v2d", {2, 64, PISA::RegEncoder::REG})
          .Case("%v3d", {3, 64, PISA::RegEncoder::REG})
          .Case("%v4d", {4, 64, PISA::RegEncoder::REG})
          .Default({0, 0, 0});

  if (std::get<0>(Result) == 0)
    llvm_unreachable("Unknown type prefix");

  auto [NumElts, EltSize, RegType] = Result;
  return DataType{NumElts, EltSize, RegType};
}
