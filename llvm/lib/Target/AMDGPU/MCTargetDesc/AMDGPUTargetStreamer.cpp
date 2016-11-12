//===-- AMDGPUTargetStreamer.cpp - Mips Target Streamer Methods -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides AMDGPU specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUTargetStreamer.h"
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "Utils/AMDKernelCodeTUtils.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/FormattedStream.h"

namespace llvm {
#include "AMDGPUPTNote.h"
}

using namespace llvm;
using namespace llvm::AMDGPU;

AMDGPUTargetStreamer::AMDGPUTargetStreamer(MCStreamer &S)
    : MCTargetStreamer(S) {}

//===----------------------------------------------------------------------===//
// AMDGPUTargetAsmStreamer
//===----------------------------------------------------------------------===//

AMDGPUTargetAsmStreamer::AMDGPUTargetAsmStreamer(MCStreamer &S,
                                                 formatted_raw_ostream &OS)
    : AMDGPUTargetStreamer(S), OS(OS) { }

void
AMDGPUTargetAsmStreamer::EmitDirectiveHSACodeObjectVersion(uint32_t Major,
                                                           uint32_t Minor) {
  OS << "\t.hsa_code_object_version " <<
        Twine(Major) << "," << Twine(Minor) << '\n';
}

void
AMDGPUTargetAsmStreamer::EmitDirectiveHSACodeObjectISA(uint32_t Major,
                                                       uint32_t Minor,
                                                       uint32_t Stepping,
                                                       StringRef VendorName,
                                                       StringRef ArchName) {
  OS << "\t.hsa_code_object_isa " <<
        Twine(Major) << "," << Twine(Minor) << "," << Twine(Stepping) <<
        ",\"" << VendorName << "\",\"" << ArchName << "\"\n";

}

void
AMDGPUTargetAsmStreamer::EmitAMDKernelCodeT(const amd_kernel_code_t &Header) {
  OS << "\t.amd_kernel_code_t\n";
  dumpAmdKernelCode(&Header, OS, "\t\t");
  OS << "\t.end_amd_kernel_code_t\n";
}

void AMDGPUTargetAsmStreamer::EmitAMDGPUSymbolType(StringRef SymbolName,
                                                   unsigned Type) {
  switch (Type) {
    default: llvm_unreachable("Invalid AMDGPU symbol type");
    case ELF::STT_AMDGPU_HSA_KERNEL:
      OS << "\t.amdgpu_hsa_kernel " << SymbolName << '\n' ;
      break;
  }
}

void AMDGPUTargetAsmStreamer::EmitAMDGPUHsaModuleScopeGlobal(
    StringRef GlobalName) {
  OS << "\t.amdgpu_hsa_module_global " << GlobalName << '\n';
}

void AMDGPUTargetAsmStreamer::EmitAMDGPUHsaProgramScopeGlobal(
    StringRef GlobalName) {
  OS << "\t.amdgpu_hsa_program_global " << GlobalName << '\n';
}

//===----------------------------------------------------------------------===//
// AMDGPUTargetELFStreamer
//===----------------------------------------------------------------------===//

AMDGPUTargetELFStreamer::AMDGPUTargetELFStreamer(MCStreamer &S)
    : AMDGPUTargetStreamer(S), Streamer(S) {}

MCELFStreamer &AMDGPUTargetELFStreamer::getStreamer() {
  return static_cast<MCELFStreamer &>(Streamer);
}

void
AMDGPUTargetELFStreamer::EmitDirectiveHSACodeObjectVersion(uint32_t Major,
                                                           uint32_t Minor) {
  MCStreamer &OS = getStreamer();
  MCSectionELF *Note =
      OS.getContext().getELFSection(PT_NOTE::SectionName, ELF::SHT_NOTE,
                                    ELF::SHF_ALLOC);

  auto NameSZ = sizeof(PT_NOTE::NoteName);
  OS.PushSection();
  OS.SwitchSection(Note);
  OS.EmitIntValue(NameSZ, 4);                                     // namesz
  OS.EmitIntValue(8, 4);                                          // descz
  OS.EmitIntValue(PT_NOTE::NT_AMDGPU_HSA_CODE_OBJECT_VERSION, 4); // type
  OS.EmitBytes(StringRef(PT_NOTE::NoteName, NameSZ));             // name
  OS.EmitValueToAlignment(4);
  OS.EmitIntValue(Major, 4);                                      // desc
  OS.EmitIntValue(Minor, 4);
  OS.EmitValueToAlignment(4);
  OS.PopSection();
}

void
AMDGPUTargetELFStreamer::EmitDirectiveHSACodeObjectISA(uint32_t Major,
                                                       uint32_t Minor,
                                                       uint32_t Stepping,
                                                       StringRef VendorName,
                                                       StringRef ArchName) {
  MCStreamer &OS = getStreamer();
  MCSectionELF *Note =
      OS.getContext().getELFSection(PT_NOTE::SectionName, ELF::SHT_NOTE,
                                    ELF::SHF_ALLOC);

  uint16_t VendorNameSize = VendorName.size() + 1;
  uint16_t ArchNameSize = ArchName.size() + 1;
  unsigned DescSZ = sizeof(VendorNameSize) + sizeof(ArchNameSize) +
                    sizeof(Major) + sizeof(Minor) + sizeof(Stepping) +
                    VendorNameSize + ArchNameSize;

  OS.PushSection();
  OS.SwitchSection(Note);
  auto NameSZ = sizeof(PT_NOTE::NoteName);
  OS.EmitIntValue(NameSZ, 4);                              // namesz
  OS.EmitIntValue(DescSZ, 4);                              // descsz
  OS.EmitIntValue(PT_NOTE::NT_AMDGPU_HSA_ISA, 4);          // type
  OS.EmitBytes(StringRef(PT_NOTE::NoteName, NameSZ));      // name
  OS.EmitValueToAlignment(4);
  OS.EmitIntValue(VendorNameSize, 2);                      // desc
  OS.EmitIntValue(ArchNameSize, 2);
  OS.EmitIntValue(Major, 4);
  OS.EmitIntValue(Minor, 4);
  OS.EmitIntValue(Stepping, 4);
  OS.EmitBytes(VendorName);
  OS.EmitIntValue(0, 1); // NULL terminate VendorName
  OS.EmitBytes(ArchName);
  OS.EmitIntValue(0, 1); // NULL terminte ArchName
  OS.EmitValueToAlignment(4);
  OS.PopSection();
}

void
AMDGPUTargetELFStreamer::EmitAMDKernelCodeT(const amd_kernel_code_t &Header) {

  MCStreamer &OS = getStreamer();
  OS.PushSection();
  OS.EmitBytes(StringRef((const char*)&Header, sizeof(Header)));
  OS.PopSection();
}

void AMDGPUTargetELFStreamer::EmitAMDGPUSymbolType(StringRef SymbolName,
                                                   unsigned Type) {
  MCSymbolELF *Symbol = cast<MCSymbolELF>(
      getStreamer().getContext().getOrCreateSymbol(SymbolName));
  Symbol->setType(ELF::STT_AMDGPU_HSA_KERNEL);
}

void AMDGPUTargetELFStreamer::EmitAMDGPUHsaModuleScopeGlobal(
    StringRef GlobalName) {

  MCSymbolELF *Symbol = cast<MCSymbolELF>(
      getStreamer().getContext().getOrCreateSymbol(GlobalName));
  Symbol->setType(ELF::STT_OBJECT);
  Symbol->setBinding(ELF::STB_LOCAL);
}

void AMDGPUTargetELFStreamer::EmitAMDGPUHsaProgramScopeGlobal(
    StringRef GlobalName) {

  MCSymbolELF *Symbol = cast<MCSymbolELF>(
      getStreamer().getContext().getOrCreateSymbol(GlobalName));
  Symbol->setType(ELF::STT_OBJECT);
  Symbol->setBinding(ELF::STB_GLOBAL);
}

void AMDGPUTargetStreamer::emitRuntimeMDIntValue(RuntimeMD::Key K, uint64_t V,
                                                 unsigned Size) {
  auto &S = getStreamer();
  S.EmitIntValue(K, 1);
  S.EmitIntValue(V, Size);
}

void AMDGPUTargetStreamer::emitRuntimeMDStringValue(RuntimeMD::Key K,
                                                    StringRef R) {
  auto &S = getStreamer();
  S.EmitIntValue(K, 1);
  S.EmitIntValue(R.size(), 4);
  S.EmitBytes(R);
}

void AMDGPUTargetStreamer::emitRuntimeMDThreeIntValues(RuntimeMD::Key K,
                                                       MDNode *Node,
                                                       unsigned Size) {
  assert(Node->getNumOperands() == 3);

  auto &S = getStreamer();
  S.EmitIntValue(K, 1);
  for (const MDOperand &Op : Node->operands()) {
    const ConstantInt *CI = mdconst::extract<ConstantInt>(Op);
    S.EmitIntValue(CI->getZExtValue(), Size);
  }
}

void AMDGPUTargetStreamer::emitStartOfRuntimeMetadata(const Module &M) {
  emitRuntimeMDIntValue(RuntimeMD::KeyMDVersion,
                        RuntimeMD::MDVersion << 8 | RuntimeMD::MDRevision, 2);
  if (auto MD = M.getNamedMetadata("opencl.ocl.version")) {
    if (MD->getNumOperands() != 0) {
      auto Node = MD->getOperand(0);
      if (Node->getNumOperands() > 1) {
        emitRuntimeMDIntValue(RuntimeMD::KeyLanguage,
                              RuntimeMD::OpenCL_C, 1);
        uint16_t Major = mdconst::extract<ConstantInt>(Node->getOperand(0))
                         ->getZExtValue();
        uint16_t Minor = mdconst::extract<ConstantInt>(Node->getOperand(1))
                         ->getZExtValue();
        emitRuntimeMDIntValue(RuntimeMD::KeyLanguageVersion,
                              Major * 100 + Minor * 10, 2);
      }
    }
  }

  if (auto MD = M.getNamedMetadata("llvm.printf.fmts")) {
    for (unsigned I = 0; I < MD->getNumOperands(); ++I) {
      auto Node = MD->getOperand(I);
      if (Node->getNumOperands() > 0)
        emitRuntimeMDStringValue(RuntimeMD::KeyPrintfInfo,
            cast<MDString>(Node->getOperand(0))->getString());
    }
  }
}

static std::string getOCLTypeName(Type *Ty, bool Signed) {
  switch (Ty->getTypeID()) {
  case Type::HalfTyID:
    return "half";
  case Type::FloatTyID:
    return "float";
  case Type::DoubleTyID:
    return "double";
  case Type::IntegerTyID: {
    if (!Signed)
      return (Twine('u') + getOCLTypeName(Ty, true)).str();
    unsigned BW = Ty->getIntegerBitWidth();
    switch (BW) {
    case 8:
      return "char";
    case 16:
      return "short";
    case 32:
      return "int";
    case 64:
      return "long";
    default:
      return (Twine('i') + Twine(BW)).str();
    }
  }
  case Type::VectorTyID: {
    VectorType *VecTy = cast<VectorType>(Ty);
    Type *EleTy = VecTy->getElementType();
    unsigned Size = VecTy->getVectorNumElements();
    return (Twine(getOCLTypeName(EleTy, Signed)) + Twine(Size)).str();
  }
  default:
    return "unknown";
  }
}

static RuntimeMD::KernelArg::ValueType getRuntimeMDValueType(
  Type *Ty, StringRef TypeName) {
  switch (Ty->getTypeID()) {
  case Type::HalfTyID:
    return RuntimeMD::KernelArg::F16;
  case Type::FloatTyID:
    return RuntimeMD::KernelArg::F32;
  case Type::DoubleTyID:
    return RuntimeMD::KernelArg::F64;
  case Type::IntegerTyID: {
    bool Signed = !TypeName.startswith("u");
    switch (Ty->getIntegerBitWidth()) {
    case 8:
      return Signed ? RuntimeMD::KernelArg::I8 : RuntimeMD::KernelArg::U8;
    case 16:
      return Signed ? RuntimeMD::KernelArg::I16 : RuntimeMD::KernelArg::U16;
    case 32:
      return Signed ? RuntimeMD::KernelArg::I32 : RuntimeMD::KernelArg::U32;
    case 64:
      return Signed ? RuntimeMD::KernelArg::I64 : RuntimeMD::KernelArg::U64;
    default:
      // Runtime does not recognize other integer types. Report as struct type.
      return RuntimeMD::KernelArg::Struct;
    }
  }
  case Type::VectorTyID:
    return getRuntimeMDValueType(Ty->getVectorElementType(), TypeName);
  case Type::PointerTyID:
    return getRuntimeMDValueType(Ty->getPointerElementType(), TypeName);
  default:
    return RuntimeMD::KernelArg::Struct;
  }
}

static RuntimeMD::KernelArg::AddressSpaceQualifer getRuntimeAddrSpace(
    AMDGPUAS::AddressSpaces A) {
  switch (A) {
  case AMDGPUAS::GLOBAL_ADDRESS:
    return RuntimeMD::KernelArg::Global;
  case AMDGPUAS::CONSTANT_ADDRESS:
    return RuntimeMD::KernelArg::Constant;
  case AMDGPUAS::LOCAL_ADDRESS:
    return RuntimeMD::KernelArg::Local;
  case AMDGPUAS::FLAT_ADDRESS:
    return RuntimeMD::KernelArg::Generic;
  case AMDGPUAS::REGION_ADDRESS:
    return RuntimeMD::KernelArg::Region;
  default:
    return RuntimeMD::KernelArg::Private;
  }
}

void AMDGPUTargetStreamer::emitRuntimeMetadataForKernelArg(const DataLayout &DL,
    Type *T, RuntimeMD::KernelArg::Kind Kind,
    StringRef BaseTypeName, StringRef TypeName,
    StringRef ArgName, StringRef TypeQual, StringRef AccQual) {
  auto &S = getStreamer();

  // Emit KeyArgBegin.
  S.EmitIntValue(RuntimeMD::KeyArgBegin, 1);

  // Emit KeyArgSize and KeyArgAlign.
  emitRuntimeMDIntValue(RuntimeMD::KeyArgSize,
                        DL.getTypeAllocSize(T), 4);
  emitRuntimeMDIntValue(RuntimeMD::KeyArgAlign,
                        DL.getABITypeAlignment(T), 4);
  if (auto PT = dyn_cast<PointerType>(T)) {
    auto ET = PT->getElementType();
    if (PT->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS && ET->isSized())
      emitRuntimeMDIntValue(RuntimeMD::KeyArgPointeeAlign,
                            DL.getABITypeAlignment(ET), 4);
  }

  // Emit KeyArgTypeName.
  if (!TypeName.empty())
    emitRuntimeMDStringValue(RuntimeMD::KeyArgTypeName, TypeName);

  // Emit KeyArgName.
  if (!ArgName.empty())
    emitRuntimeMDStringValue(RuntimeMD::KeyArgName, ArgName);

  // Emit KeyArgIsVolatile, KeyArgIsRestrict, KeyArgIsConst and KeyArgIsPipe.
  SmallVector<StringRef, 1> SplitQ;
  TypeQual.split(SplitQ, " ", -1, false /* Drop empty entry */);

  for (StringRef KeyName : SplitQ) {
    auto Key = StringSwitch<RuntimeMD::Key>(KeyName)
      .Case("volatile", RuntimeMD::KeyArgIsVolatile)
      .Case("restrict", RuntimeMD::KeyArgIsRestrict)
      .Case("const",    RuntimeMD::KeyArgIsConst)
      .Case("pipe",     RuntimeMD::KeyArgIsPipe)
      .Default(RuntimeMD::KeyNull);
    S.EmitIntValue(Key, 1);
  }

  // Emit KeyArgKind.
  emitRuntimeMDIntValue(RuntimeMD::KeyArgKind, Kind, 1);

  // Emit KeyArgValueType.
  emitRuntimeMDIntValue(RuntimeMD::KeyArgValueType,
                        getRuntimeMDValueType(T, BaseTypeName), 2);

  // Emit KeyArgAccQual.
  if (!AccQual.empty()) {
    auto AQ = StringSwitch<RuntimeMD::KernelArg::AccessQualifer>(AccQual)
      .Case("read_only",  RuntimeMD::KernelArg::ReadOnly)
      .Case("write_only", RuntimeMD::KernelArg::WriteOnly)
      .Case("read_write", RuntimeMD::KernelArg::ReadWrite)
      .Default(RuntimeMD::KernelArg::None);
    emitRuntimeMDIntValue(RuntimeMD::KeyArgAccQual, AQ, 1);
  }

  // Emit KeyArgAddrQual.
  if (auto *PT = dyn_cast<PointerType>(T))
    emitRuntimeMDIntValue(RuntimeMD::KeyArgAddrQual,
        getRuntimeAddrSpace(static_cast<AMDGPUAS::AddressSpaces>(
            PT->getAddressSpace())), 1);

  // Emit KeyArgEnd
  S.EmitIntValue(RuntimeMD::KeyArgEnd, 1);
}

void AMDGPUTargetStreamer::emitRuntimeMetadata(const Function &F) {
  if (!F.getMetadata("kernel_arg_type"))
    return;
  auto &S = getStreamer();
  S.EmitIntValue(RuntimeMD::KeyKernelBegin, 1);
  emitRuntimeMDStringValue(RuntimeMD::KeyKernelName, F.getName());

  const DataLayout &DL = F.getParent()->getDataLayout();
  for (auto &Arg : F.args()) {
    unsigned I = Arg.getArgNo();
    Type *T = Arg.getType();
    auto TypeName = dyn_cast<MDString>(F.getMetadata(
        "kernel_arg_type")->getOperand(I))->getString();
    auto BaseTypeName = cast<MDString>(F.getMetadata(
        "kernel_arg_base_type")->getOperand(I))->getString();
    StringRef ArgName;
    if (auto ArgNameMD = F.getMetadata("kernel_arg_name"))
      ArgName = cast<MDString>(ArgNameMD->getOperand(I))->getString();
    auto TypeQual = cast<MDString>(F.getMetadata(
        "kernel_arg_type_qual")->getOperand(I))->getString();
    auto AccQual = cast<MDString>(F.getMetadata(
        "kernel_arg_access_qual")->getOperand(I))->getString();
    RuntimeMD::KernelArg::Kind Kind;
    if (TypeQual.find("pipe") != StringRef::npos)
      Kind = RuntimeMD::KernelArg::Pipe;
    else Kind = StringSwitch<RuntimeMD::KernelArg::Kind>(BaseTypeName)
      .Case("sampler_t", RuntimeMD::KernelArg::Sampler)
      .Case("queue_t",   RuntimeMD::KernelArg::Queue)
      .Cases("image1d_t", "image1d_array_t", "image1d_buffer_t",
             "image2d_t" , "image2d_array_t",  RuntimeMD::KernelArg::Image)
      .Cases("image2d_depth_t", "image2d_array_depth_t",
             "image2d_msaa_t", "image2d_array_msaa_t",
             "image2d_msaa_depth_t",  RuntimeMD::KernelArg::Image)
      .Cases("image2d_array_msaa_depth_t", "image3d_t",
             RuntimeMD::KernelArg::Image)
      .Default(isa<PointerType>(T) ?
                   (T->getPointerAddressSpace() == AMDGPUAS::LOCAL_ADDRESS ?
                   RuntimeMD::KernelArg::DynamicSharedPointer :
                   RuntimeMD::KernelArg::GlobalBuffer) :
                   RuntimeMD::KernelArg::ByValue);
    emitRuntimeMetadataForKernelArg(DL, T,
        Kind, BaseTypeName, TypeName, ArgName, TypeQual, AccQual);
  }

  // Emit hidden kernel arguments for OpenCL kernels.
  if (F.getParent()->getNamedMetadata("opencl.ocl.version")) {
    auto Int64T = Type::getInt64Ty(F.getContext());
    emitRuntimeMetadataForKernelArg(DL, Int64T,
                                    RuntimeMD::KernelArg::HiddenGlobalOffsetX);
    emitRuntimeMetadataForKernelArg(DL, Int64T,
                                    RuntimeMD::KernelArg::HiddenGlobalOffsetY);
    emitRuntimeMetadataForKernelArg(DL, Int64T,
                                    RuntimeMD::KernelArg::HiddenGlobalOffsetZ);
    if (F.getParent()->getNamedMetadata("llvm.printf.fmts")) {
      auto Int8PtrT = Type::getInt8PtrTy(F.getContext(),
          RuntimeMD::KernelArg::Global);
      emitRuntimeMetadataForKernelArg(DL, Int8PtrT,
                                      RuntimeMD::KernelArg::HiddenPrintfBuffer);
    }
  }

  // Emit KeyReqdWorkGroupSize, KeyWorkGroupSizeHint, and KeyVecTypeHint.
  if (auto RWGS = F.getMetadata("reqd_work_group_size")) {
    emitRuntimeMDThreeIntValues(RuntimeMD::KeyReqdWorkGroupSize,
                                RWGS, 4);
  }

  if (auto WGSH = F.getMetadata("work_group_size_hint")) {
    emitRuntimeMDThreeIntValues(RuntimeMD::KeyWorkGroupSizeHint,
                                WGSH, 4);
  }

  if (auto VTH = F.getMetadata("vec_type_hint")) {
    auto TypeName = getOCLTypeName(cast<ValueAsMetadata>(
      VTH->getOperand(0))->getType(), mdconst::extract<ConstantInt>(
      VTH->getOperand(1))->getZExtValue());
    emitRuntimeMDStringValue(RuntimeMD::KeyVecTypeHint, TypeName);
  }

  // Emit KeyKernelEnd
  S.EmitIntValue(RuntimeMD::KeyKernelEnd, 1);
}

void AMDGPUTargetStreamer::emitRuntimeMetadataAsNoteElement(Module &M) {
  auto &S = getStreamer();
  auto &Context = S.getContext();

  auto NameSZ = sizeof(PT_NOTE::NoteName); // Size of note name including trailing null.

  S.PushSection();
  S.SwitchSection(Context.getELFSection(
      PT_NOTE::SectionName, ELF::SHT_NOTE, ELF::SHF_ALLOC));

  // Create two labels to mark the beginning and end of the desc field
  // and a MCExpr to calculate the size of the desc field.
  auto *DescBegin = Context.createTempSymbol();
  auto *DescEnd = Context.createTempSymbol();
  auto *DescSZ = MCBinaryExpr::createSub(
      MCSymbolRefExpr::create(DescEnd, Context),
      MCSymbolRefExpr::create(DescBegin, Context), Context);

  // Emit the note element for runtime metadata.
  // Name and desc should be padded to 4 byte boundary but size of name and
  // desc should not include padding 0's.
  S.EmitIntValue(NameSZ, 4);                                  // namesz
  S.EmitValue(DescSZ, 4);                                     // descz
  S.EmitIntValue(PT_NOTE::NT_AMDGPU_HSA_RUNTIME_METADATA, 4); // type
  S.EmitBytes(StringRef(PT_NOTE::NoteName, NameSZ));          // name
  S.EmitValueToAlignment(4);                                  // padding 0
  S.EmitLabel(DescBegin);
  emitRuntimeMetadata(M);                                     // desc
  S.EmitLabel(DescEnd);
  S.EmitValueToAlignment(4);                                  // padding 0
  S.PopSection();
}

void AMDGPUTargetStreamer::emitRuntimeMetadata(Module &M) {
  emitStartOfRuntimeMetadata(M);
  for (auto &F : M.functions())
    emitRuntimeMetadata(F);
}

