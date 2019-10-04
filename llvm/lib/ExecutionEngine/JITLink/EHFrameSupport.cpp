//===-------- JITLink_EHFrameSupport.cpp - JITLink eh-frame utils ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "EHFrameSupportImpl.h"

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/DynamicLibrary.h"

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {

EHFrameBinaryParser::EHFrameBinaryParser(JITTargetAddress EHFrameAddress,
                                         StringRef EHFrameContent,
                                         unsigned PointerSize,
                                         support::endianness Endianness)
    : EHFrameAddress(EHFrameAddress), EHFrameContent(EHFrameContent),
      PointerSize(PointerSize), EHFrameReader(EHFrameContent, Endianness) {}

Error EHFrameBinaryParser::addToGraph() {
  while (!EHFrameReader.empty()) {
    size_t RecordOffset = EHFrameReader.getOffset();

    LLVM_DEBUG({
      dbgs() << "Processing eh-frame record at "
             << format("0x%016" PRIx64, EHFrameAddress + RecordOffset)
             << " (offset " << RecordOffset << ")\n";
    });

    size_t RecordLength = 0;
    uint32_t RecordLengthField;
    if (auto Err = EHFrameReader.readInteger(RecordLengthField))
      return Err;

    // Process CIE/FDE length/extended-length fields to build the blocks.
    //
    // The value of these fields describe the length of the *rest* of the CIE
    // (not including data up to the end of the field itself) so we have to
    // bump RecordLength to include the data up to the end of the field: 4 bytes
    // for Length, or 12 bytes (4 bytes + 8 bytes) for ExtendedLength.
    if (RecordLengthField == 0) // Length 0 means end of __eh_frame section.
      break;

    // If the regular length field's value is 0xffffffff, use extended length.
    if (RecordLengthField == 0xffffffff) {
      uint64_t ExtendedLengthField;
      if (auto Err = EHFrameReader.readInteger(ExtendedLengthField))
        return Err;
      if (ExtendedLengthField > EHFrameReader.bytesRemaining())
        return make_error<JITLinkError>("CIE record extends past the end of "
                                        "the __eh_frame section");
      if (ExtendedLengthField + 12 > std::numeric_limits<size_t>::max())
        return make_error<JITLinkError>("CIE record too large to process");
      RecordLength = ExtendedLengthField + 12;
    } else {
      if (RecordLengthField > EHFrameReader.bytesRemaining())
        return make_error<JITLinkError>("CIE record extends past the end of "
                                        "the __eh_frame section");
      RecordLength = RecordLengthField + 4;
    }

    LLVM_DEBUG(dbgs() << "  length: " << RecordLength << "\n");

    // Read the CIE Pointer.
    size_t CIEPointerAddress = EHFrameAddress + EHFrameReader.getOffset();
    uint32_t CIEPointer;
    if (auto Err = EHFrameReader.readInteger(CIEPointer))
      return Err;

    // Based on the CIE pointer value, parse this as a CIE or FDE record.
    if (CIEPointer == 0) {
      if (auto Err = processCIE(RecordOffset, RecordLength))
        return Err;
    } else {
      if (auto Err = processFDE(RecordOffset, RecordLength, CIEPointerAddress,
                                CIEPointer))
        return Err;
    }

    EHFrameReader.setOffset(RecordOffset + RecordLength);
  }

  return Error::success();
}

void EHFrameBinaryParser::anchor() {}

Expected<EHFrameBinaryParser::AugmentationInfo>
EHFrameBinaryParser::parseAugmentationString() {
  AugmentationInfo AugInfo;
  uint8_t NextChar;
  uint8_t *NextField = &AugInfo.Fields[0];

  if (auto Err = EHFrameReader.readInteger(NextChar))
    return std::move(Err);

  while (NextChar != 0) {
    switch (NextChar) {
    case 'z':
      AugInfo.AugmentationDataPresent = true;
      break;
    case 'e':
      if (auto Err = EHFrameReader.readInteger(NextChar))
        return std::move(Err);
      if (NextChar != 'h')
        return make_error<JITLinkError>("Unrecognized substring e" +
                                        Twine(NextChar) +
                                        " in augmentation string");
      AugInfo.EHDataFieldPresent = true;
      break;
    case 'L':
    case 'P':
    case 'R':
      *NextField++ = NextChar;
      break;
    default:
      return make_error<JITLinkError>("Unrecognized character " +
                                      Twine(NextChar) +
                                      " in augmentation string");
    }

    if (auto Err = EHFrameReader.readInteger(NextChar))
      return std::move(Err);
  }

  return std::move(AugInfo);
}

Expected<JITTargetAddress> EHFrameBinaryParser::readAbsolutePointer() {
  static_assert(sizeof(JITTargetAddress) == sizeof(uint64_t),
                "Result must be able to hold a uint64_t");
  JITTargetAddress Addr;
  if (PointerSize == 8) {
    if (auto Err = EHFrameReader.readInteger(Addr))
      return std::move(Err);
  } else if (PointerSize == 4) {
    uint32_t Addr32;
    if (auto Err = EHFrameReader.readInteger(Addr32))
      return std::move(Err);
    Addr = Addr32;
  } else
    llvm_unreachable("Pointer size is not 32-bit or 64-bit");
  return Addr;
}

Error EHFrameBinaryParser::processCIE(size_t RecordOffset,
                                      size_t RecordLength) {
  // Use the dwarf namespace for convenient access to pointer encoding
  // constants.
  using namespace dwarf;

  LLVM_DEBUG(dbgs() << "  Record is CIE\n");

  auto &CIESymbol =
      createCIERecord(EHFrameAddress + RecordOffset,
                      EHFrameContent.substr(RecordOffset, RecordLength));

  CIEInformation CIEInfo(CIESymbol);

  uint8_t Version = 0;
  if (auto Err = EHFrameReader.readInteger(Version))
    return Err;

  if (Version != 0x01)
    return make_error<JITLinkError>("Bad CIE version " + Twine(Version) +
                                    " (should be 0x01) in eh-frame");

  auto AugInfo = parseAugmentationString();
  if (!AugInfo)
    return AugInfo.takeError();

  // Skip the EH Data field if present.
  if (AugInfo->EHDataFieldPresent)
    if (auto Err = EHFrameReader.skip(PointerSize))
      return Err;

  // Read and sanity check the code alignment factor.
  {
    uint64_t CodeAlignmentFactor = 0;
    if (auto Err = EHFrameReader.readULEB128(CodeAlignmentFactor))
      return Err;
    if (CodeAlignmentFactor != 1)
      return make_error<JITLinkError>("Unsupported CIE code alignment factor " +
                                      Twine(CodeAlignmentFactor) +
                                      " (expected 1)");
  }

  // Read and sanity check the data alignment factor.
  {
    int64_t DataAlignmentFactor = 0;
    if (auto Err = EHFrameReader.readSLEB128(DataAlignmentFactor))
      return Err;
    if (DataAlignmentFactor != -8)
      return make_error<JITLinkError>("Unsupported CIE data alignment factor " +
                                      Twine(DataAlignmentFactor) +
                                      " (expected -8)");
  }

  // Skip the return address register field.
  if (auto Err = EHFrameReader.skip(1))
    return Err;

  uint64_t AugmentationDataLength = 0;
  if (auto Err = EHFrameReader.readULEB128(AugmentationDataLength))
    return Err;

  uint32_t AugmentationDataStartOffset = EHFrameReader.getOffset();

  uint8_t *NextField = &AugInfo->Fields[0];
  while (uint8_t Field = *NextField++) {
    switch (Field) {
    case 'L': {
      CIEInfo.FDEsHaveLSDAField = true;
      uint8_t LSDAPointerEncoding;
      if (auto Err = EHFrameReader.readInteger(LSDAPointerEncoding))
        return Err;
      if (LSDAPointerEncoding != (DW_EH_PE_pcrel | DW_EH_PE_absptr))
        return make_error<JITLinkError>(
            "Unsupported LSDA pointer encoding " +
            formatv("{0:x2}", LSDAPointerEncoding) + " in CIE at " +
            formatv("{0:x16}", CIESymbol.getAddress()));
      break;
    }
    case 'P': {
      uint8_t PersonalityPointerEncoding = 0;
      if (auto Err = EHFrameReader.readInteger(PersonalityPointerEncoding))
        return Err;
      if (PersonalityPointerEncoding !=
          (DW_EH_PE_indirect | DW_EH_PE_pcrel | DW_EH_PE_sdata4))
        return make_error<JITLinkError>(
            "Unspported personality pointer "
            "encoding " +
            formatv("{0:x2}", PersonalityPointerEncoding) + " in CIE at " +
            formatv("{0:x16}", CIESymbol.getAddress()));
      uint32_t PersonalityPointerAddress;
      if (auto Err = EHFrameReader.readInteger(PersonalityPointerAddress))
        return Err;
      break;
    }
    case 'R': {
      uint8_t FDEPointerEncoding;
      if (auto Err = EHFrameReader.readInteger(FDEPointerEncoding))
        return Err;
      if (FDEPointerEncoding != (DW_EH_PE_pcrel | DW_EH_PE_absptr))
        return make_error<JITLinkError>(
            "Unsupported FDE address pointer "
            "encoding " +
            formatv("{0:x2}", FDEPointerEncoding) + " in CIE at " +
            formatv("{0:x16}", CIESymbol.getAddress()));
      break;
    }
    default:
      llvm_unreachable("Invalid augmentation string field");
    }
  }

  if (EHFrameReader.getOffset() - AugmentationDataStartOffset >
      AugmentationDataLength)
    return make_error<JITLinkError>("Read past the end of the augmentation "
                                    "data while parsing fields");

  assert(!CIEInfos.count(CIESymbol.getAddress()) &&
         "Multiple CIEs recorded at the same address?");
  CIEInfos[CIESymbol.getAddress()] = std::move(CIEInfo);

  return Error::success();
}

Error EHFrameBinaryParser::processFDE(size_t RecordOffset, size_t RecordLength,
                                      JITTargetAddress CIEPointerAddress,
                                      uint32_t CIEPointer) {
  LLVM_DEBUG(dbgs() << "  Record is FDE\n");

  LLVM_DEBUG({
    dbgs() << "  CIE pointer: "
           << format("0x%016" PRIx64, CIEPointerAddress - CIEPointer) << "\n";
  });

  auto CIEInfoItr = CIEInfos.find(CIEPointerAddress - CIEPointer);
  if (CIEInfoItr == CIEInfos.end())
    return make_error<JITLinkError>(
        "FDE at " + formatv("{0:x16}", EHFrameAddress + RecordOffset) +
        " points to non-existant CIE at " +
        formatv("{0:x16}", CIEPointerAddress - CIEPointer));
  auto &CIEInfo = CIEInfoItr->second;

  // Read and sanity check the PC-start pointer and size.
  JITTargetAddress PCBeginAddress = EHFrameAddress + EHFrameReader.getOffset();

  auto PCBeginDelta = readAbsolutePointer();
  if (!PCBeginDelta)
    return PCBeginDelta.takeError();

  JITTargetAddress PCBegin = PCBeginAddress + *PCBeginDelta;
  LLVM_DEBUG({
    dbgs() << "    PC begin: " << format("0x%016" PRIx64, PCBegin) << "\n";
  });

  auto *TargetSymbol = getSymbolAtAddress(PCBegin);

  if (!TargetSymbol)
    return make_error<JITLinkError>("FDE PC-begin " +
                                    formatv("{0:x16}", PCBegin) +
                                    " does not point at symbol");

  if (TargetSymbol->getAddress() != PCBegin)
    return make_error<JITLinkError>(
        "FDE PC-begin " + formatv("{0:x16}", PCBegin) +
        " does not point to start of symbol at " +
        formatv("{0:x16}", TargetSymbol->getAddress()));

  LLVM_DEBUG(dbgs() << "  FDE target: " << *TargetSymbol << "\n");

  // Skip over the PC range size field.
  if (auto Err = EHFrameReader.skip(PointerSize))
    return Err;

  Symbol *LSDASymbol = nullptr;
  JITTargetAddress LSDAAddress = 0;
  if (CIEInfo.FDEsHaveLSDAField) {
    uint64_t AugmentationDataSize;
    if (auto Err = EHFrameReader.readULEB128(AugmentationDataSize))
      return Err;
    if (AugmentationDataSize != PointerSize)
      return make_error<JITLinkError>(
          "Unexpected FDE augmentation data size (expected " +
          Twine(PointerSize) + ", got " + Twine(AugmentationDataSize) +
          ") for FDE at " + formatv("{0:x16}", EHFrameAddress + RecordOffset));
    LSDAAddress = EHFrameAddress + EHFrameReader.getOffset();
    auto LSDADelta = readAbsolutePointer();
    if (!LSDADelta)
      return LSDADelta.takeError();

    JITTargetAddress LSDA = LSDAAddress + *LSDADelta;

    LSDASymbol = getSymbolAtAddress(LSDA);

    if (!LSDASymbol)
      return make_error<JITLinkError>("FDE LSDA " + formatv("{0:x16}", LSDA) +
                                      " does not point at symbol");

    if (LSDASymbol->getAddress() != LSDA)
      return make_error<JITLinkError>(
          "FDE LSDA " + formatv("{0:x16}", LSDA) +
          " does not point to start of symbol at " +
          formatv("{0:x16}", LSDASymbol->getAddress()));

    LLVM_DEBUG(dbgs() << "  FDE LSDA: " << *LSDASymbol << "\n");
  }

  JITTargetAddress RecordAddress = EHFrameAddress + RecordOffset;
  auto FDESymbol = createFDERecord(
      RecordAddress, EHFrameContent.substr(RecordOffset, RecordLength),
      *CIEInfo.CIESymbol, CIEPointerAddress - RecordAddress, *TargetSymbol,
      PCBeginAddress - RecordAddress, LSDASymbol, LSDAAddress - RecordAddress);

  return FDESymbol.takeError();
}

// Determine whether we can register EH tables.
#if (defined(__GNUC__) && !defined(__ARM_EABI__) && !defined(__ia64__) &&      \
     !(defined(_AIX) && defined(__ibmxl__)) && !defined(__SEH__) &&            \
     !defined(__USING_SJLJ_EXCEPTIONS__))
#define HAVE_EHTABLE_SUPPORT 1
#else
#define HAVE_EHTABLE_SUPPORT 0
#endif

#if HAVE_EHTABLE_SUPPORT
extern "C" void __register_frame(const void *);
extern "C" void __deregister_frame(const void *);

Error registerFrameWrapper(const void *P) {
  __register_frame(P);
  return Error::success();
}

Error deregisterFrameWrapper(const void *P) {
  __deregister_frame(P);
  return Error::success();
}

#else

// The building compiler does not have __(de)register_frame but
// it may be found at runtime in a dynamically-loaded library.
// For example, this happens when building LLVM with Visual C++
// but using the MingW runtime.
static Error registerFrameWrapper(const void *P) {
  static void((*RegisterFrame)(const void *)) = 0;

  if (!RegisterFrame)
    *(void **)&RegisterFrame =
        llvm::sys::DynamicLibrary::SearchForAddressOfSymbol("__register_frame");

  if (RegisterFrame) {
    RegisterFrame(P);
    return Error::success();
  }

  return make_error<JITLinkError>("could not register eh-frame: "
                                  "__register_frame function not found");
}

static Error deregisterFrameWrapper(const void *P) {
  static void((*DeregisterFrame)(const void *)) = 0;

  if (!DeregisterFrame)
    *(void **)&DeregisterFrame =
        llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(
            "__deregister_frame");

  if (DeregisterFrame) {
    DeregisterFrame(P);
    return Error::success();
  }

  return make_error<JITLinkError>("could not deregister eh-frame: "
                                  "__deregister_frame function not found");
}
#endif

#ifdef __APPLE__

template <typename HandleFDEFn>
Error walkAppleEHFrameSection(const char *const SectionStart,
                              size_t SectionSize,
                              HandleFDEFn HandleFDE) {
  const char *CurCFIRecord = SectionStart;
  const char *End = SectionStart + SectionSize;
  uint64_t Size = *reinterpret_cast<const uint32_t *>(CurCFIRecord);

  while (CurCFIRecord != End && Size != 0) {
    const char *OffsetField = CurCFIRecord + (Size == 0xffffffff ? 12 : 4);
    if (Size == 0xffffffff)
      Size = *reinterpret_cast<const uint64_t *>(CurCFIRecord + 4) + 12;
    else
      Size += 4;
    uint32_t Offset = *reinterpret_cast<const uint32_t *>(OffsetField);
    if (Offset != 0)
      if (auto Err = HandleFDE(CurCFIRecord))
        return Err;

    LLVM_DEBUG({
      dbgs() << "Registering eh-frame section:\n";
      dbgs() << "Processing " << (Offset ? "FDE" : "CIE") << " @"
             << (void *)CurCFIRecord << ": [";
      for (unsigned I = 0; I < Size; ++I)
        dbgs() << format(" 0x%02" PRIx8, *(CurCFIRecord + I));
      dbgs() << " ]\n";
    });
    CurCFIRecord += Size;

    Size = *reinterpret_cast<const uint32_t *>(CurCFIRecord);
  }

  return Error::success();
}

#endif // __APPLE__

Error registerEHFrameSection(const void *EHFrameSectionAddr,
                             size_t EHFrameSectionSize) {
#ifdef __APPLE__
  // On Darwin __register_frame has to be called for each FDE entry.
  return walkAppleEHFrameSection(static_cast<const char *>(EHFrameSectionAddr),
                                 EHFrameSectionSize,
                                 registerFrameWrapper);
#else
  // On Linux __register_frame takes a single argument:
  // a pointer to the start of the .eh_frame section.

  // How can it find the end? Because crtendS.o is linked
  // in and it has an .eh_frame section with four zero chars.
  return registerFrameWrapper(EHFrameSectionAddr);
#endif
}

Error deregisterEHFrameSection(const void *EHFrameSectionAddr,
                               size_t EHFrameSectionSize) {
#ifdef __APPLE__
  return walkAppleEHFrameSection(static_cast<const char *>(EHFrameSectionAddr),
                                 EHFrameSectionSize,
                                 deregisterFrameWrapper);
#else
  return deregisterFrameWrapper(EHFrameSectionAddr);
#endif
}

EHFrameRegistrar::~EHFrameRegistrar() {}

InProcessEHFrameRegistrar &InProcessEHFrameRegistrar::getInstance() {
  static InProcessEHFrameRegistrar Instance;
  return Instance;
}

InProcessEHFrameRegistrar::InProcessEHFrameRegistrar() {}

LinkGraphPassFunction
createEHFrameRecorderPass(const Triple &TT,
                          StoreFrameRangeFunction StoreRangeAddress) {
  const char *EHFrameSectionName = nullptr;
  if (TT.getObjectFormat() == Triple::MachO)
    EHFrameSectionName = "__eh_frame";
  else
    EHFrameSectionName = ".eh_frame";

  auto RecordEHFrame =
      [EHFrameSectionName,
       StoreFrameRange = std::move(StoreRangeAddress)](LinkGraph &G) -> Error {
    // Search for a non-empty eh-frame and record the address of the first
    // symbol in it.
    JITTargetAddress Addr = 0;
    size_t Size = 0;
    if (auto *S = G.findSectionByName(EHFrameSectionName)) {
      auto R = SectionRange(*S);
      Addr = R.getStart();
      Size = R.getSize();
    }
    if (Addr == 0 && Size != 0)
      return make_error<JITLinkError>("__eh_frame section can not have zero "
                                      "address with non-zero size");
    StoreFrameRange(Addr, Size);
    return Error::success();
  };

  return RecordEHFrame;
}

} // end namespace jitlink
} // end namespace llvm
