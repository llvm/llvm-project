//===-- CodeGenData.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for codegen data that has stable summary which
// can be used to optimize the code in the subsequent codegen.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CGData/CodeGenDataReader.h"
#include "llvm/CGData/OutlinedHashTreeRecord.h"
#include "llvm/CGData/StableFunctionMapRecord.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Caching.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/WithColor.h"

#define DEBUG_TYPE "cg-data"

using namespace llvm;
using namespace cgdata;

static cl::opt<bool>
    CodeGenDataGenerate("codegen-data-generate", cl::init(false), cl::Hidden,
                        cl::desc("Emit CodeGen Data into custom sections"));
static cl::opt<std::string>
    CodeGenDataUsePath("codegen-data-use-path", cl::init(""), cl::Hidden,
                       cl::desc("File path to where .cgdata file is read"));
cl::opt<bool> CodeGenDataThinLTOTwoRounds(
    "codegen-data-thinlto-two-rounds", cl::init(false), cl::Hidden,
    cl::desc("Enable two-round ThinLTO code generation. The first round "
             "emits codegen data, while the second round uses the emitted "
             "codegen data for further optimizations."));

static std::string getCGDataErrString(cgdata_error Err,
                                      const std::string &ErrMsg = "") {
  std::string Msg;
  raw_string_ostream OS(Msg);

  switch (Err) {
  case cgdata_error::success:
    OS << "success";
    break;
  case cgdata_error::eof:
    OS << "end of File";
    break;
  case cgdata_error::bad_magic:
    OS << "invalid codegen data (bad magic)";
    break;
  case cgdata_error::bad_header:
    OS << "invalid codegen data (file header is corrupt)";
    break;
  case cgdata_error::empty_cgdata:
    OS << "empty codegen data";
    break;
  case cgdata_error::malformed:
    OS << "malformed codegen data";
    break;
  case cgdata_error::unsupported_version:
    OS << "unsupported codegen data version";
    break;
  }

  // If optional error message is not empty, append it to the message.
  if (!ErrMsg.empty())
    OS << ": " << ErrMsg;

  return OS.str();
}

namespace {

// FIXME: This class is only here to support the transition to llvm::Error. It
// will be removed once this transition is complete. Clients should prefer to
// deal with the Error value directly, rather than converting to error_code.
class CGDataErrorCategoryType : public std::error_category {
  const char *name() const noexcept override { return "llvm.cgdata"; }

  std::string message(int IE) const override {
    return getCGDataErrString(static_cast<cgdata_error>(IE));
  }
};

} // end anonymous namespace

const std::error_category &llvm::cgdata_category() {
  static CGDataErrorCategoryType ErrorCategory;
  return ErrorCategory;
}

std::string CGDataError::message() const {
  return getCGDataErrString(Err, Msg);
}

char CGDataError::ID = 0;

namespace {

const char *CodeGenDataSectNameCommon[] = {
#define CG_DATA_SECT_ENTRY(Kind, SectNameCommon, SectNameCoff, Prefix)         \
  SectNameCommon,
#include "llvm/CGData/CodeGenData.inc"
};

const char *CodeGenDataSectNameCoff[] = {
#define CG_DATA_SECT_ENTRY(Kind, SectNameCommon, SectNameCoff, Prefix)         \
  SectNameCoff,
#include "llvm/CGData/CodeGenData.inc"
};

const char *CodeGenDataSectNamePrefix[] = {
#define CG_DATA_SECT_ENTRY(Kind, SectNameCommon, SectNameCoff, Prefix) Prefix,
#include "llvm/CGData/CodeGenData.inc"
};

} // namespace

namespace llvm {

std::string getCodeGenDataSectionName(CGDataSectKind CGSK,
                                      Triple::ObjectFormatType OF,
                                      bool AddSegmentInfo) {
  std::string SectName;

  if (OF == Triple::MachO && AddSegmentInfo)
    SectName = CodeGenDataSectNamePrefix[CGSK];

  if (OF == Triple::COFF)
    SectName += CodeGenDataSectNameCoff[CGSK];
  else
    SectName += CodeGenDataSectNameCommon[CGSK];

  return SectName;
}

std::unique_ptr<CodeGenData> CodeGenData::Instance = nullptr;
std::once_flag CodeGenData::OnceFlag;

CodeGenData &CodeGenData::getInstance() {
  std::call_once(CodeGenData::OnceFlag, []() {
    Instance = std::unique_ptr<CodeGenData>(new CodeGenData());

    if (CodeGenDataGenerate || CodeGenDataThinLTOTwoRounds)
      Instance->EmitCGData = true;
    else if (!CodeGenDataUsePath.empty()) {
      // Initialize the global CGData if the input file name is given.
      // We do not error-out when failing to parse the input file.
      // Instead, just emit an warning message and fall back as if no CGData
      // were available.
      auto FS = vfs::getRealFileSystem();
      auto ReaderOrErr = CodeGenDataReader::create(CodeGenDataUsePath, *FS);
      if (Error E = ReaderOrErr.takeError()) {
        warn(std::move(E), CodeGenDataUsePath);
        return;
      }
      // Publish each CGData based on the data type in the header.
      auto Reader = ReaderOrErr->get();
      if (Reader->hasOutlinedHashTree())
        Instance->publishOutlinedHashTree(Reader->releaseOutlinedHashTree());
      if (Reader->hasStableFunctionMap())
        Instance->publishStableFunctionMap(Reader->releaseStableFunctionMap());
    }
  });
  return *Instance;
}

namespace IndexedCGData {

Expected<Header> Header::readFromBuffer(const unsigned char *Curr) {
  using namespace support;

  static_assert(std::is_standard_layout_v<llvm::IndexedCGData::Header>,
                "The header should be standard layout type since we use offset "
                "of fields to read.");
  Header H;
  H.Magic = endian::readNext<uint64_t, endianness::little, unaligned>(Curr);
  if (H.Magic != IndexedCGData::Magic)
    return make_error<CGDataError>(cgdata_error::bad_magic);
  H.Version = endian::readNext<uint32_t, endianness::little, unaligned>(Curr);
  if (H.Version > IndexedCGData::CGDataVersion::CurrentVersion)
    return make_error<CGDataError>(cgdata_error::unsupported_version);
  H.DataKind = endian::readNext<uint32_t, endianness::little, unaligned>(Curr);

  static_assert(IndexedCGData::CGDataVersion::CurrentVersion == Version3,
                "Please update the offset computation below if a new field has "
                "been added to the header.");
  H.OutlinedHashTreeOffset =
      endian::readNext<uint64_t, endianness::little, unaligned>(Curr);
  if (H.Version >= 2)
    H.StableFunctionMapOffset =
        endian::readNext<uint64_t, endianness::little, unaligned>(Curr);

  return H;
}

} // end namespace IndexedCGData

namespace cgdata {

void warn(Twine Message, StringRef Whence, StringRef Hint) {
  WithColor::warning();
  if (!Whence.empty())
    errs() << Whence << ": ";
  errs() << Message << "\n";
  if (!Hint.empty())
    WithColor::note() << Hint << "\n";
}

void warn(Error E, StringRef Whence) {
  if (E.isA<CGDataError>()) {
    handleAllErrors(std::move(E), [&](const CGDataError &IPE) {
      warn(IPE.message(), Whence, "");
    });
  }
}

void saveModuleForTwoRounds(const Module &TheModule, unsigned Task,
                            AddStreamFn AddStream) {
  LLVM_DEBUG(dbgs() << "Saving module: " << TheModule.getModuleIdentifier()
                    << " in Task " << Task << "\n");
  Expected<std::unique_ptr<CachedFileStream>> StreamOrErr =
      AddStream(Task, TheModule.getModuleIdentifier());
  if (Error Err = StreamOrErr.takeError())
    report_fatal_error(std::move(Err));
  std::unique_ptr<CachedFileStream> &Stream = *StreamOrErr;

  WriteBitcodeToFile(TheModule, *Stream->OS,
                     /*ShouldPreserveUseListOrder=*/true);

  if (Error Err = Stream->commit())
    report_fatal_error(std::move(Err));
}

std::unique_ptr<Module> loadModuleForTwoRounds(BitcodeModule &OrigModule,
                                               unsigned Task,
                                               LLVMContext &Context,
                                               ArrayRef<StringRef> IRFiles) {
  LLVM_DEBUG(dbgs() << "Loading module: " << OrigModule.getModuleIdentifier()
                    << " in Task " << Task << "\n");
  auto FileBuffer = MemoryBuffer::getMemBuffer(
      IRFiles[Task], "in-memory IR file", /*RequiresNullTerminator=*/false);
  auto RestoredModule = parseBitcodeFile(*FileBuffer, Context);
  if (!RestoredModule)
    report_fatal_error(
        Twine("Failed to parse optimized bitcode loaded for Task: ") +
        Twine(Task) + "\n");

  // Restore the original module identifier.
  (*RestoredModule)->setModuleIdentifier(OrigModule.getModuleIdentifier());
  return std::move(*RestoredModule);
}

Expected<stable_hash> mergeCodeGenData(ArrayRef<StringRef> ObjFiles) {
  OutlinedHashTreeRecord GlobalOutlineRecord;
  StableFunctionMapRecord GlobalStableFunctionMapRecord;
  stable_hash CombinedHash = 0;
  for (auto File : ObjFiles) {
    if (File.empty())
      continue;
    std::unique_ptr<MemoryBuffer> Buffer = MemoryBuffer::getMemBuffer(
        File, "in-memory object file", /*RequiresNullTerminator=*/false);
    Expected<std::unique_ptr<object::ObjectFile>> BinOrErr =
        object::ObjectFile::createObjectFile(Buffer->getMemBufferRef());
    if (!BinOrErr)
      return BinOrErr.takeError();

    std::unique_ptr<object::ObjectFile> &Obj = BinOrErr.get();
    if (auto E = CodeGenDataReader::mergeFromObjectFile(
            Obj.get(), GlobalOutlineRecord, GlobalStableFunctionMapRecord,
            &CombinedHash))
      return E;
  }

  GlobalStableFunctionMapRecord.finalize();

  if (!GlobalOutlineRecord.empty())
    cgdata::publishOutlinedHashTree(std::move(GlobalOutlineRecord.HashTree));
  if (!GlobalStableFunctionMapRecord.empty())
    cgdata::publishStableFunctionMap(
        std::move(GlobalStableFunctionMapRecord.FunctionMap));

  return CombinedHash;
}

} // end namespace cgdata

} // end namespace llvm
