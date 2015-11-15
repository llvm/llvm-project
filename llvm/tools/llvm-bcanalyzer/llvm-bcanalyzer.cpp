//===-- llvm-bcanalyzer.cpp - Bitcode Analyzer --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tool may be invoked in the following manner:
//  llvm-bcanalyzer [options]      - Read LLVM bitcode from stdin
//  llvm-bcanalyzer [options] x.bc - Read LLVM bitcode from the x.bc file
//
//  Options:
//      --help      - Output information about command line switches
//      --dump      - Dump low-level bitcode structure in readable format
//
// This tool provides analytical information about a bitcode file. It is
// intended as an aid to developers of bitcode reading and writing software. It
// produces on std::out a summary of the bitcode file that shows various
// statistics about the contents of the file. By default this information is
// detailed and contains information about individual bitcode blocks and the
// functions in the module.
// The tool is also able to print a bitcode file in a straight forward text
// format that shows the containment and relationships of the information in
// the bitcode file (-dump option).
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cctype>
#include <map>
#include <system_error>
using namespace llvm;

static cl::opt<std::string>
  InputFilename(cl::Positional, cl::desc("<input bitcode>"), cl::init("-"));

static cl::opt<bool> Dump("dump", cl::desc("Dump low level bitcode trace"));

//===----------------------------------------------------------------------===//
// Bitcode specific analysis.
//===----------------------------------------------------------------------===//

static cl::opt<bool> NoHistogram("disable-histogram",
                                 cl::desc("Do not print per-code histogram"));

static cl::opt<bool>
NonSymbolic("non-symbolic",
            cl::desc("Emit numeric info in dump even if"
                     " symbolic info is available"));

static cl::opt<std::string>
  BlockInfoFilename("block-info",
                    cl::desc("Use the BLOCK_INFO from the given file"));

static cl::opt<bool>
  ShowBinaryBlobs("show-binary-blobs",
                  cl::desc("Print binary blobs using hex escapes"));

namespace {

/// CurStreamTypeType - A type for CurStreamType
enum CurStreamTypeType {
  UnknownBitstream,
  LLVMIRBitstream
};

}

/// GetBlockName - Return a symbolic block name if known, otherwise return
/// null.
static const char *GetBlockName(unsigned BlockID,
                                const BitstreamReader &StreamFile,
                                CurStreamTypeType CurStreamType) {
  // Standard blocks for all bitcode files.
  if (BlockID < bitc::FIRST_APPLICATION_BLOCKID) {
    if (BlockID == bitc::BLOCKINFO_BLOCK_ID)
      return "BLOCKINFO_BLOCK";
    return nullptr;
  }

  // Check to see if we have a blockinfo record for this block, with a name.
  if (const BitstreamReader::BlockInfo *Info =
        StreamFile.getBlockInfo(BlockID)) {
    if (!Info->Name.empty())
      return Info->Name.c_str();
  }


  if (CurStreamType != LLVMIRBitstream) return nullptr;

  switch (BlockID) {
  default:                             return nullptr;
  case bitc::MODULE_BLOCK_ID:          return "MODULE_BLOCK";
  case bitc::PARAMATTR_BLOCK_ID:       return "PARAMATTR_BLOCK";
  case bitc::PARAMATTR_GROUP_BLOCK_ID: return "PARAMATTR_GROUP_BLOCK_ID";
  case bitc::TYPE_BLOCK_ID_NEW:        return "TYPE_BLOCK_ID";
  case bitc::CONSTANTS_BLOCK_ID:       return "CONSTANTS_BLOCK";
  case bitc::FUNCTION_BLOCK_ID:        return "FUNCTION_BLOCK";
  case bitc::IDENTIFICATION_BLOCK_ID:
    return "IDENTIFICATION_BLOCK_ID";
  case bitc::VALUE_SYMTAB_BLOCK_ID:    return "VALUE_SYMTAB";
  case bitc::METADATA_BLOCK_ID:        return "METADATA_BLOCK";
  case bitc::METADATA_KIND_BLOCK_ID:   return "METADATA_KIND_BLOCK";
  case bitc::METADATA_ATTACHMENT_ID:   return "METADATA_ATTACHMENT_BLOCK";
  case bitc::USELIST_BLOCK_ID:         return "USELIST_BLOCK_ID";
  case bitc::FUNCTION_SUMMARY_BLOCK_ID:
                                       return "FUNCTION_SUMMARY_BLOCK";
  case bitc::MODULE_STRTAB_BLOCK_ID:   return "MODULE_STRTAB_BLOCK";
  }
}

/// GetCodeName - Return a symbolic code name if known, otherwise return
/// null.
static const char *GetCodeName(unsigned CodeID, unsigned BlockID,
                               const BitstreamReader &StreamFile,
                               CurStreamTypeType CurStreamType) {
  // Standard blocks for all bitcode files.
  if (BlockID < bitc::FIRST_APPLICATION_BLOCKID) {
    if (BlockID == bitc::BLOCKINFO_BLOCK_ID) {
      switch (CodeID) {
      default: return nullptr;
      case bitc::BLOCKINFO_CODE_SETBID:        return "SETBID";
      case bitc::BLOCKINFO_CODE_BLOCKNAME:     return "BLOCKNAME";
      case bitc::BLOCKINFO_CODE_SETRECORDNAME: return "SETRECORDNAME";
      }
    }
    return nullptr;
  }

  // Check to see if we have a blockinfo record for this record, with a name.
  if (const BitstreamReader::BlockInfo *Info =
        StreamFile.getBlockInfo(BlockID)) {
    for (unsigned i = 0, e = Info->RecordNames.size(); i != e; ++i)
      if (Info->RecordNames[i].first == CodeID)
        return Info->RecordNames[i].second.c_str();
  }


  if (CurStreamType != LLVMIRBitstream) return nullptr;

#define STRINGIFY_CODE(PREFIX, CODE)                                           \
  case bitc::PREFIX##_##CODE:                                                  \
    return #CODE;
  switch (BlockID) {
  default: return nullptr;
  case bitc::MODULE_BLOCK_ID:
    switch (CodeID) {
    default: return nullptr;
      STRINGIFY_CODE(MODULE_CODE, VERSION)
      STRINGIFY_CODE(MODULE_CODE, TRIPLE)
      STRINGIFY_CODE(MODULE_CODE, DATALAYOUT)
      STRINGIFY_CODE(MODULE_CODE, ASM)
      STRINGIFY_CODE(MODULE_CODE, SECTIONNAME)
      STRINGIFY_CODE(MODULE_CODE, DEPLIB) // FIXME: Remove in 4.0
      STRINGIFY_CODE(MODULE_CODE, GLOBALVAR)
      STRINGIFY_CODE(MODULE_CODE, FUNCTION)
      STRINGIFY_CODE(MODULE_CODE, ALIAS)
      STRINGIFY_CODE(MODULE_CODE, PURGEVALS)
      STRINGIFY_CODE(MODULE_CODE, GCNAME)
      STRINGIFY_CODE(MODULE_CODE, VSTOFFSET)
    }
  case bitc::IDENTIFICATION_BLOCK_ID:
    switch (CodeID) {
    default:
      return nullptr;
      STRINGIFY_CODE(IDENTIFICATION_CODE, STRING)
      STRINGIFY_CODE(IDENTIFICATION_CODE, EPOCH)
    }
  case bitc::PARAMATTR_BLOCK_ID:
    switch (CodeID) {
    default: return nullptr;
    // FIXME: Should these be different?
    case bitc::PARAMATTR_CODE_ENTRY_OLD: return "ENTRY";
    case bitc::PARAMATTR_CODE_ENTRY:     return "ENTRY";
    case bitc::PARAMATTR_GRP_CODE_ENTRY: return "ENTRY";
    }
  case bitc::TYPE_BLOCK_ID_NEW:
    switch (CodeID) {
    default: return nullptr;
      STRINGIFY_CODE(TYPE_CODE, NUMENTRY)
      STRINGIFY_CODE(TYPE_CODE, VOID)
      STRINGIFY_CODE(TYPE_CODE, FLOAT)
      STRINGIFY_CODE(TYPE_CODE, DOUBLE)
      STRINGIFY_CODE(TYPE_CODE, LABEL)
      STRINGIFY_CODE(TYPE_CODE, OPAQUE)
      STRINGIFY_CODE(TYPE_CODE, INTEGER)
      STRINGIFY_CODE(TYPE_CODE, POINTER)
      STRINGIFY_CODE(TYPE_CODE, ARRAY)
      STRINGIFY_CODE(TYPE_CODE, VECTOR)
      STRINGIFY_CODE(TYPE_CODE, X86_FP80)
      STRINGIFY_CODE(TYPE_CODE, FP128)
      STRINGIFY_CODE(TYPE_CODE, PPC_FP128)
      STRINGIFY_CODE(TYPE_CODE, METADATA)
      STRINGIFY_CODE(TYPE_CODE, STRUCT_ANON)
      STRINGIFY_CODE(TYPE_CODE, STRUCT_NAME)
      STRINGIFY_CODE(TYPE_CODE, STRUCT_NAMED)
      STRINGIFY_CODE(TYPE_CODE, FUNCTION)
    }

  case bitc::CONSTANTS_BLOCK_ID:
    switch (CodeID) {
    default: return nullptr;
      STRINGIFY_CODE(CST_CODE, SETTYPE)
      STRINGIFY_CODE(CST_CODE, NULL)
      STRINGIFY_CODE(CST_CODE, UNDEF)
      STRINGIFY_CODE(CST_CODE, INTEGER)
      STRINGIFY_CODE(CST_CODE, WIDE_INTEGER)
      STRINGIFY_CODE(CST_CODE, FLOAT)
      STRINGIFY_CODE(CST_CODE, AGGREGATE)
      STRINGIFY_CODE(CST_CODE, STRING)
      STRINGIFY_CODE(CST_CODE, CSTRING)
      STRINGIFY_CODE(CST_CODE, CE_BINOP)
      STRINGIFY_CODE(CST_CODE, CE_CAST)
      STRINGIFY_CODE(CST_CODE, CE_GEP)
      STRINGIFY_CODE(CST_CODE, CE_INBOUNDS_GEP)
      STRINGIFY_CODE(CST_CODE, CE_SELECT)
      STRINGIFY_CODE(CST_CODE, CE_EXTRACTELT)
      STRINGIFY_CODE(CST_CODE, CE_INSERTELT)
      STRINGIFY_CODE(CST_CODE, CE_SHUFFLEVEC)
      STRINGIFY_CODE(CST_CODE, CE_CMP)
      STRINGIFY_CODE(CST_CODE, INLINEASM)
      STRINGIFY_CODE(CST_CODE, CE_SHUFVEC_EX)
    case bitc::CST_CODE_BLOCKADDRESS:    return "CST_CODE_BLOCKADDRESS";
      STRINGIFY_CODE(CST_CODE, DATA)
    }
  case bitc::FUNCTION_BLOCK_ID:
    switch (CodeID) {
    default: return nullptr;
      STRINGIFY_CODE(FUNC_CODE, DECLAREBLOCKS)
      STRINGIFY_CODE(FUNC_CODE, INST_BINOP)
      STRINGIFY_CODE(FUNC_CODE, INST_CAST)
      STRINGIFY_CODE(FUNC_CODE, INST_GEP_OLD)
      STRINGIFY_CODE(FUNC_CODE, INST_INBOUNDS_GEP_OLD)
      STRINGIFY_CODE(FUNC_CODE, INST_SELECT)
      STRINGIFY_CODE(FUNC_CODE, INST_EXTRACTELT)
      STRINGIFY_CODE(FUNC_CODE, INST_INSERTELT)
      STRINGIFY_CODE(FUNC_CODE, INST_SHUFFLEVEC)
      STRINGIFY_CODE(FUNC_CODE, INST_CMP)
      STRINGIFY_CODE(FUNC_CODE, INST_RET)
      STRINGIFY_CODE(FUNC_CODE, INST_BR)
      STRINGIFY_CODE(FUNC_CODE, INST_SWITCH)
      STRINGIFY_CODE(FUNC_CODE, INST_INVOKE)
      STRINGIFY_CODE(FUNC_CODE, INST_UNREACHABLE)
      STRINGIFY_CODE(FUNC_CODE, INST_CLEANUPRET)
      STRINGIFY_CODE(FUNC_CODE, INST_CATCHRET)
      STRINGIFY_CODE(FUNC_CODE, INST_CATCHPAD)
      STRINGIFY_CODE(FUNC_CODE, INST_CLEANUPENDPAD)
      STRINGIFY_CODE(FUNC_CODE, INST_CATCHENDPAD)
      STRINGIFY_CODE(FUNC_CODE, INST_TERMINATEPAD)
      STRINGIFY_CODE(FUNC_CODE, INST_PHI)
      STRINGIFY_CODE(FUNC_CODE, INST_ALLOCA)
      STRINGIFY_CODE(FUNC_CODE, INST_LOAD)
      STRINGIFY_CODE(FUNC_CODE, INST_VAARG)
      STRINGIFY_CODE(FUNC_CODE, INST_STORE)
      STRINGIFY_CODE(FUNC_CODE, INST_EXTRACTVAL)
      STRINGIFY_CODE(FUNC_CODE, INST_INSERTVAL)
      STRINGIFY_CODE(FUNC_CODE, INST_CMP2)
      STRINGIFY_CODE(FUNC_CODE, INST_VSELECT)
      STRINGIFY_CODE(FUNC_CODE, DEBUG_LOC_AGAIN)
      STRINGIFY_CODE(FUNC_CODE, INST_CALL)
      STRINGIFY_CODE(FUNC_CODE, DEBUG_LOC)
      STRINGIFY_CODE(FUNC_CODE, INST_GEP)
    }
  case bitc::VALUE_SYMTAB_BLOCK_ID:
    switch (CodeID) {
    default: return nullptr;
    STRINGIFY_CODE(VST_CODE, ENTRY)
    STRINGIFY_CODE(VST_CODE, BBENTRY)
    STRINGIFY_CODE(VST_CODE, FNENTRY)
    STRINGIFY_CODE(VST_CODE, COMBINED_FNENTRY)
    }
  case bitc::MODULE_STRTAB_BLOCK_ID:
    switch (CodeID) {
    default:
      return nullptr;
      STRINGIFY_CODE(MST_CODE, ENTRY)
    }
  case bitc::FUNCTION_SUMMARY_BLOCK_ID:
    switch (CodeID) {
    default:
      return nullptr;
      STRINGIFY_CODE(FS_CODE, PERMODULE_ENTRY)
      STRINGIFY_CODE(FS_CODE, COMBINED_ENTRY)
    }
  case bitc::METADATA_ATTACHMENT_ID:
    switch(CodeID) {
    default:return nullptr;
      STRINGIFY_CODE(METADATA, ATTACHMENT)
    }
  case bitc::METADATA_BLOCK_ID:
    switch(CodeID) {
    default:return nullptr;
      STRINGIFY_CODE(METADATA, STRING)
      STRINGIFY_CODE(METADATA, NAME)
      STRINGIFY_CODE(METADATA, KIND) // Older bitcode has it in a MODULE_BLOCK
      STRINGIFY_CODE(METADATA, NODE)
      STRINGIFY_CODE(METADATA, VALUE)
      STRINGIFY_CODE(METADATA, OLD_NODE)
      STRINGIFY_CODE(METADATA, OLD_FN_NODE)
      STRINGIFY_CODE(METADATA, NAMED_NODE)
      STRINGIFY_CODE(METADATA, DISTINCT_NODE)
      STRINGIFY_CODE(METADATA, LOCATION)
      STRINGIFY_CODE(METADATA, GENERIC_DEBUG)
      STRINGIFY_CODE(METADATA, SUBRANGE)
      STRINGIFY_CODE(METADATA, ENUMERATOR)
      STRINGIFY_CODE(METADATA, BASIC_TYPE)
      STRINGIFY_CODE(METADATA, FILE)
      STRINGIFY_CODE(METADATA, DERIVED_TYPE)
      STRINGIFY_CODE(METADATA, COMPOSITE_TYPE)
      STRINGIFY_CODE(METADATA, SUBROUTINE_TYPE)
      STRINGIFY_CODE(METADATA, COMPILE_UNIT)
      STRINGIFY_CODE(METADATA, SUBPROGRAM)
      STRINGIFY_CODE(METADATA, LEXICAL_BLOCK)
      STRINGIFY_CODE(METADATA, LEXICAL_BLOCK_FILE)
      STRINGIFY_CODE(METADATA, NAMESPACE)
      STRINGIFY_CODE(METADATA, TEMPLATE_TYPE)
      STRINGIFY_CODE(METADATA, TEMPLATE_VALUE)
      STRINGIFY_CODE(METADATA, GLOBAL_VAR)
      STRINGIFY_CODE(METADATA, LOCAL_VAR)
      STRINGIFY_CODE(METADATA, EXPRESSION)
      STRINGIFY_CODE(METADATA, OBJC_PROPERTY)
      STRINGIFY_CODE(METADATA, IMPORTED_ENTITY)
      STRINGIFY_CODE(METADATA, MODULE)
    }
  case bitc::METADATA_KIND_BLOCK_ID:
    switch (CodeID) {
    default:
      return nullptr;
      STRINGIFY_CODE(METADATA, KIND)
    }
  case bitc::USELIST_BLOCK_ID:
    switch(CodeID) {
    default:return nullptr;
    case bitc::USELIST_CODE_DEFAULT: return "USELIST_CODE_DEFAULT";
    case bitc::USELIST_CODE_BB:      return "USELIST_CODE_BB";
    }
  }
#undef STRINGIFY_CODE
}

struct PerRecordStats {
  unsigned NumInstances;
  unsigned NumAbbrev;
  uint64_t TotalBits;

  PerRecordStats() : NumInstances(0), NumAbbrev(0), TotalBits(0) {}
};

struct PerBlockIDStats {
  /// NumInstances - This the number of times this block ID has been seen.
  unsigned NumInstances;

  /// NumBits - The total size in bits of all of these blocks.
  uint64_t NumBits;

  /// NumSubBlocks - The total number of blocks these blocks contain.
  unsigned NumSubBlocks;

  /// NumAbbrevs - The total number of abbreviations.
  unsigned NumAbbrevs;

  /// NumRecords - The total number of records these blocks contain, and the
  /// number that are abbreviated.
  unsigned NumRecords, NumAbbreviatedRecords;

  /// CodeFreq - Keep track of the number of times we see each code.
  std::vector<PerRecordStats> CodeFreq;

  PerBlockIDStats()
    : NumInstances(0), NumBits(0),
      NumSubBlocks(0), NumAbbrevs(0), NumRecords(0), NumAbbreviatedRecords(0) {}
};

static std::map<unsigned, PerBlockIDStats> BlockIDStats;



/// Error - All bitcode analysis errors go through this function, making this a
/// good place to breakpoint if debugging.
static bool Error(const Twine &Err) {
  errs() << Err << "\n";
  return true;
}

/// ParseBlock - Read a block, updating statistics, etc.
static bool ParseBlock(BitstreamCursor &Stream, unsigned BlockID,
                       unsigned IndentLevel, CurStreamTypeType CurStreamType) {
  std::string Indent(IndentLevel*2, ' ');
  uint64_t BlockBitStart = Stream.GetCurrentBitNo();

  // Get the statistics for this BlockID.
  PerBlockIDStats &BlockStats = BlockIDStats[BlockID];

  BlockStats.NumInstances++;

  // BLOCKINFO is a special part of the stream.
  if (BlockID == bitc::BLOCKINFO_BLOCK_ID) {
    if (Dump) outs() << Indent << "<BLOCKINFO_BLOCK/>\n";
    if (Stream.ReadBlockInfoBlock())
      return Error("Malformed BlockInfoBlock");
    uint64_t BlockBitEnd = Stream.GetCurrentBitNo();
    BlockStats.NumBits += BlockBitEnd-BlockBitStart;
    return false;
  }

  unsigned NumWords = 0;
  if (Stream.EnterSubBlock(BlockID, &NumWords))
    return Error("Malformed block record");

  const char *BlockName = nullptr;
  if (Dump) {
    outs() << Indent << "<";
    if ((BlockName = GetBlockName(BlockID, *Stream.getBitStreamReader(),
                                  CurStreamType)))
      outs() << BlockName;
    else
      outs() << "UnknownBlock" << BlockID;

    if (NonSymbolic && BlockName)
      outs() << " BlockID=" << BlockID;

    outs() << " NumWords=" << NumWords
           << " BlockCodeSize=" << Stream.getAbbrevIDWidth() << ">\n";
  }

  SmallVector<uint64_t, 64> Record;

  // Read all the records for this block.
  while (1) {
    if (Stream.AtEndOfStream())
      return Error("Premature end of bitstream");

    uint64_t RecordStartBit = Stream.GetCurrentBitNo();

    BitstreamEntry Entry =
      Stream.advance(BitstreamCursor::AF_DontAutoprocessAbbrevs);
    
    switch (Entry.Kind) {
    case BitstreamEntry::Error:
      return Error("malformed bitcode file");
    case BitstreamEntry::EndBlock: {
      uint64_t BlockBitEnd = Stream.GetCurrentBitNo();
      BlockStats.NumBits += BlockBitEnd-BlockBitStart;
      if (Dump) {
        outs() << Indent << "</";
        if (BlockName)
          outs() << BlockName << ">\n";
        else
          outs() << "UnknownBlock" << BlockID << ">\n";
      }
      return false;
    }
        
    case BitstreamEntry::SubBlock: {
      uint64_t SubBlockBitStart = Stream.GetCurrentBitNo();
      if (ParseBlock(Stream, Entry.ID, IndentLevel+1, CurStreamType))
        return true;
      ++BlockStats.NumSubBlocks;
      uint64_t SubBlockBitEnd = Stream.GetCurrentBitNo();
      
      // Don't include subblock sizes in the size of this block.
      BlockBitStart += SubBlockBitEnd-SubBlockBitStart;
      continue;
    }
    case BitstreamEntry::Record:
      // The interesting case.
      break;
    }

    if (Entry.ID == bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      ++BlockStats.NumAbbrevs;
      continue;
    }
    
    Record.clear();

    ++BlockStats.NumRecords;

    StringRef Blob;
    unsigned Code = Stream.readRecord(Entry.ID, Record, &Blob);

    // Increment the # occurrences of this code.
    if (BlockStats.CodeFreq.size() <= Code)
      BlockStats.CodeFreq.resize(Code+1);
    BlockStats.CodeFreq[Code].NumInstances++;
    BlockStats.CodeFreq[Code].TotalBits +=
      Stream.GetCurrentBitNo()-RecordStartBit;
    if (Entry.ID != bitc::UNABBREV_RECORD) {
      BlockStats.CodeFreq[Code].NumAbbrev++;
      ++BlockStats.NumAbbreviatedRecords;
    }

    if (Dump) {
      outs() << Indent << "  <";
      if (const char *CodeName =
            GetCodeName(Code, BlockID, *Stream.getBitStreamReader(),
                        CurStreamType))
        outs() << CodeName;
      else
        outs() << "UnknownCode" << Code;
      if (NonSymbolic &&
          GetCodeName(Code, BlockID, *Stream.getBitStreamReader(),
                      CurStreamType))
        outs() << " codeid=" << Code;
      const BitCodeAbbrev *Abbv = nullptr;
      if (Entry.ID != bitc::UNABBREV_RECORD) {
        Abbv = Stream.getAbbrev(Entry.ID);
        outs() << " abbrevid=" << Entry.ID;
      }

      for (unsigned i = 0, e = Record.size(); i != e; ++i)
        outs() << " op" << i << "=" << (int64_t)Record[i];

      outs() << "/>";

      if (Abbv) {
        for (unsigned i = 1, e = Abbv->getNumOperandInfos(); i != e; ++i) {
          const BitCodeAbbrevOp &Op = Abbv->getOperandInfo(i);
          if (!Op.isEncoding() || Op.getEncoding() != BitCodeAbbrevOp::Array)
            continue;
          assert(i + 2 == e && "Array op not second to last");
          std::string Str;
          bool ArrayIsPrintable = true;
          for (unsigned j = i - 1, je = Record.size(); j != je; ++j) {
            if (!isprint(static_cast<unsigned char>(Record[j]))) {
              ArrayIsPrintable = false;
              break;
            }
            Str += (char)Record[j];
          }
          if (ArrayIsPrintable)
            outs() << " record string = '" << Str << "'";
          break;
        }
      }

      if (Blob.data()) {
        outs() << " blob data = ";
        if (ShowBinaryBlobs) {
          outs() << "'";
          outs().write_escaped(Blob, /*hex=*/true) << "'";
        } else {
          bool BlobIsPrintable = true;
          for (unsigned i = 0, e = Blob.size(); i != e; ++i)
            if (!isprint(static_cast<unsigned char>(Blob[i]))) {
              BlobIsPrintable = false;
              break;
            }

          if (BlobIsPrintable)
            outs() << "'" << Blob << "'";
          else
            outs() << "unprintable, " << Blob.size() << " bytes.";          
        }
      }

      outs() << "\n";
    }
  }
}

static void PrintSize(double Bits) {
  outs() << format("%.2f/%.2fB/%luW", Bits, Bits/8,(unsigned long)(Bits/32));
}
static void PrintSize(uint64_t Bits) {
  outs() << format("%lub/%.2fB/%luW", (unsigned long)Bits,
                   (double)Bits/8, (unsigned long)(Bits/32));
}

static bool openBitcodeFile(StringRef Path,
                            std::unique_ptr<MemoryBuffer> &MemBuf,
                            BitstreamReader &StreamFile,
                            BitstreamCursor &Stream,
                            CurStreamTypeType &CurStreamType) {
  // Read the input file.
  ErrorOr<std::unique_ptr<MemoryBuffer>> MemBufOrErr =
      MemoryBuffer::getFileOrSTDIN(Path);
  if (std::error_code EC = MemBufOrErr.getError())
    return Error(Twine("Error reading '") + Path + "': " + EC.message());
  MemBuf = std::move(MemBufOrErr.get());

  if (MemBuf->getBufferSize() & 3)
    return Error("Bitcode stream should be a multiple of 4 bytes in length");

  const unsigned char *BufPtr = (const unsigned char *)MemBuf->getBufferStart();
  const unsigned char *EndBufPtr = BufPtr + MemBuf->getBufferSize();

  // If we have a wrapper header, parse it and ignore the non-bc file contents.
  // The magic number is 0x0B17C0DE stored in little endian.
  if (isBitcodeWrapper(BufPtr, EndBufPtr))
    if (SkipBitcodeWrapperHeader(BufPtr, EndBufPtr, true))
      return Error("Invalid bitcode wrapper header");

  StreamFile = BitstreamReader(BufPtr, EndBufPtr);
  Stream = BitstreamCursor(StreamFile);
  StreamFile.CollectBlockInfoNames();

  // Read the stream signature.
  char Signature[6];
  Signature[0] = Stream.Read(8);
  Signature[1] = Stream.Read(8);
  Signature[2] = Stream.Read(4);
  Signature[3] = Stream.Read(4);
  Signature[4] = Stream.Read(4);
  Signature[5] = Stream.Read(4);

  // Autodetect the file contents, if it is one we know.
  CurStreamType = UnknownBitstream;
  if (Signature[0] == 'B' && Signature[1] == 'C' &&
      Signature[2] == 0x0 && Signature[3] == 0xC &&
      Signature[4] == 0xE && Signature[5] == 0xD)
    CurStreamType = LLVMIRBitstream;

  return false;
}

/// AnalyzeBitcode - Analyze the bitcode file specified by InputFilename.
static int AnalyzeBitcode() {
  std::unique_ptr<MemoryBuffer> StreamBuffer;
  BitstreamReader StreamFile;
  BitstreamCursor Stream;
  CurStreamTypeType CurStreamType;
  if (openBitcodeFile(InputFilename, StreamBuffer, StreamFile, Stream,
                      CurStreamType))
    return true;

  // Read block info from BlockInfoFilename, if specified.
  // The block info must be a top-level block.
  if (!BlockInfoFilename.empty()) {
    std::unique_ptr<MemoryBuffer> BlockInfoBuffer;
    BitstreamReader BlockInfoFile;
    BitstreamCursor BlockInfoCursor;
    CurStreamTypeType BlockInfoStreamType;
    if (openBitcodeFile(BlockInfoFilename, BlockInfoBuffer, BlockInfoFile,
                        BlockInfoCursor, BlockInfoStreamType))
      return true;

    while (!BlockInfoCursor.AtEndOfStream()) {
      unsigned Code = BlockInfoCursor.ReadCode();
      if (Code != bitc::ENTER_SUBBLOCK)
        return Error("Invalid record at top-level in block info file");

      unsigned BlockID = BlockInfoCursor.ReadSubBlockID();
      if (BlockID == bitc::BLOCKINFO_BLOCK_ID) {
        if (BlockInfoCursor.ReadBlockInfoBlock())
          return Error("Malformed BlockInfoBlock in block info file");
        break;
      }

      BlockInfoCursor.SkipBlock();
    }

    StreamFile.takeBlockInfo(std::move(BlockInfoFile));
  }

  unsigned NumTopBlocks = 0;

  // Parse the top-level structure.  We only allow blocks at the top-level.
  while (!Stream.AtEndOfStream()) {
    unsigned Code = Stream.ReadCode();
    if (Code != bitc::ENTER_SUBBLOCK)
      return Error("Invalid record at top-level");

    unsigned BlockID = Stream.ReadSubBlockID();

    if (ParseBlock(Stream, BlockID, 0, CurStreamType))
      return true;
    ++NumTopBlocks;
  }

  if (Dump) outs() << "\n\n";

  uint64_t BufferSizeBits = StreamFile.getBitcodeBytes().getExtent() * CHAR_BIT;
  // Print a summary of the read file.
  outs() << "Summary of " << InputFilename << ":\n";
  outs() << "         Total size: ";
  PrintSize(BufferSizeBits);
  outs() << "\n";
  outs() << "        Stream type: ";
  switch (CurStreamType) {
  case UnknownBitstream: outs() << "unknown\n"; break;
  case LLVMIRBitstream:  outs() << "LLVM IR\n"; break;
  }
  outs() << "  # Toplevel Blocks: " << NumTopBlocks << "\n";
  outs() << "\n";

  // Emit per-block stats.
  outs() << "Per-block Summary:\n";
  for (std::map<unsigned, PerBlockIDStats>::iterator I = BlockIDStats.begin(),
       E = BlockIDStats.end(); I != E; ++I) {
    outs() << "  Block ID #" << I->first;
    if (const char *BlockName = GetBlockName(I->first, StreamFile,
                                             CurStreamType))
      outs() << " (" << BlockName << ")";
    outs() << ":\n";

    const PerBlockIDStats &Stats = I->second;
    outs() << "      Num Instances: " << Stats.NumInstances << "\n";
    outs() << "         Total Size: ";
    PrintSize(Stats.NumBits);
    outs() << "\n";
    double pct = (Stats.NumBits * 100.0) / BufferSizeBits;
    outs() << "    Percent of file: " << format("%2.4f%%", pct) << "\n";
    if (Stats.NumInstances > 1) {
      outs() << "       Average Size: ";
      PrintSize(Stats.NumBits/(double)Stats.NumInstances);
      outs() << "\n";
      outs() << "  Tot/Avg SubBlocks: " << Stats.NumSubBlocks << "/"
             << Stats.NumSubBlocks/(double)Stats.NumInstances << "\n";
      outs() << "    Tot/Avg Abbrevs: " << Stats.NumAbbrevs << "/"
             << Stats.NumAbbrevs/(double)Stats.NumInstances << "\n";
      outs() << "    Tot/Avg Records: " << Stats.NumRecords << "/"
             << Stats.NumRecords/(double)Stats.NumInstances << "\n";
    } else {
      outs() << "      Num SubBlocks: " << Stats.NumSubBlocks << "\n";
      outs() << "        Num Abbrevs: " << Stats.NumAbbrevs << "\n";
      outs() << "        Num Records: " << Stats.NumRecords << "\n";
    }
    if (Stats.NumRecords) {
      double pct = (Stats.NumAbbreviatedRecords * 100.0) / Stats.NumRecords;
      outs() << "    Percent Abbrevs: " << format("%2.4f%%", pct) << "\n";
    }
    outs() << "\n";

    // Print a histogram of the codes we see.
    if (!NoHistogram && !Stats.CodeFreq.empty()) {
      std::vector<std::pair<unsigned, unsigned> > FreqPairs;  // <freq,code>
      for (unsigned i = 0, e = Stats.CodeFreq.size(); i != e; ++i)
        if (unsigned Freq = Stats.CodeFreq[i].NumInstances)
          FreqPairs.push_back(std::make_pair(Freq, i));
      std::stable_sort(FreqPairs.begin(), FreqPairs.end());
      std::reverse(FreqPairs.begin(), FreqPairs.end());

      outs() << "\tRecord Histogram:\n";
      outs() << "\t\t  Count    # Bits   %% Abv  Record Kind\n";
      for (unsigned i = 0, e = FreqPairs.size(); i != e; ++i) {
        const PerRecordStats &RecStats = Stats.CodeFreq[FreqPairs[i].second];

        outs() << format("\t\t%7d %9lu",
                         RecStats.NumInstances,
                         (unsigned long)RecStats.TotalBits);

        if (RecStats.NumAbbrev)
          outs() <<
              format("%7.2f  ",
                     (double)RecStats.NumAbbrev/RecStats.NumInstances*100);
        else
          outs() << "         ";

        if (const char *CodeName =
              GetCodeName(FreqPairs[i].second, I->first, StreamFile,
                          CurStreamType))
          outs() << CodeName << "\n";
        else
          outs() << "UnknownCode" << FreqPairs[i].second << "\n";
      }
      outs() << "\n";

    }
  }
  return 0;
}


int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, "llvm-bcanalyzer file analyzer\n");

  return AnalyzeBitcode();
}
