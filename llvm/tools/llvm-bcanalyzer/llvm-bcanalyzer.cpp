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

#include "llvm/ADT/OwningPtr.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/system_error.h"
#include <cstdio>
#include <map>
#include <algorithm>
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

namespace {

/// CurStreamTypeType - A type for CurStreamType
enum CurStreamTypeType {
  UnknownBitstream,
  LLVMIRBitstream
};

}

/// CurStreamType - If we can sniff the flavor of this stream, we can produce
/// better dump info.
static CurStreamTypeType CurStreamType;


/// GetBlockName - Return a symbolic block name if known, otherwise return
/// null.
static const char *GetBlockName(unsigned BlockID,
                                const BitstreamReader &StreamFile) {
  // Standard blocks for all bitcode files.
  if (BlockID < bitc::FIRST_APPLICATION_BLOCKID) {
    if (BlockID == bitc::BLOCKINFO_BLOCK_ID)
      return "BLOCKINFO_BLOCK";
    return 0;
  }

  // Check to see if we have a blockinfo record for this block, with a name.
  if (const BitstreamReader::BlockInfo *Info =
        StreamFile.getBlockInfo(BlockID)) {
    if (!Info->Name.empty())
      return Info->Name.c_str();
  }


  if (CurStreamType != LLVMIRBitstream) return 0;

  switch (BlockID) {
  default:                           return 0;
  case bitc::MODULE_BLOCK_ID:        return "MODULE_BLOCK";
  case bitc::PARAMATTR_BLOCK_ID:     return "PARAMATTR_BLOCK";
  case bitc::TYPE_BLOCK_ID:          return "TYPE_BLOCK";
  case bitc::CONSTANTS_BLOCK_ID:     return "CONSTANTS_BLOCK";
  case bitc::FUNCTION_BLOCK_ID:      return "FUNCTION_BLOCK";
  case bitc::TYPE_SYMTAB_BLOCK_ID:   return "TYPE_SYMTAB";
  case bitc::VALUE_SYMTAB_BLOCK_ID:  return "VALUE_SYMTAB";
  case bitc::METADATA_BLOCK_ID:      return "METADATA_BLOCK";
  case bitc::METADATA_ATTACHMENT_ID: return "METADATA_ATTACHMENT_BLOCK";
  }
}

/// GetCodeName - Return a symbolic code name if known, otherwise return
/// null.
static const char *GetCodeName(unsigned CodeID, unsigned BlockID,
                               const BitstreamReader &StreamFile) {
  // Standard blocks for all bitcode files.
  if (BlockID < bitc::FIRST_APPLICATION_BLOCKID) {
    if (BlockID == bitc::BLOCKINFO_BLOCK_ID) {
      switch (CodeID) {
      default: return 0;
      case bitc::BLOCKINFO_CODE_SETBID:        return "SETBID";
      case bitc::BLOCKINFO_CODE_BLOCKNAME:     return "BLOCKNAME";
      case bitc::BLOCKINFO_CODE_SETRECORDNAME: return "SETRECORDNAME";
      }
    }
    return 0;
  }

  // Check to see if we have a blockinfo record for this record, with a name.
  if (const BitstreamReader::BlockInfo *Info =
        StreamFile.getBlockInfo(BlockID)) {
    for (unsigned i = 0, e = Info->RecordNames.size(); i != e; ++i)
      if (Info->RecordNames[i].first == CodeID)
        return Info->RecordNames[i].second.c_str();
  }


  if (CurStreamType != LLVMIRBitstream) return 0;

  switch (BlockID) {
  default: return 0;
  case bitc::MODULE_BLOCK_ID:
    switch (CodeID) {
    default: return 0;
    case bitc::MODULE_CODE_VERSION:     return "VERSION";
    case bitc::MODULE_CODE_TRIPLE:      return "TRIPLE";
    case bitc::MODULE_CODE_DATALAYOUT:  return "DATALAYOUT";
    case bitc::MODULE_CODE_ASM:         return "ASM";
    case bitc::MODULE_CODE_SECTIONNAME: return "SECTIONNAME";
    case bitc::MODULE_CODE_DEPLIB:      return "DEPLIB";
    case bitc::MODULE_CODE_GLOBALVAR:   return "GLOBALVAR";
    case bitc::MODULE_CODE_FUNCTION:    return "FUNCTION";
    case bitc::MODULE_CODE_ALIAS:       return "ALIAS";
    case bitc::MODULE_CODE_PURGEVALS:   return "PURGEVALS";
    case bitc::MODULE_CODE_GCNAME:      return "GCNAME";
    }
  case bitc::PARAMATTR_BLOCK_ID:
    switch (CodeID) {
    default: return 0;
    case bitc::PARAMATTR_CODE_ENTRY: return "ENTRY";
    }
  case bitc::TYPE_BLOCK_ID:
    switch (CodeID) {
    default: return 0;
    case bitc::TYPE_CODE_NUMENTRY:  return "NUMENTRY";
    case bitc::TYPE_CODE_VOID:      return "VOID";
    case bitc::TYPE_CODE_FLOAT:     return "FLOAT";
    case bitc::TYPE_CODE_DOUBLE:    return "DOUBLE";
    case bitc::TYPE_CODE_LABEL:     return "LABEL";
    case bitc::TYPE_CODE_OPAQUE:    return "OPAQUE";
    case bitc::TYPE_CODE_INTEGER:   return "INTEGER";
    case bitc::TYPE_CODE_POINTER:   return "POINTER";
    case bitc::TYPE_CODE_FUNCTION:  return "FUNCTION";
    case bitc::TYPE_CODE_STRUCT:    return "STRUCT";
    case bitc::TYPE_CODE_ARRAY:     return "ARRAY";
    case bitc::TYPE_CODE_VECTOR:    return "VECTOR";
    case bitc::TYPE_CODE_X86_FP80:  return "X86_FP80";
    case bitc::TYPE_CODE_FP128:     return "FP128";
    case bitc::TYPE_CODE_PPC_FP128: return "PPC_FP128";
    case bitc::TYPE_CODE_METADATA:  return "METADATA";
    }

  case bitc::CONSTANTS_BLOCK_ID:
    switch (CodeID) {
    default: return 0;
    case bitc::CST_CODE_SETTYPE:         return "SETTYPE";
    case bitc::CST_CODE_NULL:            return "NULL";
    case bitc::CST_CODE_UNDEF:           return "UNDEF";
    case bitc::CST_CODE_INTEGER:         return "INTEGER";
    case bitc::CST_CODE_WIDE_INTEGER:    return "WIDE_INTEGER";
    case bitc::CST_CODE_FLOAT:           return "FLOAT";
    case bitc::CST_CODE_AGGREGATE:       return "AGGREGATE";
    case bitc::CST_CODE_STRING:          return "STRING";
    case bitc::CST_CODE_CSTRING:         return "CSTRING";
    case bitc::CST_CODE_CE_BINOP:        return "CE_BINOP";
    case bitc::CST_CODE_CE_CAST:         return "CE_CAST";
    case bitc::CST_CODE_CE_GEP:          return "CE_GEP";
    case bitc::CST_CODE_CE_INBOUNDS_GEP: return "CE_INBOUNDS_GEP";
    case bitc::CST_CODE_CE_SELECT:       return "CE_SELECT";
    case bitc::CST_CODE_CE_EXTRACTELT:   return "CE_EXTRACTELT";
    case bitc::CST_CODE_CE_INSERTELT:    return "CE_INSERTELT";
    case bitc::CST_CODE_CE_SHUFFLEVEC:   return "CE_SHUFFLEVEC";
    case bitc::CST_CODE_CE_CMP:          return "CE_CMP";
    case bitc::CST_CODE_INLINEASM:       return "INLINEASM";
    case bitc::CST_CODE_CE_SHUFVEC_EX:   return "CE_SHUFVEC_EX";
    }
  case bitc::FUNCTION_BLOCK_ID:
    switch (CodeID) {
    default: return 0;
    case bitc::FUNC_CODE_DECLAREBLOCKS: return "DECLAREBLOCKS";

    case bitc::FUNC_CODE_INST_BINOP:        return "INST_BINOP";
    case bitc::FUNC_CODE_INST_CAST:         return "INST_CAST";
    case bitc::FUNC_CODE_INST_GEP:          return "INST_GEP";
    case bitc::FUNC_CODE_INST_INBOUNDS_GEP: return "INST_INBOUNDS_GEP";
    case bitc::FUNC_CODE_INST_SELECT:       return "INST_SELECT";
    case bitc::FUNC_CODE_INST_EXTRACTELT:   return "INST_EXTRACTELT";
    case bitc::FUNC_CODE_INST_INSERTELT:    return "INST_INSERTELT";
    case bitc::FUNC_CODE_INST_SHUFFLEVEC:   return "INST_SHUFFLEVEC";
    case bitc::FUNC_CODE_INST_CMP:          return "INST_CMP";

    case bitc::FUNC_CODE_INST_RET:          return "INST_RET";
    case bitc::FUNC_CODE_INST_BR:           return "INST_BR";
    case bitc::FUNC_CODE_INST_SWITCH:       return "INST_SWITCH";
    case bitc::FUNC_CODE_INST_INVOKE:       return "INST_INVOKE";
    case bitc::FUNC_CODE_INST_UNWIND:       return "INST_UNWIND";
    case bitc::FUNC_CODE_INST_UNREACHABLE:  return "INST_UNREACHABLE";

    case bitc::FUNC_CODE_INST_PHI:          return "INST_PHI";
    case bitc::FUNC_CODE_INST_MALLOC:       return "INST_MALLOC";
    case bitc::FUNC_CODE_INST_FREE:         return "INST_FREE";
    case bitc::FUNC_CODE_INST_ALLOCA:       return "INST_ALLOCA";
    case bitc::FUNC_CODE_INST_LOAD:         return "INST_LOAD";
    case bitc::FUNC_CODE_INST_STORE:        return "INST_STORE";
    case bitc::FUNC_CODE_INST_CALL:         return "INST_CALL";
    case bitc::FUNC_CODE_INST_VAARG:        return "INST_VAARG";
    case bitc::FUNC_CODE_INST_STORE2:       return "INST_STORE2";
    case bitc::FUNC_CODE_INST_GETRESULT:    return "INST_GETRESULT";
    case bitc::FUNC_CODE_INST_EXTRACTVAL:   return "INST_EXTRACTVAL";
    case bitc::FUNC_CODE_INST_INSERTVAL:    return "INST_INSERTVAL";
    case bitc::FUNC_CODE_INST_CMP2:         return "INST_CMP2";
    case bitc::FUNC_CODE_INST_VSELECT:      return "INST_VSELECT";
    case bitc::FUNC_CODE_DEBUG_LOC:         return "DEBUG_LOC";
    case bitc::FUNC_CODE_DEBUG_LOC_AGAIN:   return "DEBUG_LOC_AGAIN";
    case bitc::FUNC_CODE_INST_CALL2:        return "INST_CALL2";
    case bitc::FUNC_CODE_DEBUG_LOC2:        return "DEBUG_LOC2";
    }
  case bitc::TYPE_SYMTAB_BLOCK_ID:
    switch (CodeID) {
    default: return 0;
    case bitc::TST_CODE_ENTRY: return "ENTRY";
    }
  case bitc::VALUE_SYMTAB_BLOCK_ID:
    switch (CodeID) {
    default: return 0;
    case bitc::VST_CODE_ENTRY: return "ENTRY";
    case bitc::VST_CODE_BBENTRY: return "BBENTRY";
    }
  case bitc::METADATA_ATTACHMENT_ID:
    switch(CodeID) {
    default:return 0;
    case bitc::METADATA_ATTACHMENT:  return "METADATA_ATTACHMENT";
    case bitc::METADATA_ATTACHMENT2: return "METADATA_ATTACHMENT2";
    }
  case bitc::METADATA_BLOCK_ID:
    switch(CodeID) {
    default:return 0;
    case bitc::METADATA_STRING:      return "METADATA_STRING";
    case bitc::METADATA_NODE:        return "METADATA_NODE";
    case bitc::METADATA_FN_NODE:     return "METADATA_FN_NODE";
    case bitc::METADATA_NAME:        return "METADATA_NAME";
    case bitc::METADATA_NAMED_NODE:  return "METADATA_NAMED_NODE";
    case bitc::METADATA_KIND:        return "METADATA_KIND";
    case bitc::METADATA_ATTACHMENT:  return "METADATA_ATTACHMENT";
    case bitc::METADATA_NODE2:       return "METADATA_NODE2";
    case bitc::METADATA_FN_NODE2:    return "METADATA_FN_NODE2";
    case bitc::METADATA_NAMED_NODE2: return "METADATA_NAMED_NODE2";
    }
  }
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
static bool Error(const std::string &Err) {
  errs() << Err << "\n";
  return true;
}

/// ParseBlock - Read a block, updating statistics, etc.
static bool ParseBlock(BitstreamCursor &Stream, unsigned IndentLevel) {
  std::string Indent(IndentLevel*2, ' ');
  uint64_t BlockBitStart = Stream.GetCurrentBitNo();
  unsigned BlockID = Stream.ReadSubBlockID();

  // Get the statistics for this BlockID.
  PerBlockIDStats &BlockStats = BlockIDStats[BlockID];

  BlockStats.NumInstances++;

  // BLOCKINFO is a special part of the stream.
  if (BlockID == bitc::BLOCKINFO_BLOCK_ID) {
    if (Dump) errs() << Indent << "<BLOCKINFO_BLOCK/>\n";
    if (Stream.ReadBlockInfoBlock())
      return Error("Malformed BlockInfoBlock");
    uint64_t BlockBitEnd = Stream.GetCurrentBitNo();
    BlockStats.NumBits += BlockBitEnd-BlockBitStart;
    return false;
  }

  unsigned NumWords = 0;
  if (Stream.EnterSubBlock(BlockID, &NumWords))
    return Error("Malformed block record");

  const char *BlockName = 0;
  if (Dump) {
    errs() << Indent << "<";
    if ((BlockName = GetBlockName(BlockID, *Stream.getBitStreamReader())))
      errs() << BlockName;
    else
      errs() << "UnknownBlock" << BlockID;

    if (NonSymbolic && BlockName)
      errs() << " BlockID=" << BlockID;

    errs() << " NumWords=" << NumWords
           << " BlockCodeSize=" << Stream.GetAbbrevIDWidth() << ">\n";
  }

  SmallVector<uint64_t, 64> Record;

  // Read all the records for this block.
  while (1) {
    if (Stream.AtEndOfStream())
      return Error("Premature end of bitstream");

    uint64_t RecordStartBit = Stream.GetCurrentBitNo();

    // Read the code for this record.
    unsigned AbbrevID = Stream.ReadCode();
    switch (AbbrevID) {
    case bitc::END_BLOCK: {
      if (Stream.ReadBlockEnd())
        return Error("Error at end of block");
      uint64_t BlockBitEnd = Stream.GetCurrentBitNo();
      BlockStats.NumBits += BlockBitEnd-BlockBitStart;
      if (Dump) {
        errs() << Indent << "</";
        if (BlockName)
          errs() << BlockName << ">\n";
        else
          errs() << "UnknownBlock" << BlockID << ">\n";
      }
      return false;
    }
    case bitc::ENTER_SUBBLOCK: {
      uint64_t SubBlockBitStart = Stream.GetCurrentBitNo();
      if (ParseBlock(Stream, IndentLevel+1))
        return true;
      ++BlockStats.NumSubBlocks;
      uint64_t SubBlockBitEnd = Stream.GetCurrentBitNo();

      // Don't include subblock sizes in the size of this block.
      BlockBitStart += SubBlockBitEnd-SubBlockBitStart;
      break;
    }
    case bitc::DEFINE_ABBREV:
      Stream.ReadAbbrevRecord();
      ++BlockStats.NumAbbrevs;
      break;
    default:
      Record.clear();

      ++BlockStats.NumRecords;
      if (AbbrevID != bitc::UNABBREV_RECORD)
        ++BlockStats.NumAbbreviatedRecords;

      const char *BlobStart = 0;
      unsigned BlobLen = 0;
      unsigned Code = Stream.ReadRecord(AbbrevID, Record, BlobStart, BlobLen);



      // Increment the # occurrences of this code.
      if (BlockStats.CodeFreq.size() <= Code)
        BlockStats.CodeFreq.resize(Code+1);
      BlockStats.CodeFreq[Code].NumInstances++;
      BlockStats.CodeFreq[Code].TotalBits +=
        Stream.GetCurrentBitNo()-RecordStartBit;
      if (AbbrevID != bitc::UNABBREV_RECORD)
        BlockStats.CodeFreq[Code].NumAbbrev++;

      if (Dump) {
        errs() << Indent << "  <";
        if (const char *CodeName =
              GetCodeName(Code, BlockID, *Stream.getBitStreamReader()))
          errs() << CodeName;
        else
          errs() << "UnknownCode" << Code;
        if (NonSymbolic &&
            GetCodeName(Code, BlockID, *Stream.getBitStreamReader()))
          errs() << " codeid=" << Code;
        if (AbbrevID != bitc::UNABBREV_RECORD)
          errs() << " abbrevid=" << AbbrevID;

        for (unsigned i = 0, e = Record.size(); i != e; ++i)
          errs() << " op" << i << "=" << (int64_t)Record[i];

        errs() << "/>";

        if (BlobStart) {
          errs() << " blob data = ";
          bool BlobIsPrintable = true;
          for (unsigned i = 0; i != BlobLen; ++i)
            if (!isprint(BlobStart[i])) {
              BlobIsPrintable = false;
              break;
            }

          if (BlobIsPrintable)
            errs() << "'" << std::string(BlobStart, BlobStart+BlobLen) <<"'";
          else
            errs() << "unprintable, " << BlobLen << " bytes.";
        }

        errs() << "\n";
      }

      break;
    }
  }
}

static void PrintSize(double Bits) {
  fprintf(stderr, "%.2f/%.2fB/%luW", Bits, Bits/8,(unsigned long)(Bits/32));
}
static void PrintSize(uint64_t Bits) {
  fprintf(stderr, "%lub/%.2fB/%luW", (unsigned long)Bits,
          (double)Bits/8, (unsigned long)(Bits/32));
}


/// AnalyzeBitcode - Analyze the bitcode file specified by InputFilename.
static int AnalyzeBitcode() {
  // Read the input file.
  OwningPtr<MemoryBuffer> MemBuf;

  if (error_code ec =
        MemoryBuffer::getFileOrSTDIN(InputFilename.c_str(), MemBuf))
    return Error("Error reading '" + InputFilename + "': " + ec.message());

  if (MemBuf->getBufferSize() & 3)
    return Error("Bitcode stream should be a multiple of 4 bytes in length");

  unsigned char *BufPtr = (unsigned char *)MemBuf->getBufferStart();
  unsigned char *EndBufPtr = BufPtr+MemBuf->getBufferSize();

  // If we have a wrapper header, parse it and ignore the non-bc file contents.
  // The magic number is 0x0B17C0DE stored in little endian.
  if (isBitcodeWrapper(BufPtr, EndBufPtr))
    if (SkipBitcodeWrapperHeader(BufPtr, EndBufPtr))
      return Error("Invalid bitcode wrapper header");

  BitstreamReader StreamFile(BufPtr, EndBufPtr);
  BitstreamCursor Stream(StreamFile);
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

  unsigned NumTopBlocks = 0;

  // Parse the top-level structure.  We only allow blocks at the top-level.
  while (!Stream.AtEndOfStream()) {
    unsigned Code = Stream.ReadCode();
    if (Code != bitc::ENTER_SUBBLOCK)
      return Error("Invalid record at top-level");

    if (ParseBlock(Stream, 0))
      return true;
    ++NumTopBlocks;
  }

  if (Dump) errs() << "\n\n";

  uint64_t BufferSizeBits = (EndBufPtr-BufPtr)*CHAR_BIT;
  // Print a summary of the read file.
  errs() << "Summary of " << InputFilename << ":\n";
  errs() << "         Total size: ";
  PrintSize(BufferSizeBits);
  errs() << "\n";
  errs() << "        Stream type: ";
  switch (CurStreamType) {
  default: assert(0 && "Unknown bitstream type");
  case UnknownBitstream: errs() << "unknown\n"; break;
  case LLVMIRBitstream:  errs() << "LLVM IR\n"; break;
  }
  errs() << "  # Toplevel Blocks: " << NumTopBlocks << "\n";
  errs() << "\n";

  // Emit per-block stats.
  errs() << "Per-block Summary:\n";
  for (std::map<unsigned, PerBlockIDStats>::iterator I = BlockIDStats.begin(),
       E = BlockIDStats.end(); I != E; ++I) {
    errs() << "  Block ID #" << I->first;
    if (const char *BlockName = GetBlockName(I->first, StreamFile))
      errs() << " (" << BlockName << ")";
    errs() << ":\n";

    const PerBlockIDStats &Stats = I->second;
    errs() << "      Num Instances: " << Stats.NumInstances << "\n";
    errs() << "         Total Size: ";
    PrintSize(Stats.NumBits);
    errs() << "\n";
    double pct = (Stats.NumBits * 100.0) / BufferSizeBits;
    errs() << "    Percent of file: " << format("%2.4f%%", pct) << "\n";
    if (Stats.NumInstances > 1) {
      errs() << "       Average Size: ";
      PrintSize(Stats.NumBits/(double)Stats.NumInstances);
      errs() << "\n";
      errs() << "  Tot/Avg SubBlocks: " << Stats.NumSubBlocks << "/"
             << Stats.NumSubBlocks/(double)Stats.NumInstances << "\n";
      errs() << "    Tot/Avg Abbrevs: " << Stats.NumAbbrevs << "/"
             << Stats.NumAbbrevs/(double)Stats.NumInstances << "\n";
      errs() << "    Tot/Avg Records: " << Stats.NumRecords << "/"
             << Stats.NumRecords/(double)Stats.NumInstances << "\n";
    } else {
      errs() << "      Num SubBlocks: " << Stats.NumSubBlocks << "\n";
      errs() << "        Num Abbrevs: " << Stats.NumAbbrevs << "\n";
      errs() << "        Num Records: " << Stats.NumRecords << "\n";
    }
    if (Stats.NumRecords) {
      double pct = (Stats.NumAbbreviatedRecords * 100.0) / Stats.NumRecords;
      errs() << "    Percent Abbrevs: " << format("%2.4f%%", pct) << "\n";
    }
    errs() << "\n";

    // Print a histogram of the codes we see.
    if (!NoHistogram && !Stats.CodeFreq.empty()) {
      std::vector<std::pair<unsigned, unsigned> > FreqPairs;  // <freq,code>
      for (unsigned i = 0, e = Stats.CodeFreq.size(); i != e; ++i)
        if (unsigned Freq = Stats.CodeFreq[i].NumInstances)
          FreqPairs.push_back(std::make_pair(Freq, i));
      std::stable_sort(FreqPairs.begin(), FreqPairs.end());
      std::reverse(FreqPairs.begin(), FreqPairs.end());

      errs() << "\tRecord Histogram:\n";
      fprintf(stderr, "\t\t  Count    # Bits   %% Abv  Record Kind\n");
      for (unsigned i = 0, e = FreqPairs.size(); i != e; ++i) {
        const PerRecordStats &RecStats = Stats.CodeFreq[FreqPairs[i].second];

        fprintf(stderr, "\t\t%7d %9lu ", RecStats.NumInstances,
                (unsigned long)RecStats.TotalBits);

        if (RecStats.NumAbbrev)
          fprintf(stderr, "%7.2f  ",
                  (double)RecStats.NumAbbrev/RecStats.NumInstances*100);
        else
          fprintf(stderr, "         ");

        if (const char *CodeName =
              GetCodeName(FreqPairs[i].second, I->first, StreamFile))
          fprintf(stderr, "%s\n", CodeName);
        else
          fprintf(stderr, "UnknownCode%d\n", FreqPairs[i].second);
      }
      errs() << "\n";

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
