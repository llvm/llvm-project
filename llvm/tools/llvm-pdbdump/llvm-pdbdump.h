//===- llvm-pdbdump.h ----------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_LLVMPDBDUMP_H
#define LLVM_TOOLS_LLVMPDBDUMP_LLVMPDBDUMP_H

#include "llvm/ADT/Optional.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

namespace opts {

namespace pretty {
extern llvm::cl::opt<bool> Compilands;
extern llvm::cl::opt<bool> Symbols;
extern llvm::cl::opt<bool> Globals;
extern llvm::cl::opt<bool> Types;
extern llvm::cl::opt<bool> All;
extern llvm::cl::opt<bool> ExcludeCompilerGenerated;

extern llvm::cl::opt<bool> NoClassDefs;
extern llvm::cl::opt<bool> NoEnumDefs;
extern llvm::cl::list<std::string> ExcludeTypes;
extern llvm::cl::list<std::string> ExcludeSymbols;
extern llvm::cl::list<std::string> ExcludeCompilands;
extern llvm::cl::list<std::string> IncludeTypes;
extern llvm::cl::list<std::string> IncludeSymbols;
extern llvm::cl::list<std::string> IncludeCompilands;
}

namespace raw {
struct BlockRange {
  uint32_t Min;
  llvm::Optional<uint32_t> Max;
};

extern llvm::Optional<BlockRange> DumpBlockRange;
extern llvm::cl::list<uint32_t> DumpStreamData;

extern llvm::cl::opt<bool> DumpHeaders;
extern llvm::cl::opt<bool> DumpStreamBlocks;
extern llvm::cl::opt<bool> DumpStreamSummary;
extern llvm::cl::opt<bool> DumpPageStats;
extern llvm::cl::opt<bool> DumpTpiHash;
extern llvm::cl::opt<bool> DumpTpiRecordBytes;
extern llvm::cl::opt<bool> DumpTpiRecords;
extern llvm::cl::opt<bool> DumpIpiRecords;
extern llvm::cl::opt<bool> DumpIpiRecordBytes;
extern llvm::cl::opt<bool> DumpModules;
extern llvm::cl::opt<bool> DumpModuleFiles;
extern llvm::cl::opt<bool> DumpModuleSyms;
extern llvm::cl::opt<bool> DumpPublics;
extern llvm::cl::opt<bool> DumpSectionContribs;
extern llvm::cl::opt<bool> DumpLineInfo;
extern llvm::cl::opt<bool> DumpSectionMap;
extern llvm::cl::opt<bool> DumpSymRecordBytes;
extern llvm::cl::opt<bool> DumpSectionHeaders;
extern llvm::cl::opt<bool> DumpFpo;
}

namespace pdb2yaml {
extern llvm::cl::opt<bool> NoFileHeaders;
extern llvm::cl::opt<bool> StreamMetadata;
extern llvm::cl::opt<bool> StreamDirectory;
extern llvm::cl::opt<bool> PdbStream;
extern llvm::cl::opt<bool> DbiStream;
extern llvm::cl::opt<bool> DbiModuleInfo;
extern llvm::cl::opt<bool> DbiModuleSourceFileInfo;
extern llvm::cl::opt<bool> TpiStream;
extern llvm::cl::opt<bool> IpiStream;
extern llvm::cl::list<std::string> InputFilename;
}
}

#endif
