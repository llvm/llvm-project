//===- MCDwarf2BTF.cpp ---------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCDwarf2BTF.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCBTFContext.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"
#include <fstream>

using namespace llvm;

void MCDwarf2BTF::addFiles(MCObjectStreamer *MCOS, std::string &FileName,
  std::vector<FileContent> &Files) {
  std::vector<std::string> Content;

  std::ifstream Inputfile(FileName);
  std::string Line;
  Content.push_back(Line); // line 0 for empty string
  while (std::getline(Inputfile, Line))
    Content.push_back(Line);

  Files.push_back(FileContent(FileName, Content));
}

void MCDwarf2BTF::addLines(MCObjectStreamer *MCOS, StringRef &SectionName,
  std::vector<FileContent> &Files,
  const MCLineSection::MCDwarfLineEntryCollection &LineEntries) {
  MCContext &Context = MCOS->getContext();
  auto &BTFCxt = Context.getBTFContext();

  unsigned SecNameOff = BTFCxt->addString(SectionName.str());
  for (const MCDwarfLineEntry &LineEntry : LineEntries) {
    BTFLineInfo LineInfo;
    unsigned FileNum = LineEntry.getFileNum();
    unsigned Line = LineEntry.getLine();

    LineInfo.Label = LineEntry.getLabel();
    if (FileNum < Files.size()) {
      LineInfo.FileNameOff = BTFCxt->addString(Files[FileNum].first);
      if (Line < Files[FileNum].second.size())
        LineInfo.LineOff = BTFCxt->addString(Files[FileNum].second[Line]);
      else
        LineInfo.LineOff = 0;
    } else {
      LineInfo.FileNameOff = 0;
      LineInfo.LineOff = 0;
    }
    LineInfo.LineNum = Line;
    LineInfo.ColumnNum = LineEntry.getColumn();
    BTFCxt->addLineInfo(SecNameOff, LineInfo);
  }
}

void MCDwarf2BTF::addDwarfLineInfo(MCObjectStreamer *MCOS) {
  MCContext &Context = MCOS->getContext();

  auto &LineTables = Context.getMCDwarfLineTables();
  if (LineTables.empty())
    return;

  for (const auto &CUIDTablePair : LineTables) {
    std::vector<std::string> Dirs;
    std::vector<FileContent> Files;

    for (auto &Dir : CUIDTablePair.second.getMCDwarfDirs())
      Dirs.push_back(Dir);
    for (auto &File : CUIDTablePair.second.getMCDwarfFiles()) {
      std::string FileName;
      if (File.DirIndex == 0)
        FileName = File.Name;
      else
        FileName = Dirs[File.DirIndex - 1] + "/" + File.Name;
      MCDwarf2BTF::addFiles(MCOS, FileName, Files);
    }
    for (const auto &LineSec: CUIDTablePair.second.getMCLineSections().getMCLineEntries()) {
      MCSection *Section = LineSec.first;
      const MCLineSection::MCDwarfLineEntryCollection &LineEntries = LineSec.second;

      StringRef SectionName;
      if (MCSectionELF *SectionELF = dyn_cast<MCSectionELF>(Section))
        SectionName = SectionELF->getSectionName();
      else
        return;
      MCDwarf2BTF::addLines(MCOS, SectionName, Files, LineEntries);
    }
  }
}
