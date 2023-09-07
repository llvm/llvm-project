//===------------ MachOBuilder.h -- Build MachO Objects ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Build MachO object files for interaction with the ObjC runtime and debugger.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_MACHOBUILDER_H
#define LLVM_EXECUTIONENGINE_ORC_MACHOBUILDER_H

#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MathExtras.h"

#include <list>
#include <map>
#include <vector>

namespace llvm {
namespace orc {

// Builds MachO objects.
template <typename MachOTraits> class MachOBuilder {
private:
  struct BufferWriter {
  public:
    BufferWriter(MutableArrayRef<char> Buffer)
        : Data(Buffer.data()), Size(Buffer.size()) {}

    size_t tell() const { return Offset; }

    void write(char C) {
      assert(Offset < Size && "Buffer overflow");
      Data[Offset++] = C;
    }

    void write(const char *Src, size_t SrcSize) {
      assert(Offset + SrcSize <= Size && "Buffer overflow");
      memcpy(&Data[Offset], Src, SrcSize);
      Offset += SrcSize;
    }

    template <typename T> void write(const T &Value) {
      assert(Offset + sizeof(T) <= Size && "Buffer overflow");
      memcpy(&Data[Offset], reinterpret_cast<const char *>(&Value), sizeof(T));
      Offset += sizeof(T);
    }

  private:
    char *Data = 0;
    size_t Offset = 0;
    size_t Size = 0;
  };

  struct SymbolContainer {
    size_t SymbolIndexBase = 0;
    std::vector<typename MachOTraits::NList> Symbols;
  };

  static void writeMachOStruct(BufferWriter &BW, MachO::relocation_info RI) {
    BW.write(RI);
  }

  template <typename MachOStruct>
  static void writeMachOStruct(BufferWriter &BW, MachOStruct S) {
    if (MachOTraits::Endianness != support::endian::system_endianness())
      MachO::swapStruct(S);
    BW.write(S);
  }

  struct LoadCommandBase {
    virtual ~LoadCommandBase() {}
    virtual size_t size() const = 0;
    virtual void write(BufferWriter &BW) = 0;
  };

  template <MachO::LoadCommandType LCType> struct LoadCommand;

#define HANDLE_LOAD_COMMAND(Name, Value, LCStruct)                             \
  template <>                                                                  \
  struct LoadCommand<MachO::Name> : public MachO::LCStruct,                    \
                                    public LoadCommandBase {                   \
    using CmdStruct = LCStruct;                                                \
    LoadCommand() {                                                            \
      memset(&rawStruct(), 0, sizeof(CmdStruct));                              \
      cmd = Value;                                                             \
      cmdsize = sizeof(CmdStruct);                                             \
    }                                                                          \
    template <typename... ArgTs>                                               \
    LoadCommand(ArgTs &&...Args)                                               \
        : CmdStruct{Value, sizeof(CmdStruct), std::forward<ArgTs>(Args)...} {} \
    CmdStruct &rawStruct() { return static_cast<CmdStruct &>(*this); }         \
    size_t size() const override { return cmdsize; }                           \
    void write(BufferWriter &BW) override {                                    \
      writeMachOStruct(BW, rawStruct());                                       \
    }                                                                          \
  };

#include "llvm/BinaryFormat/MachO.def"

#undef HANDLE_LOAD_COMMAND

  struct StringTableEntry {
    StringRef S;
    size_t Offset;
  };

  using StringTable = std::vector<StringTableEntry>;

public:
  using StringId = size_t;

  struct Section;

  // Points to either an nlist entry (as a (symbol-container, index) pair), or
  // a section.
  class RelocTarget {
  public:
    RelocTarget(const Section &S) : S(&S), Idx(~0U) {}
    RelocTarget(SymbolContainer &SC, size_t Idx) : SC(&SC), Idx(Idx) {}

    bool isSymbol() { return Idx != ~0U; }

    uint32_t getSymbolNum() {
      assert(isSymbol() && "Target is not a symbol");
      return SC->SymbolIndexBase + Idx;
    }

    uint32_t getSectionId() {
      assert(!isSymbol() && "Target is not a section");
      return S->SectionNumber;
    }

    typename MachOTraits::NList &nlist() {
      assert(isSymbol() && "Target is not a symbol");
      return SC->Symbols[Idx];
    }

  private:
    union {
      const Section *S;
      SymbolContainer *SC;
    };
    size_t Idx;
  };

  struct Reloc : public MachO::relocation_info {
    RelocTarget Target;

    Reloc(int32_t Offset, RelocTarget Target, bool PCRel, unsigned Length,
          unsigned Type)
        : Target(Target) {
      assert(Type < 16 && "Relocation type out of range");
      r_address = Offset; // Will slide to account for sec addr during layout
      r_symbolnum = 0;
      r_pcrel = PCRel;
      r_length = Length;
      r_extern = Target.isSymbol();
      r_type = Type;
    }

    MachO::relocation_info &rawStruct() {
      return static_cast<MachO::relocation_info &>(*this);
    }
  };

  struct SectionContent {
    const char *Data = nullptr;
    size_t Size = 0;
  };

  struct Section : public MachOTraits::Section, public RelocTarget {
    MachOBuilder &Builder;
    SectionContent Content;
    size_t SectionNumber = 0;
    SymbolContainer SC;
    std::vector<Reloc> Relocs;

    Section(MachOBuilder &Builder, StringRef SecName, StringRef SegName)
        : RelocTarget(*this), Builder(Builder) {
      memset(&rawStruct(), 0, sizeof(typename MachOTraits::Section));
      assert(SecName.size() <= 16 && "SecName too long");
      assert(SegName.size() <= 16 && "SegName too long");
      memcpy(this->sectname, SecName.data(), SecName.size());
      memcpy(this->segname, SegName.data(), SegName.size());
    }

    RelocTarget addSymbol(int32_t Offset, StringRef Name, uint8_t Type,
                          uint16_t Desc) {
      StringId SI = Builder.addString(Name);
      typename MachOTraits::NList Sym;
      Sym.n_strx = SI;
      Sym.n_type = Type | MachO::N_SECT;
      Sym.n_sect = MachO::NO_SECT; // Will be filled in later.
      Sym.n_desc = Desc;
      Sym.n_value = Offset;
      SC.Symbols.push_back(Sym);
      return {SC, SC.Symbols.size() - 1};
    }

    void addReloc(int32_t Offset, RelocTarget Target, bool PCRel,
                  unsigned Length, unsigned Type) {
      Relocs.push_back({Offset, Target, PCRel, Length, Type});
    }

    auto &rawStruct() {
      return static_cast<typename MachOTraits::Section &>(*this);
    }
  };

  struct Segment : public LoadCommand<MachOTraits::SegmentCmd> {
    MachOBuilder &Builder;
    std::vector<std::unique_ptr<Section>> Sections;

    Segment(MachOBuilder &Builder, StringRef SegName)
        : LoadCommand<MachOTraits::SegmentCmd>(), Builder(Builder) {
      assert(SegName.size() <= 16 && "SegName too long");
      memcpy(this->segname, SegName.data(), SegName.size());
      this->maxprot =
          MachO::VM_PROT_READ | MachO::VM_PROT_WRITE | MachO::VM_PROT_EXECUTE;
      this->initprot = this->maxprot;
    }

    Section &addSection(StringRef SecName, StringRef SegName) {
      Sections.push_back(std::make_unique<Section>(Builder, SecName, SegName));
      return *Sections.back();
    }

    void write(BufferWriter &BW) override {
      writeMachOStruct(BW, this->rawStruct());
      for (auto &Sec : Sections)
        writeMachOStruct(BW, Sec->rawStruct());
    }
  };

  MachOBuilder(size_t PageSize) : PageSize(PageSize) {
    memset((char *)&Header, 0, sizeof(Header));
    Header.magic = MachOTraits::Magic;
  }

  template <MachO::LoadCommandType LCType, typename... ArgTs>
  LoadCommand<LCType> &addLoadCommand(ArgTs &&...Args) {
    static_assert(LCType != MachOTraits::SegmentCmd,
                  "Use addSegment to add segment load command");
    auto LC =
        std::make_unique<LoadCommand<LCType>>(std::forward<ArgTs>(Args)...);
    auto &Tmp = *LC;
    LoadCommands.push_back(std::move(LC));
    return Tmp;
  }

  StringId addString(StringRef Str) {
    if (Strings.empty() && !Str.empty())
      addString("");
    return Strings.insert(std::make_pair(Str, Strings.size())).first->second;
  }

  Segment &addSegment(StringRef SegName) {
    Segments.push_back(Segment(*this, SegName));
    return Segments.back();
  }

  RelocTarget addSymbol(StringRef Name, uint8_t Type, uint8_t Sect,
                        uint16_t Desc, typename MachOTraits::UIntPtr Value) {
    StringId SI = addString(Name);
    typename MachOTraits::NList Sym;
    Sym.n_strx = SI;
    Sym.n_type = Type;
    Sym.n_sect = Sect;
    Sym.n_desc = Desc;
    Sym.n_value = Value;
    SC.Symbols.push_back(Sym);
    return {SC, SC.Symbols.size() - 1};
  }

  // Call to perform layout on the MachO. Returns the total size of the
  // resulting file.
  // This method will automatically insert some load commands (e.g.
  // LC_SYMTAB) and fill in load command fields.
  size_t layout() {

    // Build symbol table and add LC_SYMTAB command.
    makeStringTable();
    LoadCommand<MachOTraits::SymTabCmd> *SymTabLC = nullptr;
    if (!StrTab.empty())
      SymTabLC = &addLoadCommand<MachOTraits::SymTabCmd>();

    // Lay out header, segment load command, and other load commands.
    size_t Offset = sizeof(Header);
    for (auto &Seg : Segments) {
      Seg.cmdsize +=
          Seg.Sections.size() * sizeof(typename MachOTraits::Section);
      Seg.nsects = Seg.Sections.size();
      Offset += Seg.cmdsize;
    }
    for (auto &LC : LoadCommands)
      Offset += LC->size();

    Header.sizeofcmds = Offset - sizeof(Header);

    // Lay out content, set segment / section addrs and offsets.
    size_t SegVMAddr = 0;
    for (auto &Seg : Segments) {
      Seg.vmaddr = SegVMAddr;
      Seg.fileoff = Offset;
      for (auto &Sec : Seg.Sections) {
        Offset = alignTo(Offset, 1 << Sec->align);
        if (Sec->Content.Size)
          Sec->offset = Offset;
        Sec->size = Sec->Content.Size;
        Sec->addr = SegVMAddr + Sec->offset - Seg.fileoff;
        Offset += Sec->Content.Size;
      }
      size_t SegContentSize = Offset - Seg.fileoff;
      Seg.filesize = SegContentSize;
      Seg.vmsize = Header.filetype == MachO::MH_OBJECT
                       ? SegContentSize
                       : alignTo(SegContentSize, PageSize);
      SegVMAddr += Seg.vmsize;
    }

    // Set string table offsets for non-section symbols.
    for (auto &Sym : SC.Symbols)
      Sym.n_strx = StrTab[Sym.n_strx].Offset;

    // Number sections, set symbol section numbers and string table offsets,
    // count relocations.
    size_t NumSymbols = SC.Symbols.size();
    size_t SectionNumber = 0;
    for (auto &Seg : Segments) {
      for (auto &Sec : Seg.Sections) {
        ++SectionNumber;
        Sec->SectionNumber = SectionNumber;
        Sec->SC.SymbolIndexBase = NumSymbols;
        NumSymbols += Sec->SC.Symbols.size();
        for (auto &Sym : Sec->SC.Symbols) {
          Sym.n_sect = SectionNumber;
          Sym.n_strx = StrTab[Sym.n_strx].Offset;
          Sym.n_value += Sec->addr;
        }
      }
    }

    // Handle relocations
    bool OffsetAlignedForRelocs = false;
    for (auto &Seg : Segments) {
      for (auto &Sec : Seg.Sections) {
        if (!Sec->Relocs.empty()) {
          if (!OffsetAlignedForRelocs) {
            Offset = alignTo(Offset, sizeof(MachO::relocation_info));
            OffsetAlignedForRelocs = true;
          }
          Sec->reloff = Offset;
          Sec->nreloc = Sec->Relocs.size();
          Offset += Sec->Relocs.size() * sizeof(MachO::relocation_info);
          for (auto &R : Sec->Relocs)
            R.r_symbolnum = R.Target.isSymbol() ? R.Target.getSymbolNum()
                                                : R.Target.getSectionId();
        }
      }
    }

    // Calculate offset to start of nlist and update symtab command.
    if (NumSymbols > 0) {
      Offset = alignTo(Offset, sizeof(typename MachOTraits::NList));
      SymTabLC->symoff = Offset;
      SymTabLC->nsyms = NumSymbols;

      // Calculate string table bounds and update symtab command.
      if (!StrTab.empty()) {
        Offset += NumSymbols * sizeof(typename MachOTraits::NList);
        size_t StringTableSize =
            StrTab.back().Offset + StrTab.back().S.size() + 1;

        SymTabLC->stroff = Offset;
        SymTabLC->strsize = StringTableSize;
        Offset += StringTableSize;
      }
    }

    return Offset;
  }

  void write(MutableArrayRef<char> Buffer) {
    BufferWriter BW(Buffer);
    writeHeader(BW);
    writeSegments(BW);
    writeLoadCommands(BW);
    writeSectionContent(BW);
    writeRelocations(BW);
    writeSymbols(BW);
    writeStrings(BW);
  }

  typename MachOTraits::Header Header;

private:
  void makeStringTable() {
    if (Strings.empty())
      return;

    StrTab.resize(Strings.size());
    for (auto &KV : Strings)
      StrTab[KV.second] = {KV.first, 0};
    size_t Offset = 0;
    for (auto &Elem : StrTab) {
      Elem.Offset = Offset;
      Offset += Elem.S.size() + 1;
    }
  }

  void writeHeader(BufferWriter &BW) {
    Header.ncmds = Segments.size() + LoadCommands.size();
    writeMachOStruct(BW, Header);
  }

  void writeSegments(BufferWriter &BW) {
    for (auto &Seg : Segments)
      Seg.write(BW);
  }

  void writeLoadCommands(BufferWriter &BW) {
    for (auto &LC : LoadCommands)
      LC->write(BW);
  }

  void writeSectionContent(BufferWriter &BW) {
    for (auto &Seg : Segments) {
      for (auto &Sec : Seg.Sections) {
        if (!Sec->Content.Data) {
          assert(Sec->Relocs.empty() &&
                 "Cant' have relocs for zero-fill segment");
          continue;
        }
        size_t ZeroPad = Sec->offset - BW.tell();
        while (ZeroPad--)
          BW.write('\0');
        BW.write(Sec->Content.Data, Sec->Content.Size);
      }
    }
  }

  void writeRelocations(BufferWriter &BW) {
    for (auto &Seg : Segments) {
      for (auto &Sec : Seg.Sections) {
        if (!Sec->Relocs.empty()) {
          while (BW.tell() % sizeof(MachO::relocation_info))
            BW.write('\0');
        }
        for (auto &R : Sec->Relocs)
          writeMachOStruct(BW, R.rawStruct());
      }
    }
  }

  void writeSymbols(BufferWriter &BW) {

    // Count symbols.
    size_t NumSymbols = SC.Symbols.size();
    for (auto &Seg : Segments)
      for (auto &Sec : Seg.Sections)
        NumSymbols += Sec->SC.Symbols.size();

    // If none then return.
    if (NumSymbols == 0)
      return;

    size_t ZeroPad =
        alignTo(BW.tell(), sizeof(typename MachOTraits::NList)) - BW.tell();
    while (ZeroPad--)
      BW.write('\0');

    // Write non-section symbols.
    for (auto &Sym : SC.Symbols)
      writeMachOStruct(BW, Sym);

    // Write section symbols.
    for (auto &Seg : Segments) {
      for (auto &Sec : Seg.Sections) {
        for (auto &Sym : Sec->SC.Symbols) {
          writeMachOStruct(BW, Sym);
        }
      }
    }
  }

  void writeStrings(BufferWriter &BW) {
    for (auto &Elem : StrTab) {
      BW.write(Elem.S.data(), Elem.S.size());
      BW.write('\0');
    }
  }

  size_t PageSize;
  std::list<Segment> Segments;
  std::vector<std::unique_ptr<LoadCommandBase>> LoadCommands;
  SymbolContainer SC;

  // Maps strings to their "id" (addition order).
  std::map<StringRef, size_t> Strings;
  StringTable StrTab;
};

struct MachO64LE {
  using UIntPtr = uint64_t;
  using Header = MachO::mach_header_64;
  using Section = MachO::section_64;
  using NList = MachO::nlist_64;
  using Relocation = MachO::relocation_info;

  static constexpr support::endianness Endianness = support::little;
  static constexpr uint32_t Magic = MachO::MH_MAGIC_64;
  static constexpr MachO::LoadCommandType SegmentCmd = MachO::LC_SEGMENT_64;
  static constexpr MachO::LoadCommandType SymTabCmd = MachO::LC_SYMTAB;
};

} // namespace orc
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_MACHOBUILDER_H
