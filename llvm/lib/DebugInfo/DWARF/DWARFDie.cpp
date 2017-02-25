//===-- DWARFDie.cpp ------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "SyntaxHighlighting.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugAbbrev.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugInfoEntry.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace dwarf;
using namespace syntax;

namespace {
 static void dumpApplePropertyAttribute(raw_ostream &OS, uint64_t Val) {
  OS << " (";
  do {
    uint64_t Shift = countTrailingZeros(Val);
    assert(Shift < 64 && "undefined behavior");
    uint64_t Bit = 1ULL << Shift;
    auto PropName = ApplePropertyString(Bit);
    if (!PropName.empty())
      OS << PropName;
    else
      OS << format("DW_APPLE_PROPERTY_0x%" PRIx64, Bit);
    if (!(Val ^= Bit))
      break;
    OS << ", ";
  } while (true);
  OS << ")";
}

static void dumpRanges(raw_ostream &OS, const DWARFAddressRangesVector& Ranges,
                       unsigned AddressSize, unsigned Indent) {
  if (Ranges.empty())
    return;
  
  for (const auto &Range: Ranges) {
    OS << '\n';
    OS.indent(Indent);
    OS << format("[0x%0*" PRIx64 " - 0x%0*" PRIx64 ")",
                 AddressSize*2, Range.first,
                 AddressSize*2, Range.second);
  }
}

static void dumpAttribute(raw_ostream &OS, const DWARFDie &Die,
                          uint32_t *OffsetPtr, dwarf::Attribute Attr,
                          dwarf::Form Form, unsigned Indent) {
  if (!Die.isValid())
    return;
  const char BaseIndent[] = "            ";
  OS << BaseIndent;
  OS.indent(Indent+2);
  auto attrString = AttributeString(Attr);
  if (!attrString.empty())
    WithColor(OS, syntax::Attribute) << attrString;
  else
    WithColor(OS, syntax::Attribute).get() << format("DW_AT_Unknown_%x", Attr);
  
  auto formString = FormEncodingString(Form);
  if (!formString.empty())
    OS << " [" << formString << ']';
  else
    OS << format(" [DW_FORM_Unknown_%x]", Form);
  
  DWARFUnit *U = Die.getDwarfUnit();
  DWARFFormValue formValue(Form);
  
  if (!formValue.extractValue(U->getDebugInfoExtractor(), OffsetPtr, U))
    return;
  
  OS << "\t(";
  
  StringRef Name;
  std::string File;
  auto Color = syntax::Enumerator;
  if (Attr == DW_AT_decl_file || Attr == DW_AT_call_file) {
    Color = syntax::String;
    if (const auto *LT = U->getContext().getLineTableForUnit(U))
      if (LT->getFileNameByIndex(formValue.getAsUnsignedConstant().getValue(), U->getCompilationDir(), DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath, File)) {
        File = '"' + File + '"';
        Name = File;
      }
  } else if (Optional<uint64_t> Val = formValue.getAsUnsignedConstant())
    Name = AttributeValueString(Attr, *Val);
  
  if (!Name.empty())
    WithColor(OS, Color) << Name;
  else if (Attr == DW_AT_decl_line || Attr == DW_AT_call_line)
    OS << *formValue.getAsUnsignedConstant();
  else
    formValue.dump(OS);
  
  // We have dumped the attribute raw value. For some attributes
  // having both the raw value and the pretty-printed value is
  // interesting. These attributes are handled below.
  if (Attr == DW_AT_specification || Attr == DW_AT_abstract_origin) {
    if (const char *Name = Die.getAttributeValueAsReferencedDie(Attr).getName(DINameKind::LinkageName))
        OS << " \"" << Name << '\"';
  } else if (Attr == DW_AT_APPLE_property_attribute) {
    if (Optional<uint64_t> OptVal = formValue.getAsUnsignedConstant())
      dumpApplePropertyAttribute(OS, *OptVal);
  } else if (Attr == DW_AT_ranges) {
    dumpRanges(OS, Die.getAddressRanges(), U->getAddressByteSize(),
               sizeof(BaseIndent)+Indent+4);
  }
  
  OS << ")\n";
}

} // end anonymous namespace

bool DWARFDie::isSubprogramDIE() const {
  return getTag() == DW_TAG_subprogram;
}

bool DWARFDie::isSubroutineDIE() const {
  auto Tag = getTag();
  return Tag == DW_TAG_subprogram || Tag == DW_TAG_inlined_subroutine;
}

Optional<DWARFFormValue>
DWARFDie::find(dwarf::Attribute Attr) const {
  if (!isValid())
    return None;
  auto AbbrevDecl = getAbbreviationDeclarationPtr();
  if (AbbrevDecl)
    return AbbrevDecl->getAttributeValue(getOffset(), Attr, *U);
  return None;
}

Optional<DWARFFormValue>
DWARFDie::findRecursively(dwarf::Attribute Attr) const {
  if (!isValid())
    return None;
  if (auto Value = find(Attr))
    return Value;
  if (auto Die = getAttributeValueAsReferencedDie(DW_AT_abstract_origin))
    if (auto Value = Die.find(Attr))
      return Value;
  if (auto Die = getAttributeValueAsReferencedDie(DW_AT_specification))
    if (auto Value = Die.find(Attr))
      return Value;
  return None;
}

Optional<DWARFFormValue>
DWARFDie::find(ArrayRef<dwarf::Attribute> Attrs) const {
  if (!isValid())
    return None;
  auto AbbrevDecl = getAbbreviationDeclarationPtr();
  if (AbbrevDecl) {
    for (auto Attr : Attrs) {
      if (auto Value = AbbrevDecl->getAttributeValue(getOffset(), Attr, *U))
        return Value;
    }
  }
  return None;
}

Optional<DWARFFormValue>
DWARFDie::findRecursively(ArrayRef<dwarf::Attribute> Attrs) const {
  if (!isValid())
    return None;
  if (auto Value = find(Attrs))
    return Value;
  if (auto Die = getAttributeValueAsReferencedDie(DW_AT_abstract_origin))
    if (auto Value = Die.find(Attrs))
      return Value;
  if (auto Die = getAttributeValueAsReferencedDie(DW_AT_specification))
    if (auto Value = Die.find(Attrs))
      return Value;
  return None;
}

DWARFDie
DWARFDie::getAttributeValueAsReferencedDie(dwarf::Attribute Attr) const {
  auto SpecRef = toReference(find(Attr));
  if (SpecRef) {
    auto SpecUnit = U->getUnitSection().getUnitForOffset(*SpecRef);
    if (SpecUnit)
      return SpecUnit->getDIEForOffset(*SpecRef);
  }
  return DWARFDie();
}

Optional<uint64_t>
DWARFDie::getRangesBaseAttribute() const {
  return toSectionOffset(find({DW_AT_rnglists_base, DW_AT_GNU_ranges_base}));
}

Optional<uint64_t> DWARFDie::getHighPC(uint64_t LowPC) const {
  if (auto FormValue = find(DW_AT_high_pc)) {
    if (auto Address = FormValue->getAsAddress()) {
      // High PC is an address.
      return Address;
    }
    if (auto Offset = FormValue->getAsUnsignedConstant()) {
      // High PC is an offset from LowPC.
      return LowPC + *Offset;
    }
  }
  return None;
}

bool DWARFDie::getLowAndHighPC(uint64_t &LowPC, uint64_t &HighPC) const {
  auto LowPcAddr = toAddress(find(DW_AT_low_pc));
  if (!LowPcAddr)
    return false;
  if (auto HighPcAddr = getHighPC(*LowPcAddr)) {
    LowPC = *LowPcAddr;
    HighPC = *HighPcAddr;
    return true;
  }
  return false;
}

DWARFAddressRangesVector
DWARFDie::getAddressRanges() const {
  if (isNULL())
    return DWARFAddressRangesVector();
  // Single range specified by low/high PC.
  uint64_t LowPC, HighPC;
  if (getLowAndHighPC(LowPC, HighPC)) {
    return DWARFAddressRangesVector(1, std::make_pair(LowPC, HighPC));
  }
  // Multiple ranges from .debug_ranges section.
  auto RangesOffset = toSectionOffset(find(DW_AT_ranges));
  if (RangesOffset) {
    DWARFDebugRangeList RangeList;
    if (U->extractRangeList(*RangesOffset, RangeList))
      return RangeList.getAbsoluteRanges(U->getBaseAddress());
  }
  return DWARFAddressRangesVector();
}

void
DWARFDie::collectChildrenAddressRanges(DWARFAddressRangesVector& Ranges) const {
  if (isNULL())
    return;
  if (isSubprogramDIE()) {
    const auto &DIERanges = getAddressRanges();
    Ranges.insert(Ranges.end(), DIERanges.begin(), DIERanges.end());
  }

  for (auto Child: children())
    Child.collectChildrenAddressRanges(Ranges);
}

bool DWARFDie::addressRangeContainsAddress(const uint64_t Address) const {
  for (const auto& R : getAddressRanges()) {
    if (R.first <= Address && Address < R.second)
      return true;
  }
  return false;
}

const char *
DWARFDie::getSubroutineName(DINameKind Kind) const {
  if (!isSubroutineDIE())
    return nullptr;
  return getName(Kind);
}

const char *
DWARFDie::getName(DINameKind Kind) const {
  if (!isValid() || Kind == DINameKind::None)
    return nullptr;
  // Try to get mangled name only if it was asked for.
  if (Kind == DINameKind::LinkageName) {
    if (auto Name = dwarf::toString(findRecursively({DW_AT_MIPS_linkage_name,
                                    DW_AT_linkage_name}), nullptr))
      return Name;
  }
  if (auto Name = dwarf::toString(findRecursively(DW_AT_name), nullptr))
    return Name;
  return nullptr;
}

uint64_t DWARFDie::getDeclLine() const {
  return toUnsigned(findRecursively(DW_AT_decl_line), 0);
}

void DWARFDie::getCallerFrame(uint32_t &CallFile, uint32_t &CallLine,
                              uint32_t &CallColumn) const {
  CallFile = toUnsigned(find(DW_AT_call_file), 0);
  CallLine = toUnsigned(find(DW_AT_call_line), 0);
  CallColumn = toUnsigned(find(DW_AT_call_column), 0);
}

void DWARFDie::dump(raw_ostream &OS, unsigned RecurseDepth,
                    unsigned Indent) const {
  if (!isValid())
    return;
  DataExtractor debug_info_data = U->getDebugInfoExtractor();
  const uint32_t Offset = getOffset();
  uint32_t offset = Offset;
  
  if (debug_info_data.isValidOffset(offset)) {
    uint32_t abbrCode = debug_info_data.getULEB128(&offset);
    WithColor(OS, syntax::Address).get() << format("\n0x%8.8x: ", Offset);
    
    if (abbrCode) {
      auto AbbrevDecl = getAbbreviationDeclarationPtr();
      if (AbbrevDecl) {
        auto tagString = TagString(getTag());
        if (!tagString.empty())
          WithColor(OS, syntax::Tag).get().indent(Indent) << tagString;
        else
          WithColor(OS, syntax::Tag).get().indent(Indent)
          << format("DW_TAG_Unknown_%x", getTag());
        
        OS << format(" [%u] %c\n", abbrCode,
                     AbbrevDecl->hasChildren() ? '*' : ' ');
        
        // Dump all data in the DIE for the attributes.
        for (const auto &AttrSpec : AbbrevDecl->attributes()) {
          if (AttrSpec.Form == DW_FORM_implicit_const) {
            // We are dumping .debug_info section ,
            // implicit_const attribute values are not really stored here,
            // but in .debug_abbrev section. So we just skip such attrs.
            continue;
          }
          dumpAttribute(OS, *this, &offset, AttrSpec.Attr, AttrSpec.Form,
                        Indent);
        }
        
        DWARFDie child = getFirstChild();
        if (RecurseDepth > 0 && child) {
          while (child) {
            child.dump(OS, RecurseDepth-1, Indent+2);
            child = child.getSibling();
          }
        }
      } else {
        OS << "Abbreviation code not found in 'debug_abbrev' class for code: "
        << abbrCode << '\n';
      }
    } else {
      OS.indent(Indent) << "NULL\n";
    }
  }
}


void DWARFDie::getInlinedChainForAddress(
    const uint64_t Address, SmallVectorImpl<DWARFDie> &InlinedChain) const {
  if (isNULL())
    return;
  DWARFDie DIE(*this);
  while (DIE) {
    // Append current DIE to inlined chain only if it has correct tag
    // (e.g. it is not a lexical block).
    if (DIE.isSubroutineDIE())
      InlinedChain.push_back(DIE);

    // Try to get child which also contains provided address.
    DWARFDie Child = DIE.getFirstChild();
    while (Child) {
      if (Child.addressRangeContainsAddress(Address)) {
        // Assume there is only one such child.
        break;
      }
      Child = Child.getSibling();
    }
    DIE = Child;
  }
  // Reverse the obtained chain to make the root of inlined chain last.
  std::reverse(InlinedChain.begin(), InlinedChain.end());
}

DWARFDie DWARFDie::getParent() const {
  if (isValid())
    return U->getParent(Die);
  return DWARFDie();
}

DWARFDie DWARFDie::getSibling() const {
  if (isValid())
    return U->getSibling(Die);
  return DWARFDie();
}

iterator_range<DWARFDie::attribute_iterator>
DWARFDie::attributes() const {
  return make_range(attribute_iterator(*this, false),
                    attribute_iterator(*this, true));
}

DWARFDie::attribute_iterator::attribute_iterator(DWARFDie D, bool End) :
    Die(D), AttrValue(0), Index(0) {
  auto AbbrDecl = Die.getAbbreviationDeclarationPtr();
  assert(AbbrDecl && "Must have abbreviation declaration");
  if (End) {
    // This is the end iterator so we set the index to the attribute count.
    Index = AbbrDecl->getNumAttributes();
  } else {
    // This is the begin iterator so we extract the value for this->Index.
    AttrValue.Offset = D.getOffset() + AbbrDecl->getCodeByteSize();
    updateForIndex(*AbbrDecl, 0);
  }
}

void DWARFDie::attribute_iterator::updateForIndex(
    const DWARFAbbreviationDeclaration &AbbrDecl, uint32_t I) {
  Index = I;
  // AbbrDecl must be valid befor calling this function.
  auto NumAttrs = AbbrDecl.getNumAttributes();
  if (Index < NumAttrs) {
    AttrValue.Attr = AbbrDecl.getAttrByIndex(Index);
    // Add the previous byte size of any previous attribute value.
    AttrValue.Offset += AttrValue.ByteSize;
    AttrValue.Value.setForm(AbbrDecl.getFormByIndex(Index));
    uint32_t ParseOffset = AttrValue.Offset;
    auto U = Die.getDwarfUnit();
    assert(U && "Die must have valid DWARF unit");
    bool b = AttrValue.Value.extractValue(U->getDebugInfoExtractor(),
                                          &ParseOffset, U);
    (void)b;
    assert(b && "extractValue cannot fail on fully parsed DWARF");
    AttrValue.ByteSize = ParseOffset - AttrValue.Offset;
  } else {
    assert(Index == NumAttrs && "Indexes should be [0, NumAttrs) only");
    AttrValue.clear();
  }
}

DWARFDie::attribute_iterator &DWARFDie::attribute_iterator::operator++() {
  if (auto AbbrDecl = Die.getAbbreviationDeclarationPtr())
    updateForIndex(*AbbrDecl, Index + 1);
  return *this;
}
