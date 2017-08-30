//===-- ResourceScriptStmt.h ------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// This lists all the resource and statement types occurring in RC scripts.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMRC_RESOURCESCRIPTSTMT_H
#define LLVM_TOOLS_LLVMRC_RESOURCESCRIPTSTMT_H

#include "ResourceScriptToken.h"

#include "llvm/ADT/StringSet.h"

namespace llvm {
namespace rc {

// A class holding a name - either an integer or a reference to the string.
class IntOrString {
private:
  union Data {
    uint32_t Int;
    StringRef String;
    Data(uint32_t Value) : Int(Value) {}
    Data(const StringRef Value) : String(Value) {}
    Data(const RCToken &Token) {
      if (Token.kind() == RCToken::Kind::Int)
        Int = Token.intValue();
      else
        String = Token.value();
    }
  } Data;
  bool IsInt;

public:
  IntOrString() : IntOrString(0) {}
  IntOrString(uint32_t Value) : Data(Value), IsInt(1) {}
  IntOrString(StringRef Value) : Data(Value), IsInt(0) {}
  IntOrString(const RCToken &Token)
      : Data(Token), IsInt(Token.kind() == RCToken::Kind::Int) {}

  bool equalsLower(const char *Str) {
    return !IsInt && Data.String.equals_lower(Str);
  }

  friend raw_ostream &operator<<(raw_ostream &, const IntOrString &);
};

// Base resource. All the resources should derive from this base.
class RCResource {
protected:
  IntOrString ResName;

public:
  RCResource() = default;
  RCResource(RCResource &&) = default;
  void setName(const IntOrString &Name) { ResName = Name; }
  virtual raw_ostream &log(raw_ostream &OS) const {
    return OS << "Base statement\n";
  };
  virtual ~RCResource() {}
};

// Optional statement base. All such statements should derive from this base.
class OptionalStmt : public RCResource {};

class OptionalStmtList : public OptionalStmt {
  std::vector<std::unique_ptr<OptionalStmt>> Statements;

public:
  OptionalStmtList() {}
  virtual raw_ostream &log(raw_ostream &OS) const;

  void addStmt(std::unique_ptr<OptionalStmt> Stmt) {
    Statements.push_back(std::move(Stmt));
  }
};

// LANGUAGE statement. It can occur both as a top-level statement (in such
// a situation, it changes the default language until the end of the file)
// and as an optional resource statement (then it changes the language
// of a single resource).
//
// Ref: msdn.microsoft.com/en-us/library/windows/desktop/aa381019(v=vs.85).aspx
class LanguageResource : public OptionalStmt {
  uint32_t Lang, SubLang;

public:
  LanguageResource(uint32_t LangId, uint32_t SubLangId)
      : Lang(LangId), SubLang(SubLangId) {}
  raw_ostream &log(raw_ostream &) const override;
};

// ACCELERATORS resource. Defines a named table of accelerators for the app.
//
// Ref: msdn.microsoft.com/en-us/library/windows/desktop/aa380610(v=vs.85).aspx
class AcceleratorsResource : public RCResource {
public:
  class Accelerator {
  public:
    IntOrString Event;
    uint32_t Id;
    uint8_t Flags;

    enum Options {
      ASCII = (1 << 0),
      VIRTKEY = (1 << 1),
      NOINVERT = (1 << 2),
      ALT = (1 << 3),
      SHIFT = (1 << 4),
      CONTROL = (1 << 5)
    };

    static constexpr size_t NumFlags = 6;
    static StringRef OptionsStr[NumFlags];
  };

  AcceleratorsResource(OptionalStmtList &&OptStmts)
      : OptStatements(std::move(OptStmts)) {}
  void addAccelerator(IntOrString Event, uint32_t Id, uint8_t Flags) {
    Accelerators.push_back(Accelerator{Event, Id, Flags});
  }
  raw_ostream &log(raw_ostream &) const override;

private:
  std::vector<Accelerator> Accelerators;
  OptionalStmtList OptStatements;
};

// CURSOR resource. Represents a single cursor (".cur") file.
//
// Ref: msdn.microsoft.com/en-us/library/windows/desktop/aa380920(v=vs.85).aspx
class CursorResource : public RCResource {
  StringRef CursorLoc;

public:
  CursorResource(StringRef Location) : CursorLoc(Location) {}
  raw_ostream &log(raw_ostream &) const override;
};

// ICON resource. Represents a single ".ico" file containing a group of icons.
//
// Ref: msdn.microsoft.com/en-us/library/windows/desktop/aa381018(v=vs.85).aspx
class IconResource : public RCResource {
  StringRef IconLoc;

public:
  IconResource(StringRef Location) : IconLoc(Location) {}
  raw_ostream &log(raw_ostream &) const override;
};

// HTML resource. Represents a local webpage that is to be embedded into the
// resulting resource file. It embeds a file only - no additional resources
// (images etc.) are included with this resource.
//
// Ref: msdn.microsoft.com/en-us/library/windows/desktop/aa966018(v=vs.85).aspx
class HTMLResource : public RCResource {
  StringRef HTMLLoc;

public:
  HTMLResource(StringRef Location) : HTMLLoc(Location) {}
  raw_ostream &log(raw_ostream &) const override;
};

// -- MENU resource and its helper classes --
// This resource describes the contents of an application menu
// (usually located in the upper part of the dialog.)
//
// Ref: msdn.microsoft.com/en-us/library/windows/desktop/aa381025(v=vs.85).aspx

// Description of a single submenu item.
class MenuDefinition {
public:
  enum Options {
    CHECKED = (1 << 0),
    GRAYED = (1 << 1),
    HELP = (1 << 2),
    INACTIVE = (1 << 3),
    MENUBARBREAK = (1 << 4),
    MENUBREAK = (1 << 5)
  };

  static constexpr size_t NumFlags = 6;
  static StringRef OptionsStr[NumFlags];
  static raw_ostream &logFlags(raw_ostream &, uint8_t Flags);
  virtual raw_ostream &log(raw_ostream &OS) const {
    return OS << "Base menu definition\n";
  }
  virtual ~MenuDefinition() {}
};

// Recursive description of a whole submenu.
class MenuDefinitionList : public MenuDefinition {
  std::vector<std::unique_ptr<MenuDefinition>> Definitions;

public:
  void addDefinition(std::unique_ptr<MenuDefinition> Def) {
    Definitions.push_back(std::move(Def));
  }
  raw_ostream &log(raw_ostream &) const override;
};

// Separator in MENU definition (MENUITEM SEPARATOR).
//
// Ref: msdn.microsoft.com/en-us/library/windows/desktop/aa381024(v=vs.85).aspx
class MenuSeparator : public MenuDefinition {
public:
  raw_ostream &log(raw_ostream &) const override;
};

// MENUITEM statement definition.
//
// Ref: msdn.microsoft.com/en-us/library/windows/desktop/aa381024(v=vs.85).aspx
class MenuItem : public MenuDefinition {
  StringRef Name;
  uint32_t Id;
  uint8_t Flags;

public:
  MenuItem(StringRef Caption, uint32_t ItemId, uint8_t ItemFlags)
      : Name(Caption), Id(ItemId), Flags(ItemFlags) {}
  raw_ostream &log(raw_ostream &) const override;
};

// POPUP statement definition.
//
// Ref: msdn.microsoft.com/en-us/library/windows/desktop/aa381030(v=vs.85).aspx
class PopupItem : public MenuDefinition {
  StringRef Name;
  uint8_t Flags;
  MenuDefinitionList SubItems;

public:
  PopupItem(StringRef Caption, uint8_t ItemFlags,
            MenuDefinitionList &&SubItemsList)
      : Name(Caption), Flags(ItemFlags), SubItems(std::move(SubItemsList)) {}
  raw_ostream &log(raw_ostream &) const override;
};

// Menu resource definition.
class MenuResource : public RCResource {
  OptionalStmtList OptStatements;
  MenuDefinitionList Elements;

public:
  MenuResource(OptionalStmtList &&OptStmts, MenuDefinitionList &&Items)
      : OptStatements(std::move(OptStmts)), Elements(std::move(Items)) {}
  raw_ostream &log(raw_ostream &) const override;
};

// STRINGTABLE resource. Contains a list of strings, each having its unique ID.
//
// Ref: msdn.microsoft.com/en-us/library/windows/desktop/aa381050(v=vs.85).aspx
class StringTableResource : public RCResource {
  OptionalStmtList OptStatements;
  std::vector<std::pair<uint32_t, StringRef>> Table;

public:
  StringTableResource(OptionalStmtList &&OptStmts)
      : OptStatements(std::move(OptStmts)) {}
  void addString(uint32_t ID, StringRef String) {
    Table.emplace_back(ID, String);
  }
  raw_ostream &log(raw_ostream &) const override;
};

// -- DIALOG(EX) resource and its helper classes --
//
// This resource describes dialog boxes and controls residing inside them.
//
// Ref: msdn.microsoft.com/en-us/library/windows/desktop/aa381003(v=vs.85).aspx
// Ref: msdn.microsoft.com/en-us/library/windows/desktop/aa381002(v=vs.85).aspx

// Single control definition.
class Control {
  StringRef Type, Title;
  uint32_t ID, X, Y, Width, Height;
  Optional<uint32_t> Style, ExtStyle, HelpID;

public:
  Control(StringRef CtlType, StringRef CtlTitle, uint32_t CtlID, uint32_t PosX,
          uint32_t PosY, uint32_t ItemWidth, uint32_t ItemHeight,
          Optional<uint32_t> ItemStyle, Optional<uint32_t> ExtItemStyle,
          Optional<uint32_t> CtlHelpID)
      : Type(CtlType), Title(CtlTitle), ID(CtlID), X(PosX), Y(PosY),
        Width(ItemWidth), Height(ItemHeight), Style(ItemStyle),
        ExtStyle(ExtItemStyle), HelpID(CtlHelpID) {}

  static const StringSet<> SupportedCtls;
  static const StringSet<> CtlsWithTitle;

  raw_ostream &log(raw_ostream &) const;
};

// Single dialog definition. We don't create distinct classes for DIALOG and
// DIALOGEX because of their being too similar to each other. We only have a
// flag determining the type of the dialog box.
class DialogResource : public RCResource {
  uint32_t X, Y, Width, Height, HelpID;
  OptionalStmtList OptStatements;
  std::vector<Control> Controls;
  bool IsExtended;

public:
  DialogResource(uint32_t PosX, uint32_t PosY, uint32_t DlgWidth,
                 uint32_t DlgHeight, uint32_t DlgHelpID,
                 OptionalStmtList &&OptStmts, bool IsDialogEx)
      : X(PosX), Y(PosY), Width(DlgWidth), Height(DlgHeight), HelpID(DlgHelpID),
        OptStatements(std::move(OptStmts)), IsExtended(IsDialogEx) {}

  void addControl(Control &&Ctl) { Controls.push_back(std::move(Ctl)); }

  raw_ostream &log(raw_ostream &) const override;
};

// CHARACTERISTICS optional statement.
//
// Ref: msdn.microsoft.com/en-us/library/windows/desktop/aa380872(v=vs.85).aspx
class CharacteristicsStmt : public OptionalStmt {
  uint32_t Value;

public:
  CharacteristicsStmt(uint32_t Characteristic) : Value(Characteristic) {}
  raw_ostream &log(raw_ostream &) const override;
};

// VERSION optional statement.
//
// Ref: msdn.microsoft.com/en-us/library/windows/desktop/aa381059(v=vs.85).aspx
class VersionStmt : public OptionalStmt {
  uint32_t Value;

public:
  VersionStmt(uint32_t Version) : Value(Version) {}
  raw_ostream &log(raw_ostream &) const override;
};

// CAPTION optional statement.
//
// Ref: msdn.microsoft.com/en-us/library/windows/desktop/aa380778(v=vs.85).aspx
class CaptionStmt : public OptionalStmt {
  StringRef Value;

public:
  CaptionStmt(StringRef Caption) : Value(Caption) {}
  raw_ostream &log(raw_ostream &) const override;
};

// FONT optional statement.
// Note that the documentation is inaccurate: it expects five arguments to be
// given, however the example provides only two. In fact, the original tool
// expects two arguments - point size and name of the typeface.
//
// Ref: msdn.microsoft.com/en-us/library/windows/desktop/aa381013(v=vs.85).aspx
class FontStmt : public OptionalStmt {
  uint32_t Size;
  StringRef Typeface;

public:
  FontStmt(uint32_t FontSize, StringRef FontName)
      : Size(FontSize), Typeface(FontName) {}
  raw_ostream &log(raw_ostream &) const override;
};

// STYLE optional statement.
//
// Ref: msdn.microsoft.com/en-us/library/windows/desktop/aa381051(v=vs.85).aspx
class StyleStmt : public OptionalStmt {
  uint32_t Value;

public:
  StyleStmt(uint32_t Style) : Value(Style) {}
  raw_ostream &log(raw_ostream &) const override;
};

} // namespace rc
} // namespace llvm

#endif
