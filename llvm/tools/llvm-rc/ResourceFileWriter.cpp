//===-- ResourceFileWriter.cpp --------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// This implements the visitor serializing resources to a .res stream.
//
//===---------------------------------------------------------------------===//

#include "ResourceFileWriter.h"

#include "llvm/Object/WindowsResource.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"

using namespace llvm::support;

// Take an expression returning llvm::Error and forward the error if it exists.
#define RETURN_IF_ERROR(Expr)                                                  \
  if (auto Err = (Expr))                                                       \
    return Err;

namespace llvm {
namespace rc {

// Class that employs RAII to save the current serializator object state
// and revert to it as soon as we leave the scope. This is useful if resources
// declare their own resource-local statements.
class ContextKeeper {
  ResourceFileWriter *FileWriter;
  ResourceFileWriter::ObjectInfo SavedInfo;

public:
  ContextKeeper(ResourceFileWriter *V)
      : FileWriter(V), SavedInfo(V->ObjectData) {}
  ~ContextKeeper() { FileWriter->ObjectData = SavedInfo; }
};

static Error createError(Twine Message,
                         std::errc Type = std::errc::invalid_argument) {
  return make_error<StringError>(Message, std::make_error_code(Type));
}

static Error checkNumberFits(uint32_t Number, size_t MaxBits, Twine FieldName) {
  assert(1 <= MaxBits && MaxBits <= 32);
  if (!(Number >> MaxBits))
    return Error::success();
  return createError(FieldName + " (" + Twine(Number) + ") does not fit in " +
                         Twine(MaxBits) + " bits.",
                     std::errc::value_too_large);
}

template <typename FitType>
static Error checkNumberFits(uint32_t Number, Twine FieldName) {
  return checkNumberFits(Number, sizeof(FitType) * 8, FieldName);
}

// A similar function for signed integers.
template <typename FitType>
static Error checkSignedNumberFits(uint32_t Number, Twine FieldName,
                                   bool CanBeNegative) {
  int32_t SignedNum = Number;
  if (SignedNum < std::numeric_limits<FitType>::min() ||
      SignedNum > std::numeric_limits<FitType>::max())
    return createError(FieldName + " (" + Twine(SignedNum) +
                           ") does not fit in " + Twine(sizeof(FitType) * 8) +
                           "-bit signed integer type.",
                       std::errc::value_too_large);

  if (!CanBeNegative && SignedNum < 0)
    return createError(FieldName + " (" + Twine(SignedNum) +
                       ") cannot be negative.");

  return Error::success();
}

static Error checkIntOrString(IntOrString Value, Twine FieldName) {
  if (!Value.isInt())
    return Error::success();
  return checkNumberFits<uint16_t>(Value.getInt(), FieldName);
}

static bool stripQuotes(StringRef &Str, bool &IsLongString) {
  if (!Str.contains('"'))
    return false;

  // Just take the contents of the string, checking if it's been marked long.
  IsLongString = Str.startswith_lower("L");
  if (IsLongString)
    Str = Str.drop_front();

  bool StripSuccess = Str.consume_front("\"") && Str.consume_back("\"");
  (void)StripSuccess;
  assert(StripSuccess && "Strings should be enclosed in quotes.");
  return true;
}

// Describes a way to handle '\0' characters when processing the string.
// rc.exe tool sometimes behaves in a weird way in postprocessing.
// If the string to be output is equivalent to a C-string (e.g. in MENU
// titles), string is (predictably) truncated after first 0-byte.
// When outputting a string table, the behavior is equivalent to appending
// '\0\0' at the end of the string, and then stripping the string
// before the first '\0\0' occurrence.
// Finally, when handling strings in user-defined resources, 0-bytes
// aren't stripped, nor do they terminate the string.

enum class NullHandlingMethod {
  UserResource,   // Don't terminate string on '\0'.
  CutAtNull,      // Terminate string on '\0'.
  CutAtDoubleNull // Terminate string on '\0\0'; strip final '\0'.
};

// Parses an identifier or string and returns a processed version of it.
// For now, it only strips the string boundaries, but TODO:
//   * Squash "" to a single ".
//   * Replace the escape sequences with their processed version.
// For identifiers, this is no-op.
static Error processString(StringRef Str, NullHandlingMethod NullHandler,
                           bool &IsLongString, SmallVectorImpl<UTF16> &Result) {
  assert(NullHandler == NullHandlingMethod::CutAtNull);

  bool IsString = stripQuotes(Str, IsLongString);
  convertUTF8ToUTF16String(Str, Result);

  if (!IsString) {
    // It's an identifier if it's not a string. Make all characters uppercase.
    for (UTF16 &Ch : Result) {
      assert(Ch <= 0x7F && "We didn't allow identifiers to be non-ASCII");
      Ch = toupper(Ch);
    }
    return Error::success();
  }

  // We don't process the string contents. Only cut at '\0'.

  for (size_t Pos = 0; Pos < Result.size(); ++Pos)
    if (Result[Pos] == '\0')
      Result.resize(Pos);

  return Error::success();
}

uint64_t ResourceFileWriter::writeObject(const ArrayRef<uint8_t> Data) {
  uint64_t Result = tell();
  FS->write((const char *)Data.begin(), Data.size());
  return Result;
}

Error ResourceFileWriter::writeCString(StringRef Str, bool WriteTerminator) {
  SmallVector<UTF16, 128> ProcessedString;
  bool IsLongString;
  RETURN_IF_ERROR(processString(Str, NullHandlingMethod::CutAtNull,
                                IsLongString, ProcessedString));
  for (auto Ch : ProcessedString)
    writeInt<uint16_t>(Ch);
  if (WriteTerminator)
    writeInt<uint16_t>(0);
  return Error::success();
}

Error ResourceFileWriter::writeIdentifier(const IntOrString &Ident) {
  return writeIntOrString(Ident);
}

Error ResourceFileWriter::writeIntOrString(const IntOrString &Value) {
  if (!Value.isInt())
    return writeCString(Value.getString());

  writeInt<uint16_t>(0xFFFF);
  writeInt<uint16_t>(Value.getInt());
  return Error::success();
}

Error ResourceFileWriter::appendFile(StringRef Filename) {
  bool IsLong;
  stripQuotes(Filename, IsLong);

  // Filename path should be relative to the current working directory.
  // FIXME: docs say so, but reality is more complicated, script
  // location and include paths must be taken into account.
  ErrorOr<std::unique_ptr<MemoryBuffer>> File =
      MemoryBuffer::getFile(Filename, -1, false);
  if (!File)
    return make_error<StringError>("Error opening file '" + Filename +
                                       "': " + File.getError().message(),
                                   File.getError());
  *FS << (*File)->getBuffer();
  return Error::success();
}

void ResourceFileWriter::padStream(uint64_t Length) {
  assert(Length > 0);
  uint64_t Location = tell();
  Location %= Length;
  uint64_t Pad = (Length - Location) % Length;
  for (uint64_t i = 0; i < Pad; ++i)
    writeInt<uint8_t>(0);
}

Error ResourceFileWriter::handleError(Error &&Err, const RCResource *Res) {
  if (Err)
    return joinErrors(createError("Error in " + Res->getResourceTypeName() +
                                  " statement (ID " + Twine(Res->ResName) +
                                  "): "),
                      std::move(Err));
  return Error::success();
}

Error ResourceFileWriter::visitNullResource(const RCResource *Res) {
  return writeResource(Res, &ResourceFileWriter::writeNullBody);
}

Error ResourceFileWriter::visitAcceleratorsResource(const RCResource *Res) {
  return writeResource(Res, &ResourceFileWriter::writeAcceleratorsBody);
}

Error ResourceFileWriter::visitDialogResource(const RCResource *Res) {
  return writeResource(Res, &ResourceFileWriter::writeDialogBody);
}

Error ResourceFileWriter::visitHTMLResource(const RCResource *Res) {
  return writeResource(Res, &ResourceFileWriter::writeHTMLBody);
}

Error ResourceFileWriter::visitMenuResource(const RCResource *Res) {
  return writeResource(Res, &ResourceFileWriter::writeMenuBody);
}

Error ResourceFileWriter::visitCharacteristicsStmt(
    const CharacteristicsStmt *Stmt) {
  ObjectData.Characteristics = Stmt->Value;
  return Error::success();
}

Error ResourceFileWriter::visitLanguageStmt(const LanguageResource *Stmt) {
  RETURN_IF_ERROR(checkNumberFits(Stmt->Lang, 10, "Primary language ID"));
  RETURN_IF_ERROR(checkNumberFits(Stmt->SubLang, 6, "Sublanguage ID"));
  ObjectData.LanguageInfo = Stmt->Lang | (Stmt->SubLang << 10);
  return Error::success();
}

Error ResourceFileWriter::visitVersionStmt(const VersionStmt *Stmt) {
  ObjectData.VersionInfo = Stmt->Value;
  return Error::success();
}

Error ResourceFileWriter::writeResource(
    const RCResource *Res,
    Error (ResourceFileWriter::*BodyWriter)(const RCResource *)) {
  // We don't know the sizes yet.
  object::WinResHeaderPrefix HeaderPrefix{ulittle32_t(0U), ulittle32_t(0U)};
  uint64_t HeaderLoc = writeObject(HeaderPrefix);

  auto ResType = Res->getResourceType();
  RETURN_IF_ERROR(checkIntOrString(ResType, "Resource type"));
  RETURN_IF_ERROR(checkIntOrString(Res->ResName, "Resource ID"));
  RETURN_IF_ERROR(handleError(writeIdentifier(ResType), Res));
  RETURN_IF_ERROR(handleError(writeIdentifier(Res->ResName), Res));

  // Apply the resource-local optional statements.
  ContextKeeper RAII(this);
  RETURN_IF_ERROR(handleError(Res->applyStmts(this), Res));

  padStream(sizeof(uint32_t));
  object::WinResHeaderSuffix HeaderSuffix{
      ulittle32_t(0), // DataVersion; seems to always be 0
      ulittle16_t(Res->getMemoryFlags()), ulittle16_t(ObjectData.LanguageInfo),
      ulittle32_t(ObjectData.VersionInfo),
      ulittle32_t(ObjectData.Characteristics)};
  writeObject(HeaderSuffix);

  uint64_t DataLoc = tell();
  RETURN_IF_ERROR(handleError((this->*BodyWriter)(Res), Res));
  // RETURN_IF_ERROR(handleError(dumpResource(Ctx)));

  // Update the sizes.
  HeaderPrefix.DataSize = tell() - DataLoc;
  HeaderPrefix.HeaderSize = DataLoc - HeaderLoc;
  writeObjectAt(HeaderPrefix, HeaderLoc);
  padStream(sizeof(uint32_t));

  return Error::success();
}

// --- NullResource helpers. --- //

Error ResourceFileWriter::writeNullBody(const RCResource *) {
  return Error::success();
}

// --- AcceleratorsResource helpers. --- //

Error ResourceFileWriter::writeSingleAccelerator(
    const AcceleratorsResource::Accelerator &Obj, bool IsLastItem) {
  using Accelerator = AcceleratorsResource::Accelerator;
  using Opt = Accelerator::Options;

  struct AccelTableEntry {
    ulittle16_t Flags;
    ulittle16_t ANSICode;
    ulittle16_t Id;
    uint16_t Padding;
  } Entry{ulittle16_t(0), ulittle16_t(0), ulittle16_t(0), 0};

  bool IsASCII = Obj.Flags & Opt::ASCII, IsVirtKey = Obj.Flags & Opt::VIRTKEY;

  // Remove ASCII flags (which doesn't occur in .res files).
  Entry.Flags = Obj.Flags & ~Opt::ASCII;

  if (IsLastItem)
    Entry.Flags |= 0x80;

  RETURN_IF_ERROR(checkNumberFits<uint16_t>(Obj.Id, "ACCELERATORS entry ID"));
  Entry.Id = ulittle16_t(Obj.Id);

  auto createAccError = [&Obj](const char *Msg) {
    return createError("Accelerator ID " + Twine(Obj.Id) + ": " + Msg);
  };

  if (IsASCII && IsVirtKey)
    return createAccError("Accelerator can't be both ASCII and VIRTKEY");

  if (!IsVirtKey && (Obj.Flags & (Opt::ALT | Opt::SHIFT | Opt::CONTROL)))
    return createAccError("Can only apply ALT, SHIFT or CONTROL to VIRTKEY"
                          " accelerators");

  if (Obj.Event.isInt()) {
    if (!IsASCII && !IsVirtKey)
      return createAccError(
          "Accelerator with a numeric event must be either ASCII"
          " or VIRTKEY");

    uint32_t EventVal = Obj.Event.getInt();
    RETURN_IF_ERROR(
        checkNumberFits<uint16_t>(EventVal, "Numeric event key ID"));
    Entry.ANSICode = ulittle16_t(EventVal);
    writeObject(Entry);
    return Error::success();
  }

  StringRef Str = Obj.Event.getString();
  bool IsWide;
  stripQuotes(Str, IsWide);

  if (Str.size() == 0 || Str.size() > 2)
    return createAccError(
        "Accelerator string events should have length 1 or 2");

  if (Str[0] == '^') {
    if (Str.size() == 1)
      return createAccError("No character following '^' in accelerator event");
    if (IsVirtKey)
      return createAccError(
          "VIRTKEY accelerator events can't be preceded by '^'");

    char Ch = Str[1];
    if (Ch >= 'a' && Ch <= 'z')
      Entry.ANSICode = ulittle16_t(Ch - 'a' + 1);
    else if (Ch >= 'A' && Ch <= 'Z')
      Entry.ANSICode = ulittle16_t(Ch - 'A' + 1);
    else
      return createAccError("Control character accelerator event should be"
                            " alphabetic");

    writeObject(Entry);
    return Error::success();
  }

  if (Str.size() == 2)
    return createAccError("Event string should be one-character, possibly"
                          " preceded by '^'");

  uint8_t EventCh = Str[0];
  // The original tool just warns in this situation. We chose to fail.
  if (IsVirtKey && !isalnum(EventCh))
    return createAccError("Non-alphanumeric characters cannot describe virtual"
                          " keys");
  if (EventCh > 0x7F)
    return createAccError("Non-ASCII description of accelerator");

  if (IsVirtKey)
    EventCh = toupper(EventCh);
  Entry.ANSICode = ulittle16_t(EventCh);
  writeObject(Entry);
  return Error::success();
}

Error ResourceFileWriter::writeAcceleratorsBody(const RCResource *Base) {
  auto *Res = cast<AcceleratorsResource>(Base);
  size_t AcceleratorId = 0;
  for (auto &Acc : Res->Accelerators) {
    ++AcceleratorId;
    RETURN_IF_ERROR(
        writeSingleAccelerator(Acc, AcceleratorId == Res->Accelerators.size()));
  }
  return Error::success();
}

// --- DialogResource helpers. --- //

Error ResourceFileWriter::writeSingleDialogControl(const Control &Ctl,
                                                   bool IsExtended) {
  // Each control should be aligned to DWORD.
  padStream(sizeof(uint32_t));

  auto TypeInfo = Control::SupportedCtls.lookup(Ctl.Type);
  uint32_t CtlStyle = TypeInfo.Style | Ctl.Style.getValueOr(0);
  uint32_t CtlExtStyle = Ctl.ExtStyle.getValueOr(0);

  // DIALOG(EX) item header prefix.
  if (!IsExtended) {
    struct {
      ulittle32_t Style;
      ulittle32_t ExtStyle;
    } Prefix{ulittle32_t(CtlStyle), ulittle32_t(CtlExtStyle)};
    writeObject(Prefix);
  } else {
    struct {
      ulittle32_t HelpID;
      ulittle32_t ExtStyle;
      ulittle32_t Style;
    } Prefix{ulittle32_t(Ctl.HelpID.getValueOr(0)), ulittle32_t(CtlExtStyle),
             ulittle32_t(CtlStyle)};
    writeObject(Prefix);
  }

  // Common fixed-length part.
  RETURN_IF_ERROR(checkSignedNumberFits<int16_t>(
      Ctl.X, "Dialog control x-coordinate", true));
  RETURN_IF_ERROR(checkSignedNumberFits<int16_t>(
      Ctl.Y, "Dialog control y-coordinate", true));
  RETURN_IF_ERROR(
      checkSignedNumberFits<int16_t>(Ctl.Width, "Dialog control width", false));
  RETURN_IF_ERROR(checkSignedNumberFits<int16_t>(
      Ctl.Height, "Dialog control height", false));
  struct {
    ulittle16_t X;
    ulittle16_t Y;
    ulittle16_t Width;
    ulittle16_t Height;
  } Middle{ulittle16_t(Ctl.X), ulittle16_t(Ctl.Y), ulittle16_t(Ctl.Width),
           ulittle16_t(Ctl.Height)};
  writeObject(Middle);

  // ID; it's 16-bit in DIALOG and 32-bit in DIALOGEX.
  if (!IsExtended) {
    RETURN_IF_ERROR(checkNumberFits<uint16_t>(
        Ctl.ID, "Control ID in simple DIALOG resource"));
    writeInt<uint16_t>(Ctl.ID);
  } else {
    writeInt<uint32_t>(Ctl.ID);
  }

  // Window class - either 0xFFFF + 16-bit integer or a string.
  RETURN_IF_ERROR(writeIntOrString(IntOrString(TypeInfo.CtlClass)));

  // Element caption/reference ID. ID is preceded by 0xFFFF.
  RETURN_IF_ERROR(checkIntOrString(Ctl.Title, "Control reference ID"));
  RETURN_IF_ERROR(writeIntOrString(Ctl.Title));

  // # bytes of extra creation data count. Don't pass any.
  writeInt<uint16_t>(0);

  return Error::success();
}

Error ResourceFileWriter::writeDialogBody(const RCResource *Base) {
  auto *Res = cast<DialogResource>(Base);

  // Default style: WS_POPUP | WS_BORDER | WS_SYSMENU.
  const uint32_t UsedStyle = 0x80880000;

  // Write DIALOG(EX) header prefix. These are pretty different.
  if (!Res->IsExtended) {
    struct {
      ulittle32_t Style;
      ulittle32_t ExtStyle;
    } Prefix{ulittle32_t(UsedStyle),
             ulittle32_t(0)}; // As of now, we don't keep EXSTYLE.

    writeObject(Prefix);
  } else {
    const uint16_t DialogExMagic = 0xFFFF;

    struct {
      ulittle16_t Version;
      ulittle16_t Magic;
      ulittle32_t HelpID;
      ulittle32_t ExtStyle;
      ulittle32_t Style;
    } Prefix{ulittle16_t(1), ulittle16_t(DialogExMagic),
             ulittle32_t(Res->HelpID), ulittle32_t(0), ulittle32_t(UsedStyle)};

    writeObject(Prefix);
  }

  // Now, a common part. First, fixed-length fields.
  RETURN_IF_ERROR(checkNumberFits<uint16_t>(Res->Controls.size(),
                                            "Number of dialog controls"));
  RETURN_IF_ERROR(
      checkSignedNumberFits<int16_t>(Res->X, "Dialog x-coordinate", true));
  RETURN_IF_ERROR(
      checkSignedNumberFits<int16_t>(Res->Y, "Dialog y-coordinate", true));
  RETURN_IF_ERROR(
      checkSignedNumberFits<int16_t>(Res->Width, "Dialog width", false));
  RETURN_IF_ERROR(
      checkSignedNumberFits<int16_t>(Res->Height, "Dialog height", false));
  struct {
    ulittle16_t Count;
    ulittle16_t PosX;
    ulittle16_t PosY;
    ulittle16_t DialogWidth;
    ulittle16_t DialogHeight;
  } Middle{ulittle16_t(Res->Controls.size()), ulittle16_t(Res->X),
           ulittle16_t(Res->Y), ulittle16_t(Res->Width),
           ulittle16_t(Res->Height)};
  writeObject(Middle);

  // MENU field. As of now, we don't keep them in the state and can peacefully
  // think there is no menu attached to the dialog.
  writeInt<uint16_t>(0);

  // Window CLASS field. Not kept here.
  writeInt<uint16_t>(0);

  // Window title. There is no title for now, so all we output is '\0'.
  writeInt<uint16_t>(0);

  auto handleCtlError = [&](Error &&Err, const Control &Ctl) -> Error {
    if (!Err)
      return Error::success();
    return joinErrors(createError("Error in " + Twine(Ctl.Type) +
                                  " control  (ID " + Twine(Ctl.ID) + "):"),
                      std::move(Err));
  };

  for (auto &Ctl : Res->Controls)
    RETURN_IF_ERROR(
        handleCtlError(writeSingleDialogControl(Ctl, Res->IsExtended), Ctl));

  return Error::success();
}

// --- HTMLResource helpers. --- //

Error ResourceFileWriter::writeHTMLBody(const RCResource *Base) {
  return appendFile(cast<HTMLResource>(Base)->HTMLLoc);
}

// --- MenuResource helpers. --- //

Error ResourceFileWriter::writeMenuDefinition(
    const std::unique_ptr<MenuDefinition> &Def, uint16_t Flags) {
  assert(Def);
  const MenuDefinition *DefPtr = Def.get();

  if (auto *MenuItemPtr = dyn_cast<MenuItem>(DefPtr)) {
    writeInt<uint16_t>(Flags);
    RETURN_IF_ERROR(
        checkNumberFits<uint16_t>(MenuItemPtr->Id, "MENUITEM action ID"));
    writeInt<uint16_t>(MenuItemPtr->Id);
    RETURN_IF_ERROR(writeCString(MenuItemPtr->Name));
    return Error::success();
  }

  if (isa<MenuSeparator>(DefPtr)) {
    writeInt<uint16_t>(Flags);
    writeInt<uint32_t>(0);
    return Error::success();
  }

  auto *PopupPtr = cast<PopupItem>(DefPtr);
  writeInt<uint16_t>(Flags);
  RETURN_IF_ERROR(writeCString(PopupPtr->Name));
  return writeMenuDefinitionList(PopupPtr->SubItems);
}

Error ResourceFileWriter::writeMenuDefinitionList(
    const MenuDefinitionList &List) {
  for (auto &Def : List.Definitions) {
    uint16_t Flags = Def->getResFlags();
    // Last element receives an additional 0x80 flag.
    const uint16_t LastElementFlag = 0x0080;
    if (&Def == &List.Definitions.back())
      Flags |= LastElementFlag;

    RETURN_IF_ERROR(writeMenuDefinition(Def, Flags));
  }
  return Error::success();
}

Error ResourceFileWriter::writeMenuBody(const RCResource *Base) {
  // At first, MENUHEADER structure. In fact, these are two WORDs equal to 0.
  // Ref: msdn.microsoft.com/en-us/library/windows/desktop/ms648018.aspx
  writeObject<uint32_t>(0);

  return writeMenuDefinitionList(cast<MenuResource>(Base)->Elements);
}

} // namespace rc
} // namespace llvm
