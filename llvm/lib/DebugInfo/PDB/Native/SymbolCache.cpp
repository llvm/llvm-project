#include "llvm/DebugInfo/PDB/Native/SymbolCache.h"

#include "llvm/DebugInfo/CodeView/TypeDeserializer.h"
#include "llvm/DebugInfo/PDB/Native/DbiStream.h"
#include "llvm/DebugInfo/PDB/Native/NativeCompilandSymbol.h"
#include "llvm/DebugInfo/PDB/Native/NativeEnumTypes.h"
#include "llvm/DebugInfo/PDB/Native/NativeRawSymbol.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/Native/NativeTypeBuiltin.h"
#include "llvm/DebugInfo/PDB/Native/NativeTypeEnum.h"
#include "llvm/DebugInfo/PDB/Native/NativeTypePointer.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

// Maps codeview::SimpleTypeKind of a built-in type to the parameters necessary
// to instantiate a NativeBuiltinSymbol for that type.
static const struct BuiltinTypeEntry {
  codeview::SimpleTypeKind Kind;
  PDB_BuiltinType Type;
  uint32_t Size;
} BuiltinTypes[] = {
    {codeview::SimpleTypeKind::Int16Short, PDB_BuiltinType::Int, 2},
    {codeview::SimpleTypeKind::UInt16Short, PDB_BuiltinType::UInt, 2},
    {codeview::SimpleTypeKind::Int32, PDB_BuiltinType::Int, 4},
    {codeview::SimpleTypeKind::UInt32, PDB_BuiltinType::UInt, 4},
    {codeview::SimpleTypeKind::Int32Long, PDB_BuiltinType::Int, 4},
    {codeview::SimpleTypeKind::UInt32Long, PDB_BuiltinType::UInt, 4},
    {codeview::SimpleTypeKind::Int64Quad, PDB_BuiltinType::Int, 8},
    {codeview::SimpleTypeKind::UInt64Quad, PDB_BuiltinType::UInt, 8},
    {codeview::SimpleTypeKind::NarrowCharacter, PDB_BuiltinType::Char, 1},
    {codeview::SimpleTypeKind::SignedCharacter, PDB_BuiltinType::Char, 1},
    {codeview::SimpleTypeKind::UnsignedCharacter, PDB_BuiltinType::UInt, 1},
    {codeview::SimpleTypeKind::Boolean8, PDB_BuiltinType::Bool, 1}
    // This table can be grown as necessary, but these are the only types we've
    // needed so far.
};

SymbolCache::SymbolCache(NativeSession &Session, DbiStream *Dbi)
    : Session(Session), Dbi(Dbi) {
  // Id 0 is reserved for the invalid symbol.
  Cache.push_back(nullptr);

  if (Dbi)
    Compilands.resize(Dbi->modules().getModuleCount());
}

std::unique_ptr<IPDBEnumSymbols>
SymbolCache::createTypeEnumerator(codeview::TypeLeafKind Kind) {
  auto Tpi = Session.getPDBFile().getPDBTpiStream();
  if (!Tpi) {
    consumeError(Tpi.takeError());
    return nullptr;
  }
  auto &Types = Tpi->typeCollection();
  return std::unique_ptr<IPDBEnumSymbols>(
      new NativeEnumTypes(Session, Types, Kind));
}

SymIndexId SymbolCache::createSimpleType(TypeIndex Index,
                                         ModifierOptions Mods) {
  if (Index.getSimpleMode() != codeview::SimpleTypeMode::Direct) {
    SymIndexId Id = Cache.size();
    Cache.emplace_back(
        llvm::make_unique<NativeTypePointer>(Session, Id, Index));
    return Id;
  }

  SymIndexId Id = Cache.size();
  const auto Kind = Index.getSimpleKind();
  const auto It = std::find_if(
      std::begin(BuiltinTypes), std::end(BuiltinTypes),
      [Kind](const BuiltinTypeEntry &Builtin) { return Builtin.Kind == Kind; });
  if (It == std::end(BuiltinTypes))
    return 0;
  Cache.emplace_back(llvm::make_unique<NativeTypeBuiltin>(Session, Id, Mods,
                                                          It->Type, It->Size));
  TypeIndexToSymbolId[Index] = Id;
  return Id;
}

SymIndexId
SymbolCache::createSymbolForModifiedType(codeview::TypeIndex ModifierTI,
                                         codeview::CVType CVT) {
  ModifierRecord Record;
  if (auto EC = TypeDeserializer::deserializeAs<ModifierRecord>(CVT, Record)) {
    consumeError(std::move(EC));
    return 0;
  }

  if (Record.ModifiedType.isSimple())
    return createSimpleType(Record.ModifiedType, Record.Modifiers);

  auto Tpi = Session.getPDBFile().getPDBTpiStream();
  if (!Tpi) {
    consumeError(Tpi.takeError());
    return 0;
  }
  codeview::LazyRandomTypeCollection &Types = Tpi->typeCollection();

  codeview::CVType UnmodifiedType = Types.getType(Record.ModifiedType);

  switch (UnmodifiedType.kind()) {
  case LF_ENUM: {
    EnumRecord ER;
    if (auto EC =
            TypeDeserializer::deserializeAs<EnumRecord>(UnmodifiedType, ER)) {
      consumeError(std::move(EC));
      return 0;
    }
    return createSymbol<NativeTypeEnum>(Record.ModifiedType, std::move(Record),
                                        std::move(ER));
  }
  case LF_STRUCTURE:
  case LF_UNION:
  case LF_CLASS:
    // FIXME: Handle these
    break;
  default:
    // No other types can be modified.  (LF_POINTER, for example, records
    // its modifiers a different way.
    assert(false && "Invalid LF_MODIFIER record");
    break;
  }
  return createSymbolPlaceholder();
}

SymIndexId SymbolCache::findSymbolByTypeIndex(codeview::TypeIndex Index) {
  // First see if it's already in our cache.
  const auto Entry = TypeIndexToSymbolId.find(Index);
  if (Entry != TypeIndexToSymbolId.end())
    return Entry->second;

  // Symbols for built-in types are created on the fly.
  if (Index.isSimple())
    return createSimpleType(Index, ModifierOptions::None);

  // We need to instantiate and cache the desired type symbol.
  auto Tpi = Session.getPDBFile().getPDBTpiStream();
  if (!Tpi) {
    consumeError(Tpi.takeError());
    return 0;
  }
  codeview::LazyRandomTypeCollection &Types = Tpi->typeCollection();
  codeview::CVType CVT = Types.getType(Index);
  // TODO(amccarth):  Make this handle all types.
  SymIndexId Id = 0;

  switch (CVT.kind()) {
  case codeview::LF_ENUM:
    Id = createSymbolForType<NativeTypeEnum, EnumRecord>(Index, std::move(CVT));
    break;
  case codeview::LF_POINTER:
    Id = createSymbolForType<NativeTypePointer, PointerRecord>(Index,
                                                               std::move(CVT));
    break;
  case codeview::LF_MODIFIER:
    Id = createSymbolForModifiedType(Index, std::move(CVT));
    break;
  default:
    Id = createSymbolPlaceholder();
    break;
  }
  if (Id != 0)
    TypeIndexToSymbolId[Index] = Id;
  return Id;
}

std::unique_ptr<PDBSymbol>
SymbolCache::getSymbolById(SymIndexId SymbolId) const {
  assert(SymbolId < Cache.size());

  // Id 0 is reserved.
  if (SymbolId == 0 || SymbolId >= Cache.size())
    return nullptr;

  // Make sure to handle the case where we've inserted a placeholder symbol
  // for types we don't yet suppport.
  NativeRawSymbol *NRS = Cache[SymbolId].get();
  if (!NRS)
    return nullptr;

  return PDBSymbol::create(Session, *NRS);
}

NativeRawSymbol &SymbolCache::getNativeSymbolById(SymIndexId SymbolId) const {
  return *Cache[SymbolId];
}

uint32_t SymbolCache::getNumCompilands() const {
  if (!Dbi)
    return 0;

  return Dbi->modules().getModuleCount();
}

std::unique_ptr<PDBSymbolCompiland>
SymbolCache::getOrCreateCompiland(uint32_t Index) {
  if (!Dbi)
    return nullptr;

  if (Index >= Compilands.size())
    return nullptr;

  if (Compilands[Index] == 0) {
    const DbiModuleList &Modules = Dbi->modules();
    Compilands[Index] =
        createSymbol<NativeCompilandSymbol>(Modules.getModuleDescriptor(Index));
  }

  return Session.getConcreteSymbolById<PDBSymbolCompiland>(Compilands[Index]);
}
