#include "llvm/DebugInfo/PDB/Native/SymbolCache.h"

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
using namespace llvm::pdb;

// Maps codeview::SimpleTypeKind of a built-in type to the parameters necessary
// to instantiate a NativeBuiltinSymbol for that type.
static const struct BuiltinTypeEntry {
  codeview::SimpleTypeKind Kind;
  PDB_BuiltinType Type;
  uint32_t Size;
} BuiltinTypes[] = {
    {codeview::SimpleTypeKind::Int32, PDB_BuiltinType::Int, 4},
    {codeview::SimpleTypeKind::UInt32, PDB_BuiltinType::UInt, 4},
    {codeview::SimpleTypeKind::UInt32Long, PDB_BuiltinType::UInt, 4},
    {codeview::SimpleTypeKind::UInt64Quad, PDB_BuiltinType::UInt, 8},
    {codeview::SimpleTypeKind::NarrowCharacter, PDB_BuiltinType::Char, 1},
    {codeview::SimpleTypeKind::SignedCharacter, PDB_BuiltinType::Char, 1},
    {codeview::SimpleTypeKind::UnsignedCharacter, PDB_BuiltinType::UInt, 1},
    {codeview::SimpleTypeKind::UInt16Short, PDB_BuiltinType::UInt, 2},
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

std::unique_ptr<PDBSymbolTypeEnum>
SymbolCache::createEnumSymbol(codeview::TypeIndex Index) {
  const auto Id = findSymbolByTypeIndex(Index);
  return PDBSymbol::createAs<PDBSymbolTypeEnum>(Session, *Cache[Id]);
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

SymIndexId SymbolCache::findSymbolByTypeIndex(codeview::TypeIndex Index) {
  // First see if it's already in our cache.
  const auto Entry = TypeIndexToSymbolId.find(Index);
  if (Entry != TypeIndexToSymbolId.end())
    return Entry->second;

  // Symbols for built-in types are created on the fly.
  if (Index.isSimple()) {
    // FIXME:  We will eventually need to handle pointers to other simple types,
    // which are still simple types in the world of CodeView TypeIndexes.
    if (Index.getSimpleMode() != codeview::SimpleTypeMode::Direct)
      return 0;
    const auto Kind = Index.getSimpleKind();
    const auto It =
        std::find_if(std::begin(BuiltinTypes), std::end(BuiltinTypes),
                     [Kind](const BuiltinTypeEntry &Builtin) {
                       return Builtin.Kind == Kind;
                     });
    if (It == std::end(BuiltinTypes))
      return 0;
    SymIndexId Id = Cache.size();
    Cache.emplace_back(
        llvm::make_unique<NativeTypeBuiltin>(Session, Id, It->Type, It->Size));
    TypeIndexToSymbolId[Index] = Id;
    return Id;
  }

  // We need to instantiate and cache the desired type symbol.
  auto Tpi = Session.getPDBFile().getPDBTpiStream();
  if (!Tpi) {
    consumeError(Tpi.takeError());
    return 0;
  }
  codeview::LazyRandomTypeCollection &Types = Tpi->typeCollection();
  const codeview::CVType &CVT = Types.getType(Index);
  // TODO(amccarth):  Make this handle all types, not just LF_ENUMs.
  SymIndexId Id = 0;
  switch (CVT.kind()) {
  case codeview::LF_ENUM:
    Id = createSymbol<NativeTypeEnum>(CVT);
    break;
  case codeview::LF_POINTER:
    Id = createSymbol<NativeTypePointer>(CVT);
    break;
  default:
    Id = createSymbolPlaceholder();
    break;
  }
  TypeIndexToSymbolId[Index] = Id;
  return Id;
}

std::unique_ptr<PDBSymbol>
SymbolCache::getSymbolById(SymIndexId SymbolId) const {
  // Id 0 is reserved.
  if (SymbolId == 0)
    return nullptr;

  // If the caller has a SymbolId, it'd better be in our SymbolCache.
  return SymbolId < Cache.size() ? PDBSymbol::create(Session, *Cache[SymbolId])
                                 : nullptr;
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
