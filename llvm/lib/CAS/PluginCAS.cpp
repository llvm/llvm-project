//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implements \c ObjectStore and \c ActionCache on top of a dynamically loaded
/// plugin that provides the C API in \c "llvm-c/CAS/PluginAPI_functions.h".
///
/// The asynchronous entry points of the plugin API are not called yet; they
/// will be wired up once \c ObjectStore and \c ActionCache grow asynchronous
/// interfaces.
///
//===----------------------------------------------------------------------===//

#include "PluginAPI.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"

using namespace llvm;
using namespace llvm::cas;

namespace {

class PluginCASContext : public CASContext {
public:
  void printIDImpl(raw_ostream &OS, const CASID &ID) const final;

  StringRef getHashSchemaIdentifier() const final { return SchemaName; }

  static Expected<std::shared_ptr<PluginCASContext>>
  create(StringRef PluginPath, StringRef OnDiskPath,
         ArrayRef<std::pair<std::string, std::string>> PluginArgs);

  ~PluginCASContext() { Functions.cas_dispose(c_cas); }

  llcas_functions_t Functions{};
  llcas_cas_t c_cas = nullptr;
  std::string SchemaName;

  static Error errorAndDispose(char *c_err, const llcas_functions_t &Funcs) {
    Error E = createStringError(inconvertibleErrorCode(), c_err);
    Funcs.string_dispose(c_err);
    return E;
  }

  Error errorAndDispose(char *c_err) const {
    return errorAndDispose(c_err, Functions);
  }
};

} // anonymous namespace

void PluginCASContext::printIDImpl(raw_ostream &OS, const CASID &ID) const {
  ArrayRef<uint8_t> Hash = ID.getHash();
  char *c_printed_id = nullptr;
  char *c_err = nullptr;
  if (Functions.digest_print(c_cas, llcas_digest_t{Hash.data(), Hash.size()},
                             &c_printed_id, &c_err))
    report_fatal_error(errorAndDispose(c_err));
  OS << c_printed_id;
  Functions.string_dispose(c_printed_id);
}

Expected<std::shared_ptr<PluginCASContext>> PluginCASContext::create(
    StringRef PluginPath, StringRef OnDiskPath,
    ArrayRef<std::pair<std::string, std::string>> PluginArgs) {
  auto reportError = [PluginPath](const Twine &Description) -> Error {
    std::error_code EC = inconvertibleErrorCode();
    return createStringError(EC, "error loading '" + PluginPath +
                                     "': " + Description);
  };

  SmallString<256> PathBuf = PluginPath;
  std::string ErrMsg;
  sys::DynamicLibrary Lib =
      sys::DynamicLibrary::getPermanentLibrary(PathBuf.c_str(), &ErrMsg);
  if (!Lib.isValid())
    return reportError(ErrMsg);

  llcas_functions_t Functions{};

#define CASPLUGINAPI_FUNCTION(name, required)                                  \
  if (!(Functions.name = (decltype(llcas_functions_t::name))                   \
                             Lib.getAddressOfSymbol("llcas_" #name))) {        \
    if (required)                                                              \
      return reportError("failed symbol 'llcas_" #name "' lookup");            \
  }
#include "PluginAPI_functions.def"
#undef CASPLUGINAPI_FUNCTION

  llcas_cas_options_t c_opts = Functions.cas_options_create();
  scope_exit DisposeOptions([&]() { Functions.cas_options_dispose(c_opts); });

  Functions.cas_options_set_client_version(c_opts, LLCAS_VERSION_MAJOR,
                                           LLCAS_VERSION_MINOR);
  SmallString<256> OnDiskPathBuf = OnDiskPath;
  Functions.cas_options_set_ondisk_path(c_opts, OnDiskPathBuf.c_str());
  for (const auto &Pair : PluginArgs) {
    char *c_err = nullptr;
    if (Functions.cas_options_set_option(c_opts, Pair.first.c_str(),
                                         Pair.second.c_str(), &c_err))
      return errorAndDispose(c_err, Functions);
  }

  char *c_err = nullptr;
  llcas_cas_t c_cas = Functions.cas_create(c_opts, &c_err);
  if (!c_cas)
    return errorAndDispose(c_err, Functions);

  char *c_schema = Functions.cas_get_hash_schema_name(c_cas);
  std::string SchemaName = c_schema;
  Functions.string_dispose(c_schema);

  auto Ctx = std::make_shared<PluginCASContext>();
  Ctx->Functions = Functions;
  Ctx->c_cas = c_cas;
  Ctx->SchemaName = std::move(SchemaName);
  return Ctx;
}

//===----------------------------------------------------------------------===//
// ObjectStore API
//===----------------------------------------------------------------------===//

namespace {

class PluginObjectStore : public ObjectStore {
public:
  Expected<CASID> parseID(StringRef ID) final;
  Expected<ObjectRef> store(ArrayRef<ObjectRef> Refs,
                            ArrayRef<char> Data) final;
  Expected<ObjectRef> storeFromFile(StringRef Path) final;
  Error exportDataToFile(ObjectHandle Node, StringRef Path) const final;
  CASID getID(ObjectRef Ref) const final;
  std::optional<ObjectRef> getReference(const CASID &ID) const final;
  Expected<bool> isMaterialized(ObjectRef Ref) const final;
  Expected<std::optional<ObjectHandle>> loadIfExists(ObjectRef Ref) final;
  uint64_t getDataSize(ObjectHandle Node) const final;
  Error forEachRef(ObjectHandle Node,
                   function_ref<Error(ObjectRef)> Callback) const final;
  ObjectRef readRef(ObjectHandle Node, size_t I) const final;
  size_t getNumRefs(ObjectHandle Node) const final;
  ArrayRef<char> getData(ObjectHandle Node,
                         bool RequiresNullTerminator = false) const final;
  Error validateObject(const CASID &ID) final {
    // Not supported yet. Always return success.
    return Error::success();
  }

  Error validate(bool CheckHash) const final;

  Error setSizeLimit(std::optional<uint64_t> SizeLimit) final;
  Expected<std::optional<uint64_t>> getStorageSize() const final;
  Error pruneStorageData() final;

  PluginObjectStore(std::shared_ptr<PluginCASContext>);

  /// Exposes \c makeObjectRef to the file-local helpers below.
  ObjectRef makeRef(uint64_t InternalRef) const {
    return makeObjectRef(InternalRef);
  }

  std::shared_ptr<PluginCASContext> Ctx;
};

} // anonymous namespace

Expected<CASID> PluginObjectStore::parseID(StringRef ID) {
  // Use big enough stack so that we don't have to allocate in the heap.
  SmallString<148> IDBuf(ID);
  SmallVector<uint8_t, 68> BytesBuf(68);

  auto parseDigest = [&]() -> Expected<unsigned> {
    char *c_err = nullptr;
    unsigned NumBytes = Ctx->Functions.digest_parse(
        Ctx->c_cas, IDBuf.c_str(), BytesBuf.data(), BytesBuf.size(), &c_err);
    if (NumBytes == 0)
      return Ctx->errorAndDispose(c_err);
    return NumBytes;
  };

  Expected<unsigned> NumBytes = parseDigest();
  if (!NumBytes)
    return NumBytes.takeError();

  if (*NumBytes > BytesBuf.size()) {
    BytesBuf.resize(*NumBytes);
    NumBytes = parseDigest();
    if (!NumBytes)
      return NumBytes.takeError();
    assert(*NumBytes == BytesBuf.size());
  } else {
    BytesBuf.truncate(*NumBytes);
  }

  return CASID::create(Ctx.get(), toStringRef(BytesBuf));
}

Expected<ObjectRef> PluginObjectStore::store(ArrayRef<ObjectRef> Refs,
                                             ArrayRef<char> Data) {
  SmallVector<llcas_objectid_t, 64> c_ids;
  c_ids.reserve(Refs.size());
  for (ObjectRef Ref : Refs) {
    c_ids.push_back(llcas_objectid_t{Ref.getInternalRef(*this)});
  }

  llcas_objectid_t c_stored_id;
  char *c_err = nullptr;
  if (Ctx->Functions.cas_store_object(
          Ctx->c_cas, llcas_data_t{Data.data(), Data.size()}, c_ids.data(),
          c_ids.size(), &c_stored_id, &c_err))
    return Ctx->errorAndDispose(c_err);

  return makeObjectRef(c_stored_id.opaque);
}

Expected<ObjectRef> PluginObjectStore::storeFromFile(StringRef Path) {
  if (!Ctx->Functions.cas_store_from_filepath)
    return ObjectStore::storeFromFile(Path);

  llcas_objectid_t c_stored_id;
  char *c_err = nullptr;
  std::string PathStr = Path.str();
  if (Ctx->Functions.cas_store_from_filepath(Ctx->c_cas, PathStr.c_str(),
                                             &c_stored_id, &c_err))
    return Ctx->errorAndDispose(c_err);

  return makeObjectRef(c_stored_id.opaque);
}

Error PluginObjectStore::exportDataToFile(ObjectHandle Node,
                                          StringRef Path) const {
  if (!Ctx->Functions.loaded_object_export_data_to_filepath)
    return ObjectStore::exportDataToFile(Node, Path);

  char *c_err = nullptr;
  std::string PathStr = Path.str();
  if (Ctx->Functions.loaded_object_export_data_to_filepath(
          Ctx->c_cas, llcas_loaded_object_t{Node.getInternalRef(*this)},
          PathStr.c_str(), &c_err))
    return Ctx->errorAndDispose(c_err);

  return Error::success();
}

static StringRef toStringRef(llcas_digest_t c_digest) {
  return StringRef((const char *)c_digest.data, c_digest.size);
}

CASID PluginObjectStore::getID(ObjectRef Ref) const {
  llcas_objectid_t c_id{Ref.getInternalRef(*this)};
  llcas_digest_t c_digest =
      Ctx->Functions.objectid_get_digest(Ctx->c_cas, c_id);
  return CASID::create(Ctx.get(), toStringRef(c_digest));
}

std::optional<ObjectRef>
PluginObjectStore::getReference(const CASID &ID) const {
  ArrayRef<uint8_t> Hash = ID.getHash();
  llcas_objectid_t c_id;
  char *c_err = nullptr;
  if (Ctx->Functions.cas_get_objectid(
          Ctx->c_cas, llcas_digest_t{Hash.data(), Hash.size()}, &c_id, &c_err))
    report_fatal_error(Ctx->errorAndDispose(c_err));

  return makeObjectRef(c_id.opaque);
}

Expected<bool> PluginObjectStore::isMaterialized(ObjectRef Ref) const {
  llcas_objectid_t c_id{Ref.getInternalRef(*this)};
  char *c_err = nullptr;
  llcas_lookup_result_t c_result = Ctx->Functions.cas_contains_object(
      Ctx->c_cas, c_id, /*globally=*/false, &c_err);
  switch (c_result) {
  case LLCAS_LOOKUP_RESULT_SUCCESS:
    return true;
  case LLCAS_LOOKUP_RESULT_NOTFOUND:
    return false;
  case LLCAS_LOOKUP_RESULT_ERROR:
    return Ctx->errorAndDispose(c_err);
  }
  llvm_unreachable("unknown llcas_lookup_result_t value");
}

Expected<std::optional<ObjectHandle>>
PluginObjectStore::loadIfExists(ObjectRef Ref) {
  llcas_objectid_t c_id{Ref.getInternalRef(*this)};
  llcas_loaded_object_t c_obj;
  char *c_err = nullptr;
  llcas_lookup_result_t c_result =
      Ctx->Functions.cas_load_object(Ctx->c_cas, c_id, &c_obj, &c_err);
  switch (c_result) {
  case LLCAS_LOOKUP_RESULT_SUCCESS:
    return makeObjectHandle(c_obj.opaque);
  case LLCAS_LOOKUP_RESULT_NOTFOUND:
    return std::nullopt;
  case LLCAS_LOOKUP_RESULT_ERROR:
    return Ctx->errorAndDispose(c_err);
  }
  llvm_unreachable("unknown llcas_lookup_result_t value");
}

namespace {

class ObjectRefsWrapper {
public:
  ObjectRefsWrapper(const ObjectHandle &Node, const PluginObjectStore &Store)
      : Store(Store), Ctx(*Store.Ctx) {
    llcas_loaded_object_t c_obj{Node.getInternalRef(Store)};
    this->c_refs = Ctx.Functions.loaded_object_get_refs(Ctx.c_cas, c_obj);
  }

  size_t size() const {
    return Ctx.Functions.object_refs_get_count(Ctx.c_cas, c_refs);
  }

  ObjectRef operator[](size_t I) const {
    llcas_objectid_t c_id =
        Ctx.Functions.object_refs_get_id(Ctx.c_cas, c_refs, I);
    return Store.makeRef(c_id.opaque);
  }

private:
  const PluginObjectStore &Store;
  PluginCASContext &Ctx;
  llcas_object_refs_t c_refs;
};

} // namespace

// FIXME: Replace forEachRef/readRef/getNumRefs APIs with an iterator interface.
Error PluginObjectStore::forEachRef(
    ObjectHandle Node, function_ref<Error(ObjectRef)> Callback) const {
  ObjectRefsWrapper Refs(Node, *this);
  for (unsigned I = 0, E = Refs.size(); I != E; ++I) {
    if (Error E = Callback(Refs[I]))
      return E;
  }
  return Error::success();
}

ObjectRef PluginObjectStore::readRef(ObjectHandle Node, size_t I) const {
  ObjectRefsWrapper Refs(Node, *this);
  return Refs[I];
}

size_t PluginObjectStore::getNumRefs(ObjectHandle Node) const {
  ObjectRefsWrapper Refs(Node, *this);
  return Refs.size();
}

// FIXME: Remove getDataSize(ObjectHandle) from API requirement,
// \c getData(ObjectHandle) should be enough.
uint64_t PluginObjectStore::getDataSize(ObjectHandle Node) const {
  ArrayRef<char> Data = getData(Node);
  return Data.size();
}

ArrayRef<char> PluginObjectStore::getData(ObjectHandle Node,
                                          bool RequiresNullTerminator) const {
  // FIXME: Remove RequiresNullTerminator from ObjectStore API requirement?
  // It is a requirement for the plugin API.
  llcas_data_t c_data = Ctx->Functions.loaded_object_get_data(
      Ctx->c_cas, llcas_loaded_object_t{Node.getInternalRef(*this)});
  return ArrayRef((const char *)c_data.data, c_data.size);
}

Error PluginObjectStore::setSizeLimit(std::optional<uint64_t> SizeLimit) {
  if (Ctx->Functions.cas_set_ondisk_size_limit) {
    char *c_err = nullptr;
    if (Ctx->Functions.cas_set_ondisk_size_limit(Ctx->c_cas,
                                                 SizeLimit.value_or(0), &c_err))
      return Ctx->errorAndDispose(c_err);
  }
  return Error::success();
}

Expected<std::optional<uint64_t>> PluginObjectStore::getStorageSize() const {
  if (!Ctx->Functions.cas_get_ondisk_size)
    return std::nullopt;
  char *c_err = nullptr;
  int64_t ret = Ctx->Functions.cas_get_ondisk_size(Ctx->c_cas, &c_err);
  switch (ret) {
  case -1:
    return std::nullopt;
  case -2:
    return Ctx->errorAndDispose(c_err);
  default:
    return ret;
  }
}

Error PluginObjectStore::pruneStorageData() {
  if (Ctx->Functions.cas_prune_ondisk_data) {
    char *c_err = nullptr;
    if (Ctx->Functions.cas_prune_ondisk_data(Ctx->c_cas, &c_err))
      return Ctx->errorAndDispose(c_err);
  }
  return Error::success();
}

Error PluginObjectStore::validate(bool CheckHash) const {
  if (Ctx->Functions.cas_validate) {
    char *c_err = nullptr;
    if (Ctx->Functions.cas_validate(Ctx->c_cas, CheckHash, &c_err))
      return Ctx->errorAndDispose(c_err);
    return Error::success();
  }
  return createStringError("plugin cas doesn't support validation");
}

PluginObjectStore::PluginObjectStore(std::shared_ptr<PluginCASContext> CASCtx)
    : ObjectStore(*CASCtx), Ctx(std::move(CASCtx)) {}

//===----------------------------------------------------------------------===//
// ActionCache API
//===----------------------------------------------------------------------===//

namespace {

class PluginActionCache : public ActionCache {
public:
  Expected<std::optional<CASID>> getImpl(ArrayRef<uint8_t> ResolvedKey,
                                         bool CanBeDistributed) const final;

  Error putImpl(ArrayRef<uint8_t> ResolvedKey, const CASID &Result,
                bool CanBeDistributed) final;

  PluginActionCache(std::shared_ptr<PluginCASContext>);

  Error validate() const final;

private:
  std::shared_ptr<PluginCASContext> Ctx;
};

} // anonymous namespace

Expected<std::optional<CASID>>
PluginActionCache::getImpl(ArrayRef<uint8_t> ResolvedKey,
                           bool CanBeDistributed) const {
  llcas_objectid_t c_value;
  char *c_err = nullptr;
  llcas_lookup_result_t c_result = Ctx->Functions.actioncache_get_for_digest(
      Ctx->c_cas, llcas_digest_t{ResolvedKey.data(), ResolvedKey.size()},
      &c_value, CanBeDistributed, &c_err);
  switch (c_result) {
  case LLCAS_LOOKUP_RESULT_SUCCESS: {
    llcas_digest_t c_digest =
        Ctx->Functions.objectid_get_digest(Ctx->c_cas, c_value);
    return CASID::create(Ctx.get(), toStringRef(c_digest));
  }
  case LLCAS_LOOKUP_RESULT_NOTFOUND:
    return std::nullopt;
  case LLCAS_LOOKUP_RESULT_ERROR:
    return Ctx->errorAndDispose(c_err);
  }
  llvm_unreachable("unknown llcas_lookup_result_t value");
}

Error PluginActionCache::putImpl(ArrayRef<uint8_t> ResolvedKey,
                                 const CASID &Result, bool CanBeDistributed) {
  ArrayRef<uint8_t> Hash = Result.getHash();
  llcas_objectid_t c_value;
  char *c_err = nullptr;
  if (Ctx->Functions.cas_get_objectid(Ctx->c_cas,
                                      llcas_digest_t{Hash.data(), Hash.size()},
                                      &c_value, &c_err))
    return Ctx->errorAndDispose(c_err);

  if (Ctx->Functions.actioncache_put_for_digest(
          Ctx->c_cas, llcas_digest_t{ResolvedKey.data(), ResolvedKey.size()},
          c_value, CanBeDistributed, &c_err))
    return Ctx->errorAndDispose(c_err);

  return Error::success();
}

PluginActionCache::PluginActionCache(std::shared_ptr<PluginCASContext> CASCtx)
    : ActionCache(*CASCtx), Ctx(std::move(CASCtx)) {}

Error PluginActionCache::validate() const {
  if (Ctx->Functions.actioncache_validate) {
    char *c_err = nullptr;
    if (Ctx->Functions.actioncache_validate(Ctx->c_cas, &c_err))
      return Ctx->errorAndDispose(c_err);
    return Error::success();
  }
  return createStringError("plugin action cache doesn't support validation");
}

//===----------------------------------------------------------------------===//
// createPluginCASDatabases API
//===----------------------------------------------------------------------===//

Expected<std::pair<std::shared_ptr<ObjectStore>, std::shared_ptr<ActionCache>>>
cas::createPluginCASDatabases(
    StringRef PluginPath, StringRef OnDiskPath,
    ArrayRef<std::pair<std::string, std::string>> PluginArgs) {
  std::shared_ptr<PluginCASContext> Ctx;
  if (Error E = PluginCASContext::create(PluginPath, OnDiskPath, PluginArgs)
                    .moveInto(Ctx))
    return std::move(E);
  auto CAS = std::make_shared<PluginObjectStore>(Ctx);
  auto AC = std::make_shared<PluginActionCache>(std::move(Ctx));
  return std::make_pair(std::move(CAS), std::move(AC));
}
