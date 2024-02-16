//===- llvm/tools/libCASPluginTest/libCASPluginTest.cpp ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the LLVM CAS plugin API, for testing purposes.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/CAS/PluginAPI_functions.h"
#include "llvm/CAS/BuiltinCASContext.h"
#include "llvm/CAS/BuiltinObjectHasher.h"
#include "llvm/CAS/UnifiedOnDiskCache.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ThreadPool.h"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::cas::builtin;
using namespace llvm::cas::ondisk;

static char *copyNewMallocString(StringRef Str) {
  char *c_str = (char *)malloc(Str.size() + 1);
  std::uninitialized_copy(Str.begin(), Str.end(), c_str);
  c_str[Str.size()] = '\0';
  return c_str;
}

template <typename ResT>
static ResT reportError(Error &&E, char **error, ResT Result = ResT()) {
  if (error)
    *error = copyNewMallocString(toString(std::move(E)));
  return Result;
}

void llcas_get_plugin_version(unsigned *major, unsigned *minor) {
  *major = LLCAS_VERSION_MAJOR;
  *minor = LLCAS_VERSION_MINOR;
}

void llcas_string_dispose(char *str) { free(str); }

namespace {

struct CASPluginOptions {
  std::string OnDiskPath;
  std::string UpstreamPath;
  std::string FirstPrefix;
  std::string SecondPrefix;
  bool SimulateMissingObjects = false;
  bool Logging = true;

  Error setOption(StringRef Name, StringRef Value);
};

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(CASPluginOptions, llcas_cas_options_t)

} // namespace

Error CASPluginOptions::setOption(StringRef Name, StringRef Value) {
  if (Name == "first-prefix")
    FirstPrefix = Value;
  else if (Name == "second-prefix")
    SecondPrefix = Value;
  else if (Name == "upstream-path")
    UpstreamPath = Value;
  else if (Name == "simulate-missing-objects")
    SimulateMissingObjects = true;
  else if (Name == "no-logging")
    Logging = false;
  else
    return createStringError(errc::invalid_argument,
                             Twine("unknown option: ") + Name);
  return Error::success();
}

llcas_cas_options_t llcas_cas_options_create(void) {
  return wrap(new CASPluginOptions());
}

void llcas_cas_options_dispose(llcas_cas_options_t c_opts) {
  delete unwrap(c_opts);
}

void llcas_cas_options_set_ondisk_path(llcas_cas_options_t c_opts,
                                       const char *path) {
  auto &Opts = *unwrap(c_opts);
  Opts.OnDiskPath = path;
}

bool llcas_cas_options_set_option(llcas_cas_options_t c_opts, const char *name,
                                  const char *value, char **error) {
  auto &Opts = *unwrap(c_opts);
  if (Error E = Opts.setOption(name, value))
    return reportError(std::move(E), error, true);
  return false;
}

namespace {

struct CASWrapper {
  std::string FirstPrefix;
  std::string SecondPrefix;
  /// If true, asynchronous "download" of an object will treat it as missing.
  bool SimulateMissingObjects = false;
  bool Logging = true;
  std::unique_ptr<UnifiedOnDiskCache> DB;
  /// Used for testing the \c globally parameter of action cache APIs. Simulates
  /// "uploading"/"downloading" objects from/to the primary on-disk path.
  std::unique_ptr<UnifiedOnDiskCache> UpstreamDB;
  ThreadPool Pool{llvm::hardware_concurrency()};

  std::mutex Lock{};

  /// Check if the object is contained, in the "local" CAS only or "globally".
  bool containsObject(ObjectID ID, bool Globally);

  /// Load the object, potentially "downloading" it from upstream.
  Expected<std::optional<ondisk::ObjectHandle>> loadObject(ObjectID ID);

  /// "Uploads" a key and the associated full node graph.
  Error upstreamKey(ArrayRef<uint8_t> Key, ObjectID Value);

  /// "Downloads" the ID associated with the key but not the node data. The node
  /// itself and the rest of the nodes in the graph will be "downloaded" lazily
  /// as they are visited.
  Expected<std::optional<ObjectID>> downstreamKey(ArrayRef<uint8_t> Key);

  /// Synchronized access to \c llvm::errs().
  void syncErrs(llvm::function_ref<void(raw_ostream &OS)> Fn) {
    if (!Logging) {
      // Ignore log output.
      SmallString<32> Buf;
      raw_svector_ostream OS(Buf);
      Fn(OS);
      return;
    }
    std::unique_lock<std::mutex> LockGuard(Lock);
    Fn(errs());
    errs().flush();
  }

private:
  /// "Uploads" the full object node graph.
  Expected<ObjectID> upstreamNode(ObjectID Node);
  /// "Downloads" only a single object node. The rest of the nodes in the graph
  /// will be "downloaded" lazily as they are visited.
  Expected<ObjectID> downstreamNode(ObjectID Node);
};

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(CASWrapper, llcas_cas_t)

} // namespace

bool CASWrapper::containsObject(ObjectID ID, bool Globally) {
  if (DB->getGraphDB().containsObject(ID))
    return true;
  if (!Globally || !UpstreamDB)
    return false;

  ObjectID UpstreamID =
      UpstreamDB->getGraphDB().getReference(DB->getGraphDB().getDigest(ID));
  return UpstreamDB->getGraphDB().containsObject(UpstreamID);
}

Expected<std::optional<ondisk::ObjectHandle>>
CASWrapper::loadObject(ObjectID ID) {
  std::optional<ondisk::ObjectHandle> Obj;
  if (Error E = DB->getGraphDB().load(ID).moveInto(Obj))
    return std::move(E);
  if (Obj)
    return Obj;
  if (!UpstreamDB)
    return std::nullopt;

  // Try "downloading" the node from upstream.
  ObjectID UpstreamID =
      UpstreamDB->getGraphDB().getReference(DB->getGraphDB().getDigest(ID));
  std::optional<ObjectID> Ret;
  if (Error E = downstreamNode(UpstreamID).moveInto(Ret))
    return std::move(E);
  return DB->getGraphDB().load(ID);
}

/// Imports a single object node.
static Expected<ObjectID> importNode(ObjectID FromID, OnDiskGraphDB &FromDB,
                                     OnDiskGraphDB &ToDB) {
  ObjectID ToID = ToDB.getReference(FromDB.getDigest(FromID));
  if (ToDB.containsObject(ToID))
    return ToID;

  std::optional<ondisk::ObjectHandle> FromH;
  if (Error E = FromDB.load(FromID).moveInto(FromH))
    return std::move(E);
  if (!FromH)
    return ToID;

  auto Data = FromDB.getObjectData(*FromH);
  auto FromRefs = FromDB.getObjectRefs(*FromH);
  SmallVector<ObjectID> Refs;
  for (ObjectID FromRef : FromRefs)
    Refs.push_back(ToDB.getReference(FromDB.getDigest(FromRef)));

  if (Error E = ToDB.store(ToID, Refs, Data))
    return std::move(E);
  return ToID;
}

Expected<ObjectID> CASWrapper::upstreamNode(ObjectID Node) {
  OnDiskGraphDB &FromDB = DB->getGraphDB();
  OnDiskGraphDB &ToDB = UpstreamDB->getGraphDB();

  std::optional<ondisk::ObjectHandle> FromH;
  if (Error E = FromDB.load(Node).moveInto(FromH))
    return std::move(E);
  if (!FromH)
    return createStringError(errc::invalid_argument, "node doesn't exist");

  for (ObjectID Ref : FromDB.getObjectRefs(*FromH)) {
    std::optional<ObjectID> ID;
    if (Error E = upstreamNode(Ref).moveInto(ID))
      return std::move(E);
  }

  return importNode(Node, FromDB, ToDB);
}

Expected<ObjectID> CASWrapper::downstreamNode(ObjectID Node) {
  OnDiskGraphDB &FromDB = UpstreamDB->getGraphDB();
  OnDiskGraphDB &ToDB = DB->getGraphDB();
  return importNode(Node, FromDB, ToDB);
}

Error CASWrapper::upstreamKey(ArrayRef<uint8_t> Key, ObjectID Value) {
  if (!UpstreamDB)
    return Error::success();
  Expected<ObjectID> UpstreamVal = upstreamNode(Value);
  if (!UpstreamVal)
    return UpstreamVal.takeError();
  Expected<ObjectID> PutValue = UpstreamDB->KVPut(Key, *UpstreamVal);
  if (!PutValue)
    return PutValue.takeError();
  assert(*PutValue == *UpstreamVal);
  return Error::success();
}

Expected<std::optional<ObjectID>>
CASWrapper::downstreamKey(ArrayRef<uint8_t> Key) {
  if (!UpstreamDB)
    return std::nullopt;
  std::optional<ObjectID> UpstreamValue;
  if (Error E = UpstreamDB->KVGet(Key).moveInto(UpstreamValue))
    return std::move(E);
  if (!UpstreamValue)
    return std::nullopt;

  ObjectID Value = DB->getGraphDB().getReference(
      UpstreamDB->getGraphDB().getDigest(*UpstreamValue));
  Expected<ObjectID> PutValue = DB->KVPut(Key, Value);
  if (!PutValue)
    return PutValue.takeError();
  assert(*PutValue == Value);
  return PutValue;
}

llcas_cas_t llcas_cas_create(llcas_cas_options_t c_opts, char **error) {
  auto &Opts = *unwrap(c_opts);
  Expected<std::unique_ptr<UnifiedOnDiskCache>> DB = UnifiedOnDiskCache::open(
      Opts.OnDiskPath, /*SizeLimit=*/std::nullopt,
      BuiltinCASContext::getHashName(), sizeof(HashType));
  if (!DB)
    return reportError<llcas_cas_t>(DB.takeError(), error);

  std::unique_ptr<UnifiedOnDiskCache> UpstreamDB;
  if (!Opts.UpstreamPath.empty()) {
    if (Error E = UnifiedOnDiskCache::open(
                      Opts.UpstreamPath, /*SizeLimit=*/std::nullopt,
                      BuiltinCASContext::getHashName(), sizeof(HashType))
                      .moveInto(UpstreamDB))
      return reportError<llcas_cas_t>(std::move(E), error);
  }

  return wrap(new CASWrapper{Opts.FirstPrefix, Opts.SecondPrefix,
                             Opts.SimulateMissingObjects, Opts.Logging,
                             std::move(*DB), std::move(UpstreamDB)});
}

void llcas_cas_dispose(llcas_cas_t c_cas) { delete unwrap(c_cas); }

int64_t llcas_cas_get_ondisk_size(llcas_cas_t c_cas, char **error) {
  return unwrap(c_cas)->DB->getStorageSize();
}

bool llcas_cas_set_ondisk_size_limit(llcas_cas_t c_cas, int64_t size_limit,
                                     char **error) {
  std::optional<uint64_t> SizeLimit;
  if (size_limit < 0) {
    return reportError(
        llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "invalid size limit passed to llcas_cas_set_ondisk_size_limit"),
        error, true);
  }
  if (size_limit > 0) {
    SizeLimit = size_limit;
  }
  unwrap(c_cas)->DB->setSizeLimit(SizeLimit);
  return false;
}

bool llcas_cas_prune_ondisk_data(llcas_cas_t c_cas, char **error) {
  if (Error E = unwrap(c_cas)->DB->collectGarbage())
    return reportError(std::move(E), error, true);
  return false;
}

void llcas_cas_options_set_client_version(llcas_cas_options_t, unsigned major,
                                          unsigned minor) {
  // Ignore for now.
}

char *llcas_cas_get_hash_schema_name(llcas_cas_t) {
  // Using same name as builtin CAS so that it's interchangeable for testing
  // purposes.
  return copyNewMallocString("llvm.cas.builtin.v2[BLAKE3]");
}

unsigned llcas_digest_parse(llcas_cas_t c_cas, const char *printed_digest,
                            uint8_t *bytes, size_t bytes_size, char **error) {
  auto &Wrapper = *unwrap(c_cas);
  if (bytes_size < sizeof(HashType))
    return sizeof(HashType);

  StringRef PrintedDigest = printed_digest;
  bool Consumed = PrintedDigest.consume_front(Wrapper.FirstPrefix);
  assert(Consumed);
  (void)Consumed;
  Consumed = PrintedDigest.consume_front(Wrapper.SecondPrefix);
  assert(Consumed);
  (void)Consumed;

  Expected<HashType> Digest = BuiltinCASContext::parseID(PrintedDigest);
  if (!Digest)
    return reportError(Digest.takeError(), error, 0);
  std::uninitialized_copy(Digest->begin(), Digest->end(), bytes);
  return Digest->size();
}

bool llcas_digest_print(llcas_cas_t c_cas, llcas_digest_t c_digest,
                        char **printed_id, char **error) {
  auto &Wrapper = *unwrap(c_cas);
  SmallString<74> PrintDigest;
  raw_svector_ostream OS(PrintDigest);
  // Include these for testing purposes.
  OS << Wrapper.FirstPrefix << Wrapper.SecondPrefix;
  BuiltinCASContext::printID(ArrayRef(c_digest.data, c_digest.size), OS);
  *printed_id = copyNewMallocString(PrintDigest);
  return false;
}

bool llcas_cas_get_objectid(llcas_cas_t c_cas, llcas_digest_t c_digest,
                            llcas_objectid_t *c_id_p, char **error) {
  auto &CAS = unwrap(c_cas)->DB->getGraphDB();
  ObjectID ID = CAS.getReference(ArrayRef(c_digest.data, c_digest.size));
  *c_id_p = llcas_objectid_t{ID.getOpaqueData()};
  return false;
}

llcas_digest_t llcas_objectid_get_digest(llcas_cas_t c_cas,
                                         llcas_objectid_t c_id) {
  auto &CAS = unwrap(c_cas)->DB->getGraphDB();
  ObjectID ID = ObjectID::fromOpaqueData(c_id.opaque);
  ArrayRef<uint8_t> Digest = CAS.getDigest(ID);
  return llcas_digest_t{Digest.data(), Digest.size()};
}

llcas_lookup_result_t llcas_cas_contains_object(llcas_cas_t c_cas,
                                                llcas_objectid_t c_id,
                                                bool globally, char **error) {
  ObjectID ID = ObjectID::fromOpaqueData(c_id.opaque);
  return unwrap(c_cas)->containsObject(ID, globally)
             ? LLCAS_LOOKUP_RESULT_SUCCESS
             : LLCAS_LOOKUP_RESULT_NOTFOUND;
}

llcas_lookup_result_t llcas_cas_load_object(llcas_cas_t c_cas,
                                            llcas_objectid_t c_id,
                                            llcas_loaded_object_t *c_obj_p,
                                            char **error) {
  ObjectID ID = ObjectID::fromOpaqueData(c_id.opaque);
  Expected<std::optional<ondisk::ObjectHandle>> ObjOpt =
      unwrap(c_cas)->loadObject(ID);
  if (!ObjOpt)
    return reportError(ObjOpt.takeError(), error, LLCAS_LOOKUP_RESULT_ERROR);
  if (!*ObjOpt)
    return LLCAS_LOOKUP_RESULT_NOTFOUND;

  ondisk::ObjectHandle Obj = **ObjOpt;
  *c_obj_p = llcas_loaded_object_t{Obj.getOpaqueData()};
  return LLCAS_LOOKUP_RESULT_SUCCESS;
}

void llcas_cas_load_object_async(llcas_cas_t c_cas, llcas_objectid_t c_id,
                                 void *ctx_cb, llcas_cas_load_object_cb cb) {
  std::string PrintedDigest;
  {
    llcas_digest_t c_digest = llcas_objectid_get_digest(c_cas, c_id);
    char *printed_id;
    char *c_err;
    bool failed = llcas_digest_print(c_cas, c_digest, &printed_id, &c_err);
    if (failed)
      report_fatal_error(Twine("digest printing failed: ") + c_err);
    PrintedDigest = printed_id;
    llcas_string_dispose(printed_id);
  }

  auto passObject = [ctx_cb,
                     cb](Expected<std::optional<ondisk::ObjectHandle>> Obj) {
    if (!Obj) {
      cb(ctx_cb, LLCAS_LOOKUP_RESULT_ERROR, llcas_loaded_object_t(),
         copyNewMallocString(toString(Obj.takeError())));
    } else if (!*Obj) {
      cb(ctx_cb, LLCAS_LOOKUP_RESULT_NOTFOUND, llcas_loaded_object_t(),
         nullptr);
    } else {
      cb(ctx_cb, LLCAS_LOOKUP_RESULT_SUCCESS,
         llcas_loaded_object_t{(*Obj)->getOpaqueData()}, nullptr);
    }
  };

  auto &CAS = unwrap(c_cas)->DB->getGraphDB();
  ObjectID ID = ObjectID::fromOpaqueData(c_id.opaque);
  if (CAS.containsObject(ID)) {
    unwrap(c_cas)->syncErrs([&](raw_ostream &OS) {
      OS << "load_object_async existing: " << PrintedDigest << '\n';
    });
    return passObject(unwrap(c_cas)->loadObject(ID));
  }

  if (!unwrap(c_cas)->UpstreamDB)
    return passObject(std::nullopt);

  // Try "downloading" the node from upstream.

  unwrap(c_cas)->syncErrs([&](raw_ostream &OS) {
    OS << "load_object_async downstream begin: " << PrintedDigest << '\n';
  });
  unwrap(c_cas)->Pool.async([=] {
    // Wait a bit for the caller to proceed.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto &Wrap = *unwrap(c_cas);
    Wrap.syncErrs([&](raw_ostream &OS) {
      OS << "load_object_async downstream end: " << PrintedDigest << '\n';
    });
    if (Wrap.SimulateMissingObjects)
      return passObject(std::nullopt);
    passObject(Wrap.loadObject(ID));
  });
}

bool llcas_cas_store_object(llcas_cas_t c_cas, llcas_data_t c_data,
                            const llcas_objectid_t *c_refs, size_t c_refs_count,
                            llcas_objectid_t *c_id_p, char **error) {
  auto &CAS = unwrap(c_cas)->DB->getGraphDB();
  SmallVector<ObjectID, 64> Refs;
  Refs.reserve(c_refs_count);
  for (unsigned I = 0; I != c_refs_count; ++I) {
    Refs.push_back(ObjectID::fromOpaqueData(c_refs[I].opaque));
  }
  ArrayRef Data((const char *)c_data.data, c_data.size);

  SmallVector<ArrayRef<uint8_t>, 8> RefHashes;
  RefHashes.reserve(c_refs_count);
  for (ObjectID Ref : Refs)
    RefHashes.push_back(CAS.getDigest(Ref));
  HashType Digest = BuiltinObjectHasher<HasherT>::hashObject(RefHashes, Data);
  ObjectID StoredID = CAS.getReference(Digest);

  if (Error E = CAS.store(StoredID, Refs, Data))
    return reportError(std::move(E), error, true);
  *c_id_p = llcas_objectid_t{StoredID.getOpaqueData()};
  return false;
}

llcas_data_t llcas_loaded_object_get_data(llcas_cas_t c_cas,
                                          llcas_loaded_object_t c_obj) {
  auto &CAS = unwrap(c_cas)->DB->getGraphDB();
  ondisk::ObjectHandle Obj = ondisk::ObjectHandle::fromOpaqueData(c_obj.opaque);
  auto Data = CAS.getObjectData(Obj);
  return llcas_data_t{Data.data(), Data.size()};
}

llcas_object_refs_t llcas_loaded_object_get_refs(llcas_cas_t c_cas,
                                                 llcas_loaded_object_t c_obj) {
  auto &CAS = unwrap(c_cas)->DB->getGraphDB();
  ondisk::ObjectHandle Obj = ondisk::ObjectHandle::fromOpaqueData(c_obj.opaque);
  auto Refs = CAS.getObjectRefs(Obj);
  return llcas_object_refs_t{Refs.begin().getOpaqueData(),
                             Refs.end().getOpaqueData()};
}

size_t llcas_object_refs_get_count(llcas_cas_t c_cas,
                                   llcas_object_refs_t c_refs) {
  auto B = object_refs_iterator::fromOpaqueData(c_refs.opaque_b);
  auto E = object_refs_iterator::fromOpaqueData(c_refs.opaque_e);
  return E - B;
}

llcas_objectid_t llcas_object_refs_get_id(llcas_cas_t c_cas,
                                          llcas_object_refs_t c_refs,
                                          size_t index) {
  auto RefsI = object_refs_iterator::fromOpaqueData(c_refs.opaque_b);
  ObjectID Ref = *(RefsI + index);
  return llcas_objectid_t{Ref.getOpaqueData()};
}

llcas_lookup_result_t
llcas_actioncache_get_for_digest(llcas_cas_t c_cas, llcas_digest_t c_key,
                                 llcas_objectid_t *p_value, bool globally,
                                 char **error) {
  auto &Wrap = *unwrap(c_cas);
  auto &DB = *Wrap.DB;
  ArrayRef Key(c_key.data, c_key.size);
  std::optional<ObjectID> Value;
  if (Error E = DB.KVGet(Key).moveInto(Value))
    return reportError(std::move(E), error, LLCAS_LOOKUP_RESULT_ERROR);
  if (!Value) {
    if (!globally)
      return LLCAS_LOOKUP_RESULT_NOTFOUND;

    if (Error E = Wrap.downstreamKey(Key).moveInto(Value))
      return reportError(std::move(E), error, LLCAS_LOOKUP_RESULT_ERROR);
    if (!Value)
      return LLCAS_LOOKUP_RESULT_NOTFOUND;
  }
  *p_value = llcas_objectid_t{Value->getOpaqueData()};
  return LLCAS_LOOKUP_RESULT_SUCCESS;
}

void llcas_actioncache_get_for_digest_async(llcas_cas_t c_cas,
                                            llcas_digest_t c_key, bool globally,
                                            void *ctx_cb,
                                            llcas_actioncache_get_cb cb) {
  ArrayRef Key(c_key.data, c_key.size);
  SmallVector<uint8_t, 32> KeyBuf(Key);

  unwrap(c_cas)->Pool.async([=] {
    llcas_objectid_t c_value;
    char *c_err;
    llcas_lookup_result_t result = llcas_actioncache_get_for_digest(
        c_cas, llcas_digest_t{KeyBuf.data(), KeyBuf.size()}, &c_value, globally,
        &c_err);
    cb(ctx_cb, result, c_value, c_err);
  });
}

bool llcas_actioncache_put_for_digest(llcas_cas_t c_cas, llcas_digest_t c_key,
                                      llcas_objectid_t c_value, bool globally,
                                      char **error) {
  auto &Wrap = *unwrap(c_cas);
  auto &DB = *Wrap.DB;
  ObjectID Value = ObjectID::fromOpaqueData(c_value.opaque);
  ArrayRef Key(c_key.data, c_key.size);
  Expected<ObjectID> Ret = DB.KVPut(Key, Value);
  if (!Ret)
    return reportError(Ret.takeError(), error, true);
  if (*Ret != Value)
    return reportError(
        createStringError(errc::invalid_argument, "cache poisoned"), error,
        true);

  if (globally) {
    if (Error E = Wrap.upstreamKey(Key, Value))
      return reportError(std::move(E), error, true);
  }

  return false;
}

void llcas_actioncache_put_for_digest_async(llcas_cas_t c_cas,
                                            llcas_digest_t c_key,
                                            llcas_objectid_t c_value,
                                            bool globally, void *ctx_cb,
                                            llcas_actioncache_put_cb cb) {
  ArrayRef Key(c_key.data, c_key.size);
  SmallVector<uint8_t, 32> KeyBuf(Key);

  unwrap(c_cas)->Pool.async([=] {
    char *c_err;
    bool failed = llcas_actioncache_put_for_digest(
        c_cas, llcas_digest_t{KeyBuf.data(), KeyBuf.size()}, c_value, globally,
        &c_err);
    cb(ctx_cb, failed, c_err);
  });
}
