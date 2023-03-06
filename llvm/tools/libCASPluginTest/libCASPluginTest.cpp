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
  std::string FirstPrefix;
  std::string SecondPrefix;

  Error setOption(StringRef Name, StringRef Value);
};

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(CASPluginOptions, llcas_cas_options_t)

} // namespace

Error CASPluginOptions::setOption(StringRef Name, StringRef Value) {
  if (Name == "first-prefix")
    FirstPrefix = Value;
  else if (Name == "second-prefix")
    SecondPrefix = Value;
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
  std::unique_ptr<UnifiedOnDiskCache> DB;
};

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(CASWrapper, llcas_cas_t)

} // namespace

llcas_cas_t llcas_cas_create(llcas_cas_options_t c_opts, char **error) {
  auto &Opts = *unwrap(c_opts);
  Expected<std::unique_ptr<UnifiedOnDiskCache>> DB = UnifiedOnDiskCache::open(
      Opts.OnDiskPath, /*SizeLimit=*/std::nullopt,
      BuiltinCASContext::getHashName(), sizeof(HashType));
  if (!DB)
    return reportError<llcas_cas_t>(DB.takeError(), error);
  return wrap(
      new CASWrapper{Opts.FirstPrefix, Opts.SecondPrefix, std::move(*DB)});
}

void llcas_cas_dispose(llcas_cas_t c_cas) { delete unwrap(c_cas); }

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
                                                char **error) {
  auto &CAS = unwrap(c_cas)->DB->getGraphDB();
  ObjectID ID = ObjectID::fromOpaqueData(c_id.opaque);
  return CAS.containsObject(ID) ? LLCAS_LOOKUP_RESULT_SUCCESS
                                : LLCAS_LOOKUP_RESULT_NOTFOUND;
}

llcas_lookup_result_t llcas_cas_load_object(llcas_cas_t c_cas,
                                            llcas_objectid_t c_id,
                                            llcas_loaded_object_t *c_obj_p,
                                            char **error) {
  auto &CAS = unwrap(c_cas)->DB->getGraphDB();
  ObjectID ID = ObjectID::fromOpaqueData(c_id.opaque);
  Expected<std::optional<ondisk::ObjectHandle>> ObjOpt = CAS.load(ID);
  if (!ObjOpt)
    return reportError(ObjOpt.takeError(), error, LLCAS_LOOKUP_RESULT_ERROR);
  if (!*ObjOpt)
    return LLCAS_LOOKUP_RESULT_NOTFOUND;

  ondisk::ObjectHandle Obj = **ObjOpt;
  *c_obj_p = llcas_loaded_object_t{Obj.getOpaqueData()};
  return LLCAS_LOOKUP_RESULT_SUCCESS;
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
llcas_actioncache_get_for_digest(llcas_cas_t c_cas, llcas_digest_t key,
                                 llcas_objectid_t *p_value, char **error) {
  auto &DB = *unwrap(c_cas)->DB;
  Expected<std::optional<ObjectID>> Value =
      DB.KVGet(ArrayRef(key.data, key.size));
  if (!Value)
    return reportError(Value.takeError(), error, LLCAS_LOOKUP_RESULT_ERROR);
  if (!*Value)
    return LLCAS_LOOKUP_RESULT_NOTFOUND;
  *p_value = llcas_objectid_t{(*Value)->getOpaqueData()};
  return LLCAS_LOOKUP_RESULT_SUCCESS;
}

bool llcas_actioncache_put_for_digest(llcas_cas_t c_cas, llcas_digest_t key,
                                      llcas_objectid_t c_value, char **error) {
  auto &DB = *unwrap(c_cas)->DB;
  ObjectID Value = ObjectID::fromOpaqueData(c_value.opaque);
  Expected<ObjectID> Ret = DB.KVPut(ArrayRef(key.data, key.size), Value);
  if (!Ret)
    return reportError(Ret.takeError(), error, true);
  if (*Ret != Value)
    return reportError(
        createStringError(errc::invalid_argument, "cache poisoned"), error,
        true);
  return false;
}
