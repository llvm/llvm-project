//===-- Client.cpp - Remote Client ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/RemoteCachingService/Client.h"
#include "compilation_caching_cas.grpc.pb.h"
#include "compilation_caching_kv.grpc.pb.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include <grpcpp/grpcpp.h>
#include <memory>

using namespace compilation_cache_service::cas::v1;
using namespace compilation_cache_service::keyvalue::v1;
using namespace llvm;
using namespace llvm::cas;
using namespace llvm::cas::remote;

void AsyncCallerContext::anchor() {}
void AsyncQueueBase::anchor() {}

void KeyValueDBClient::GetValueAsyncQueue::anchor() {}
void KeyValueDBClient::PutValueAsyncQueue::anchor() {}
void KeyValueDBClient::anchor() {}

void CASDBClient::LoadAsyncQueue::anchor() {}
void CASDBClient::SaveAsyncQueue::anchor() {}
void CASDBClient::GetAsyncQueue::anchor() {}
void CASDBClient::PutAsyncQueue::anchor() {}
void CASDBClient::anchor() {}

static Error errorFromGRPCStatus(const grpc::Status &Status) {
  return createStringError(inconvertibleErrorCode(), Status.error_message());
}

static Expected<std::optional<KeyValueDBClient::ValueTy>>
createGetValueResponse(const GetValueResponse &Resp) {
  if (Resp.has_error())
    return createStringError(inconvertibleErrorCode(),
                             Resp.error().description());
  if (!Resp.has_value())
    return std::nullopt;

  KeyValueDBClient::ValueTy Ret;
  const Value &V = Resp.value();
  for (const auto &Entry : V.entries()) {
    Ret[Entry.first] = Entry.second;
  }
  return Ret;
}

static Expected<CASDBClient::LoadResponse>
createLoadResponse(const CASLoadResponse &Response,
                   std::optional<std::string> OutFilePath) {
  CASDBClient::LoadResponse Resp;
  switch (Response.outcome()) {
  case CASLoadResponse_Outcome_OBJECT_NOT_FOUND:
    Resp.KeyNotFound = true;
    return Resp;
  case CASLoadResponse_Outcome_ERROR:
    return createStringError(inconvertibleErrorCode(),
                             Response.error().description());
  default:
    break;
  }
  assert(Response.outcome() == CASLoadResponse_Outcome_SUCCESS);
  const auto &Blob = Response.data().blob();
  if (OutFilePath.has_value()) {
    if (Blob.has_file_path()) {
      if (std::error_code EC = sys::fs::rename(Blob.file_path(), *OutFilePath))
        return createStringError(EC, "failed rename '" + Blob.file_path() +
                                         "' to '" + *OutFilePath +
                                         "': " + EC.message());
    } else {
      std::error_code EC;
      raw_fd_ostream OS(*OutFilePath, EC);
      if (EC)
        return createStringError(EC, "failed creating ''" + *OutFilePath +
                                         "': " + EC.message());
      OS << Blob.data();
      OS.close();
    }
  } else {
    if (Blob.has_file_path()) {
      ErrorOr<std::unique_ptr<MemoryBuffer>> FileBuf =
          MemoryBuffer::getFile(Blob.file_path());
      if (!FileBuf)
        return createStringError(FileBuf.getError(),
                                 "failed reading '" + Blob.file_path() +
                                     "': " + FileBuf.getError().message());
      Resp.BlobData = (*FileBuf)->getBuffer().str();
    } else {
      Resp.BlobData = Blob.data();
    }
  }
  return Resp;
}

static Expected<CASDBClient::GetResponse>
createGetResponse(const CASGetResponse &Response,
                  std::optional<std::string> OutFilePath) {
  CASDBClient::GetResponse Resp;
  switch (Response.outcome()) {
  case CASGetResponse_Outcome_OBJECT_NOT_FOUND:
    Resp.KeyNotFound = true;
    return Resp;
  case CASGetResponse_Outcome_ERROR:
    return createStringError(inconvertibleErrorCode(),
                             Response.error().description());
  default:
    break;
  }
  assert(Response.outcome() == CASGetResponse_Outcome_SUCCESS);
  const auto &Blob = Response.data().blob();
  if (OutFilePath.has_value()) {
    if (Blob.has_file_path()) {
      if (std::error_code EC = sys::fs::rename(Blob.file_path(), *OutFilePath))
        return createStringError(EC, "failed rename '" + Blob.file_path() +
                                         "' to '" + *OutFilePath +
                                         "': " + EC.message());
    } else {
      std::error_code EC;
      raw_fd_ostream OS(*OutFilePath, EC);
      if (EC)
        return createStringError(EC, "failed creating ''" + *OutFilePath +
                                         "': " + EC.message());
      OS << Blob.data();
      OS.close();
    }
  } else {
    if (Blob.has_file_path()) {
      ErrorOr<std::unique_ptr<MemoryBuffer>> FileBuf =
          MemoryBuffer::getFile(Blob.file_path());
      if (!FileBuf)
        return createStringError(FileBuf.getError(),
                                 "failed reading '" + Blob.file_path() +
                                     "': " + FileBuf.getError().message());
      Resp.BlobData = (*FileBuf)->getBuffer().str();
    } else {
      Resp.BlobData = Blob.data();
    }
  }
  for (const CASDataID &ID : Response.data().references())
    Resp.Refs.emplace_back(ID.id());

  return Resp;
}

namespace {

template <typename ResponseT> struct AsyncClientCall {
  std::shared_ptr<AsyncCallerContext> CallCtx;

  // Container for the data we expect from the server.
  ResponseT Response;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  grpc::ClientContext Context;

  // Storage for the status of the RPC upon completion.
  grpc::Status Status;

  std::unique_ptr<grpc::ClientAsyncResponseReader<ResponseT>> Reader;
};

class KVGetAsyncQueueImpl : public KeyValueDBClient::GetValueAsyncQueue {
  KeyValueDB::Stub &Stub;
  grpc::CompletionQueue CQ;

public:
  KVGetAsyncQueueImpl(KeyValueDB::Stub &Stub) : Stub(Stub) {}

  void getValueAsyncImpl(std::string Key,
                         std::shared_ptr<AsyncCallerContext> CallCtx) override {
    GetValueRequest Request;
    Request.set_key(std::move(Key));

    auto *call = new AsyncClientCall<GetValueResponse>();
    call->CallCtx = std::move(CallCtx);
    call->Reader = Stub.AsyncGetValue(&call->Context, Request, &CQ);
    call->Reader->Finish(&call->Response, &call->Status, call);
  }

  Expected<Response> receiveNextImpl() override {
    void *got_tag;
    bool ok = false;

    // Block until the next result is available in the completion queue.
    CQ.Next(&got_tag, &ok);
    // The tag in this example is the memory location of the call object
    auto *call = static_cast<AsyncClientCall<GetValueResponse> *>(got_tag);
    auto _ = llvm::make_scope_exit([&]() { delete call; });

    if (!call->Status.ok())
      return errorFromGRPCStatus(call->Status);
    if (!ok)
      return createStringError(inconvertibleErrorCode(),
                               "service channel shutdown");

    auto Value = createGetValueResponse(call->Response);
    if (!Value)
      return Value.takeError();
    Response Resp;
    Resp.CallCtx = std::move(call->CallCtx);
    Resp.Value = std::move(*Value);
    return Resp;
  }
};

class KVPutAsyncQueueImpl : public KeyValueDBClient::PutValueAsyncQueue {
  KeyValueDB::Stub &Stub;
  grpc::CompletionQueue CQ;

public:
  KVPutAsyncQueueImpl(KeyValueDB::Stub &Stub) : Stub(Stub) {}

  void putValueAsyncImpl(std::string Key,
                         const KeyValueDBClient::ValueTy &Value,
                         std::shared_ptr<AsyncCallerContext> CallCtx) override {
    PutValueRequest Request;
    Request.set_key(std::move(Key));
    auto &PBMap = *Request.mutable_value()->mutable_entries();
    for (const auto &Entry : Value) {
      PBMap[Entry.first().str()] = Entry.second;
    }

    auto *call = new AsyncClientCall<PutValueResponse>();
    call->CallCtx = std::move(CallCtx);
    call->Reader = Stub.AsyncPutValue(&call->Context, Request, &CQ);
    call->Reader->Finish(&call->Response, &call->Status, call);
  }

  Expected<Response> receiveNextImpl() override {
    void *got_tag;
    bool ok = false;

    // Block until the next result is available in the completion queue.
    CQ.Next(&got_tag, &ok);
    // The tag in this example is the memory location of the call object
    auto *call = static_cast<AsyncClientCall<PutValueResponse> *>(got_tag);
    auto _ = llvm::make_scope_exit([&]() { delete call; });

    if (!call->Status.ok())
      return errorFromGRPCStatus(call->Status);
    if (!ok)
      return createStringError(inconvertibleErrorCode(),
                               "service channel shutdown");

    Response Resp;
    Resp.CallCtx = std::move(call->CallCtx);
    if (call->Response.has_error())
      return createStringError(inconvertibleErrorCode(),
                               call->Response.error().description());
    return Resp;
  }
};

struct CASLoadBlobClientCall : public AsyncClientCall<CASLoadResponse> {
  std::optional<std::string> OutFilePath;
};

class CASLoadAsyncQueueImpl : public CASDBClient::LoadAsyncQueue {
  CASDBService::Stub &Stub;
  grpc::CompletionQueue CQ;

public:
  CASLoadAsyncQueueImpl(CASDBService::Stub &Stub) : Stub(Stub) {}

  void loadAsyncImpl(std::string CASID, std::optional<std::string> OutFilePath,
                     std::shared_ptr<AsyncCallerContext> CallCtx) override {
    assert(!OutFilePath || !OutFilePath->empty());
    CASLoadRequest Request;
    Request.mutable_cas_id()->set_id(std::move(CASID));
    Request.set_write_to_disk(OutFilePath.has_value());

    auto *call = new CASLoadBlobClientCall();
    call->OutFilePath = std::move(OutFilePath);
    call->CallCtx = std::move(CallCtx);
    call->Reader = Stub.AsyncLoad(&call->Context, Request, &CQ);
    call->Reader->Finish(&call->Response, &call->Status, call);
  }

  Expected<Response> receiveNextImpl() override {
    void *got_tag;
    bool ok = false;

    // Block until the next result is available in the completion queue.
    CQ.Next(&got_tag, &ok);
    // The tag in this example is the memory location of the call object
    auto *call = static_cast<CASLoadBlobClientCall *>(got_tag);
    auto _ = llvm::make_scope_exit([&]() { delete call; });

    if (!call->Status.ok())
      return errorFromGRPCStatus(call->Status);
    if (!ok)
      return createStringError(inconvertibleErrorCode(),
                               "service channel shutdown");

    Response Resp;
    Resp.CallCtx = std::move(call->CallCtx);

    auto Value = createLoadResponse(call->Response, call->OutFilePath);
    if (!Value)
      return Value.takeError();
    Resp.KeyNotFound = Value->KeyNotFound;
    Resp.BlobData = std::move(Value->BlobData);
    return Resp;
  }
};

class CASSaveAsyncQueueImpl : public CASDBClient::SaveAsyncQueue {
  CASDBService::Stub &Stub;
  grpc::CompletionQueue CQ;

public:
  CASSaveAsyncQueueImpl(CASDBService::Stub &Stub) : Stub(Stub) {}

  void saveDataAsyncImpl(std::string BlobData,
                         std::shared_ptr<AsyncCallerContext> CallCtx) override {
    CASSaveRequest Request;
    Request.mutable_data()->mutable_blob()->set_data(std::move(BlobData));
    casSaveAsync(Request, std::move(CallCtx));
  }

  void saveFileAsyncImpl(std::string FilePath,
                         std::shared_ptr<AsyncCallerContext> CallCtx) override {
    assert(!FilePath.empty());
    CASSaveRequest Request;
    Request.mutable_data()->mutable_blob()->set_file_path(std::move(FilePath));
    casSaveAsync(Request, std::move(CallCtx));
  }

  void casSaveAsync(const CASSaveRequest &Request,
                    std::shared_ptr<AsyncCallerContext> CallCtx) {
    auto *call = new AsyncClientCall<CASSaveResponse>();
    call->CallCtx = std::move(CallCtx);
    call->Reader = Stub.AsyncSave(&call->Context, Request, &CQ);
    call->Reader->Finish(&call->Response, &call->Status, call);
  }

  Expected<Response> receiveNextImpl() override {
    void *got_tag;
    bool ok = false;

    // Block until the next result is available in the completion queue.
    CQ.Next(&got_tag, &ok);
    // The tag in this example is the memory location of the call object
    auto *call = static_cast<AsyncClientCall<CASSaveResponse> *>(got_tag);
    auto _ = llvm::make_scope_exit([&]() { delete call; });

    if (!call->Status.ok())
      return errorFromGRPCStatus(call->Status);
    if (!ok)
      return createStringError(inconvertibleErrorCode(),
                               "service channel shutdown");

    Response Resp;
    Resp.CallCtx = std::move(call->CallCtx);
    if (call->Response.has_error())
      return createStringError(inconvertibleErrorCode(),
                               call->Response.error().description());
    Resp.CASID = call->Response.cas_id().id();
    return Resp;
  }
};

struct CASGetBlobClientCall : public AsyncClientCall<CASGetResponse> {
  std::optional<std::string> OutFilePath;
};

class CASGetAsyncQueueImpl : public CASDBClient::GetAsyncQueue {
  CASDBService::Stub &Stub;
  grpc::CompletionQueue CQ;

public:
  CASGetAsyncQueueImpl(CASDBService::Stub &Stub) : Stub(Stub) {}

  void getAsyncImpl(std::string CASID, std::optional<std::string> OutFilePath,
                    std::shared_ptr<AsyncCallerContext> CallCtx) override {
    assert(!OutFilePath || !OutFilePath->empty());
    CASGetRequest Request;
    Request.mutable_cas_id()->set_id(std::move(CASID));
    Request.set_write_to_disk(OutFilePath.has_value());

    auto *call = new CASGetBlobClientCall();
    call->OutFilePath = std::move(OutFilePath);
    call->CallCtx = std::move(CallCtx);
    call->Reader = Stub.AsyncGet(&call->Context, Request, &CQ);
    call->Reader->Finish(&call->Response, &call->Status, call);
  }

  Expected<Response> receiveNextImpl() override {
    void *got_tag;
    bool ok = false;

    // Block until the next result is available in the completion queue.
    CQ.Next(&got_tag, &ok);
    // The tag in this example is the memory location of the call object
    auto *call = static_cast<CASGetBlobClientCall *>(got_tag);
    auto _ = llvm::make_scope_exit([&]() { delete call; });

    if (!call->Status.ok())
      return errorFromGRPCStatus(call->Status);
    if (!ok)
      return createStringError(inconvertibleErrorCode(),
                               "service channel shutdown");

    Response Resp;
    Resp.CallCtx = std::move(call->CallCtx);
    auto Value = createGetResponse(call->Response, call->OutFilePath);
    if (!Value)
      return Value.takeError();

    Resp.KeyNotFound = Value->KeyNotFound;
    Resp.BlobData = std::move(Value->BlobData);
    Resp.Refs = std::move(Value->Refs);
    return Resp;
  }
};

class CASPutAsyncQueueImpl : public CASDBClient::PutAsyncQueue {
  CASDBService::Stub &Stub;
  grpc::CompletionQueue CQ;

public:
  CASPutAsyncQueueImpl(CASDBService::Stub &Stub) : Stub(Stub) {}

  void putDataAsyncImpl(std::string BlobData, ArrayRef<std::string> Refs,
                        std::shared_ptr<AsyncCallerContext> CallCtx) override {
    CASPutRequest Request;
    Request.mutable_data()->mutable_blob()->set_data(std::move(BlobData));
    for (auto &Ref : Refs) {
      CASDataID *NewRef = Request.mutable_data()->add_references();
      NewRef->set_id(Ref);
    }
    casPutAsync(Request, std::move(CallCtx));
  }

  void putFileAsyncImpl(std::string FilePath, ArrayRef<std::string> Refs,
                        std::shared_ptr<AsyncCallerContext> CallCtx) override {
    assert(!FilePath.empty());
    CASPutRequest Request;
    Request.mutable_data()->mutable_blob()->set_file_path(std::move(FilePath));
    for (auto &Ref : Refs) {
      CASDataID *NewRef = Request.mutable_data()->add_references();
      NewRef->set_id(Ref);
    }
    casPutAsync(Request, std::move(CallCtx));
  }

  void casPutAsync(const CASPutRequest &Request,
                   std::shared_ptr<AsyncCallerContext> CallCtx) {
    auto *call = new AsyncClientCall<CASPutResponse>();
    call->CallCtx = std::move(CallCtx);
    call->Reader = Stub.AsyncPut(&call->Context, Request, &CQ);
    call->Reader->Finish(&call->Response, &call->Status, call);
  }

  Expected<Response> receiveNextImpl() override {
    void *got_tag;
    bool ok = false;

    // Block until the next result is available in the completion queue.
    CQ.Next(&got_tag, &ok);
    // The tag in this example is the memory location of the call object
    auto *call = static_cast<AsyncClientCall<CASPutResponse> *>(got_tag);
    auto _ = llvm::make_scope_exit([&]() { delete call; });

    if (!call->Status.ok())
      return errorFromGRPCStatus(call->Status);
    if (!ok)
      return createStringError(inconvertibleErrorCode(),
                               "service channel shutdown");

    Response Resp;
    Resp.CallCtx = std::move(call->CallCtx);
    if (call->Response.has_error())
      return createStringError(inconvertibleErrorCode(),
                               call->Response.error().description());
    Resp.CASID = call->Response.cas_id().id();
    return Resp;
  }
};

class KeyValueDBClientImpl : public KeyValueDBClient {
  std::unique_ptr<KeyValueDB::Stub> Stub;

  Expected<std::optional<KeyValueDBClient::ValueTy>>
  getValueSyncImpl(std::string Key) override {
    GetValueRequest Request;
    grpc::ClientContext Context;
    GetValueResponse Resp;
    Request.set_key(Key);
    grpc::Status Status = Stub->GetValue(&Context, Request, &Resp);
    if (!Status.ok())
      return errorFromGRPCStatus(Status);

    return createGetValueResponse(Resp);
  }

  Error putValueSyncImpl(std::string Key,
                         const KeyValueDBClient::ValueTy &Value) override {
    PutValueRequest Request;
    grpc::ClientContext Context;
    PutValueResponse Resp;
    Request.set_key(std::move(Key));
    auto &PBMap = *Request.mutable_value()->mutable_entries();
    for (const auto &Entry : Value) {
      PBMap[Entry.first().str()] = Entry.second;
    }
    grpc::Status Status = Stub->PutValue(&Context, Request, &Resp);
    if (!Status.ok())
      return errorFromGRPCStatus(Status);

    if (Resp.has_error())
      return createStringError(inconvertibleErrorCode(),
                               Resp.error().description());

    return Error::success();
  }

public:
  KeyValueDBClientImpl(std::shared_ptr<grpc::Channel> Channel)
      : Stub(KeyValueDB::NewStub(std::move(Channel))) {
    GetValueQueue = std::make_unique<KVGetAsyncQueueImpl>(*Stub);
    PutValueQueue = std::make_unique<KVPutAsyncQueueImpl>(*Stub);
  }
};

class CASDBClientImpl : public CASDBClient {
  std::unique_ptr<CASDBService::Stub> Stub;

  Expected<CASDBClient::LoadResponse>
  loadSyncImpl(std::string CASID,
               std::optional<std::string> OutFilePath) override {
    CASLoadRequest Request;
    grpc::ClientContext Context;
    CASLoadResponse Response;
    Request.mutable_cas_id()->set_id(std::move(CASID));
    Request.set_write_to_disk(OutFilePath.has_value());
    grpc::Status Status = Stub->Load(&Context, Request, &Response);
    if (!Status.ok())
      return errorFromGRPCStatus(Status);
    return createLoadResponse(Response, OutFilePath);
  }

  Expected<std::string> saveDataSyncImpl(std::string BlobData) override {
    CASSaveRequest Request;
    Request.mutable_data()->mutable_blob()->set_data(std::move(BlobData));
    return casSaveSync(Request);
  }

  Expected<std::string> saveFileSyncImpl(std::string FilePath) override {
    assert(!FilePath.empty());
    CASSaveRequest Request;
    Request.mutable_data()->mutable_blob()->set_file_path(std::move(FilePath));
    return casSaveSync(Request);
  }

  Expected<std::string> casSaveSync(const CASSaveRequest &Request) {
    grpc::ClientContext Context;
    CASSaveResponse Response;
    grpc::Status Status = Stub->Save(&Context, Request, &Response);
    if (!Status.ok())
      return errorFromGRPCStatus(Status);
    if (Response.has_error())
      return createStringError(inconvertibleErrorCode(),
                               Response.error().description());
    return Response.cas_id().id();
  }

  Expected<CASDBClient::GetResponse>
  getSyncImpl(std::string CASID,
              std::optional<std::string> OutFilePath) override {
    CASGetRequest Request;
    grpc::ClientContext Context;
    CASGetResponse Response;
    Request.mutable_cas_id()->set_id(std::move(CASID));
    Request.set_write_to_disk(OutFilePath.has_value());
    grpc::Status Status = Stub->Get(&Context, Request, &Response);
    return createGetResponse(Response, OutFilePath);
  }

  Expected<std::string> putDataSyncImpl(std::string BlobData,
                                        ArrayRef<std::string> Refs) override {
    CASPutRequest Request;
    Request.mutable_data()->mutable_blob()->set_data(std::move(BlobData));
    for (auto &Ref : Refs) {
      CASDataID *NewRef = Request.mutable_data()->add_references();
      NewRef->set_id(Ref);
    }
    return casPutSync(Request);
  }

  Expected<std::string> putFileSyncImpl(std::string FilePath,
                                        ArrayRef<std::string> Refs) override {
    assert(!FilePath.empty());
    CASPutRequest Request;
    Request.mutable_data()->mutable_blob()->set_file_path(std::move(FilePath));
    for (auto &Ref : Refs) {
      CASDataID *NewRef = Request.mutable_data()->add_references();
      NewRef->set_id(Ref);
    }
    return casPutSync(Request);
  }

  Expected<std::string> casPutSync(const CASPutRequest &Request) {
    grpc::ClientContext Context;
    CASPutResponse Response;
    grpc::Status Status = Stub->Put(&Context, Request, &Response);
    if (!Status.ok())
      return errorFromGRPCStatus(Status);

    if (Response.has_error())
      return createStringError(inconvertibleErrorCode(),
                               Response.error().description());
    return Response.cas_id().id();
  }

public:
  CASDBClientImpl(std::shared_ptr<grpc::Channel> Channel)
      : Stub(CASDBService::NewStub(std::move(Channel))) {
    LoadQueue = std::make_unique<CASLoadAsyncQueueImpl>(*Stub);
    SaveQueue = std::make_unique<CASSaveAsyncQueueImpl>(*Stub);
    GetQueue = std::make_unique<CASGetAsyncQueueImpl>(*Stub);
    PutQueue = std::make_unique<CASPutAsyncQueueImpl>(*Stub);
  }
};

} // namespace

static std::shared_ptr<grpc::Channel> createGRPCChannel(StringRef SocketPath) {
  std::string Address("unix:");
  Address += SocketPath;
  grpc::ChannelArguments Args{};
  // Remove the initial connection timeout. If the execution environment gets
  // into a state where the processes don't make progress, for a period longer
  // than the timeout, it can cause unnecessary compilation errors if the
  // connection fails due to the timeout. Note that the connection still fails
  // if the socket path does not exist or there is no process listening on it.
  Args.SetInt(GRPC_ARG_MIN_RECONNECT_BACKOFF_MS, INT_MAX);
  Args.SetInt(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, -1);
  return grpc::CreateCustomChannel(std::move(Address),
                                   grpc::InsecureChannelCredentials(), Args);
}

Expected<ClientServices>
cas::remote::createCompilationCachingRemoteClient(StringRef SocketPath) {
  const auto Channel = createGRPCChannel(SocketPath);
  auto KVClient = std::make_unique<KeyValueDBClientImpl>(Channel);
  auto CASClient = std::make_unique<CASDBClientImpl>(std::move(Channel));
  return ClientServices{std::move(KVClient), std::move(CASClient)};
}

Expected<std::unique_ptr<CASDBClient>>
cas::remote::createRemoteCASDBClient(StringRef SocketPath) {
  return std::make_unique<CASDBClientImpl>(createGRPCChannel(SocketPath));
}

Expected<std::unique_ptr<KeyValueDBClient>>
cas::remote::createRemoteKeyValueClient(StringRef SocketPath) {
  return std::make_unique<KeyValueDBClientImpl>(createGRPCChannel(SocketPath));
}
