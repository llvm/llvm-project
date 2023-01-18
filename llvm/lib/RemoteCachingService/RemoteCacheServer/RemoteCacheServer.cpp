//===-- RemoteCacheServer.cpp - gRPC Server for Remote Caching Protocol ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/RemoteCachingService/RemoteCacheServer.h"
#include "llvm/RemoteCachingService/RemoteCacheProvider.h"

#include "compilation_caching_cas.grpc.pb.h"
#include "compilation_caching_kv.grpc.pb.h"
#include <grpcpp/grpcpp.h>

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::cas::remote;
using namespace compilation_cache_service::cas::v1;
using namespace compilation_cache_service::keyvalue::v1;

using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerCompletionQueue;
using grpc::ServerContext;
using grpc::Status;

void RemoteCacheProvider::anchor() {}

static GetValueResponse GetValueWithError(Error &&E) {
  GetValueResponse Response;
  Response.set_outcome(GetValueResponse_Outcome_ERROR);
  Response.mutable_error()->set_description(toString(std::move(E)));
  return Response;
}

static PutValueResponse PutValueWithError(Error &&E) {
  PutValueResponse Response;
  Response.mutable_error()->set_description(toString(std::move(E)));
  return Response;
}

static CASLoadResponse CASLoadWithError(Error &&E) {
  CASLoadResponse Response;
  Response.set_outcome(CASLoadResponse_Outcome_ERROR);
  Response.mutable_error()->set_description(toString(std::move(E)));
  return Response;
}

static CASSaveResponse CASSaveWithError(Error &&E) {
  CASSaveResponse Response;
  Response.mutable_error()->set_description(toString(std::move(E)));
  return Response;
}

static CASGetResponse CASGetWithError(Error &&E) {
  CASGetResponse Response;
  Response.set_outcome(CASGetResponse_Outcome_ERROR);
  Response.mutable_error()->set_description(toString(std::move(E)));
  return Response;
}

static CASPutResponse CASPutWithError(Error &&E) {
  CASPutResponse Response;
  Response.mutable_error()->set_description(toString(std::move(E)));
  return Response;
}

static void
GetValueAdapter(const GetValueRequest &Request, RemoteCacheProvider &Provider,
                std::function<void(const GetValueResponse &)> Receiver) {
  Provider.GetValueAsync(
      Request.key(), [Receiver = std::move(Receiver)](
                         Expected<std::optional<std::string>> Response) {
        if (!Response)
          return Receiver(GetValueWithError(Response.takeError()));

        GetValueResponse grpcResponse;
        if (*Response) {
          StringRef Value = **Response;
          grpcResponse.mutable_value()->ParseFromArray(Value.data(),
                                                       Value.size());
        } else {
          grpcResponse.set_outcome(GetValueResponse_Outcome_KEY_NOT_FOUND);
        }
        return Receiver(grpcResponse);
      });
}

static void
PutValueAdapter(const PutValueRequest &Request, RemoteCacheProvider &Provider,
                std::function<void(const PutValueResponse &)> Receiver) {
  Provider.PutValueAsync(Request.key(), Request.value().SerializeAsString(),
                         [Receiver = std::move(Receiver)](Error E) {
                           if (E)
                             return Receiver(PutValueWithError(std::move(E)));
                           return Receiver(PutValueResponse());
                         });
}

static void
CASLoadAdapter(const CASLoadRequest &Request, RemoteCacheProvider &Provider,
               std::function<void(const CASLoadResponse &)> Receiver) {
  Provider.CASLoadAsync(
      Request.cas_id().id(), Request.write_to_disk(),
      [Receiver = std::move(Receiver)](
          Expected<RemoteCacheProvider::LoadResponse> Response) {
        if (!Response)
          return Receiver(CASLoadWithError(Response.takeError()));

        CASLoadResponse grpcResponse;
        grpcResponse.set_outcome(CASLoadResponse_Outcome_SUCCESS);
        if (Response->Blob.IsFilePath) {
          grpcResponse.mutable_data()->mutable_blob()->set_file_path(
              std::move(Response->Blob.DataOrPath));
        } else {
          grpcResponse.mutable_data()->mutable_blob()->set_data(
              std::move(Response->Blob.DataOrPath));
        }
        return Receiver(grpcResponse);
      });
}

static void
CASSaveAdapter(const CASSaveRequest &Request, RemoteCacheProvider &Provider,
               std::function<void(const CASSaveResponse &)> Receiver) {
  const CASBytes &grpcBlob = Request.data().blob();
  RemoteCacheProvider::BlobContents Blob{
      grpcBlob.has_file_path(),
      grpcBlob.has_file_path() ? grpcBlob.file_path() : grpcBlob.data()};

  Provider.CASSaveAsync(std::move(Blob), [Receiver = std::move(Receiver)](
                                             Expected<std::string> ID) {
    if (!ID)
      return Receiver(CASSaveWithError(ID.takeError()));
    CASSaveResponse grpcResponse;
    grpcResponse.mutable_cas_id()->set_id(std::move(*ID));
    return Receiver(grpcResponse);
  });
}

static void
CASGetAdapter(const CASGetRequest &Request, RemoteCacheProvider &Provider,
              std::function<void(const CASGetResponse &)> Receiver) {
  Provider.CASGetAsync(
      Request.cas_id().id(), Request.write_to_disk(),
      [Receiver = std::move(Receiver)](
          Expected<RemoteCacheProvider::GetResponse> Response) {
        if (!Response)
          return Receiver(CASGetWithError(Response.takeError()));

        CASGetResponse grpcResponse;
        grpcResponse.set_outcome(CASGetResponse_Outcome_SUCCESS);
        if (Response->Blob.IsFilePath) {
          grpcResponse.mutable_data()->mutable_blob()->set_file_path(
              std::move(Response->Blob.DataOrPath));
        } else {
          grpcResponse.mutable_data()->mutable_blob()->set_data(
              std::move(Response->Blob.DataOrPath));
        }
        for (std::string &Ref : Response->Refs) {
          grpcResponse.mutable_data()->add_references()->set_id(std::move(Ref));
        }
        return Receiver(grpcResponse);
      });
}

static void
CASPutAdapter(const CASPutRequest &Request, RemoteCacheProvider &Provider,
              std::function<void(const CASPutResponse &)> Receiver) {
  const CASBytes &grpcBlob = Request.data().blob();
  RemoteCacheProvider::BlobContents Blob{
      grpcBlob.has_file_path(),
      grpcBlob.has_file_path() ? grpcBlob.file_path() : grpcBlob.data()};
  SmallVector<std::string> Refs;
  Refs.reserve(Request.data().references().size());
  for (auto &Ref : Request.data().references())
    Refs.push_back(Ref.id());

  Provider.CASPutAsync(
      std::move(Blob), std::move(Refs),
      [Receiver = std::move(Receiver)](Expected<std::string> ID) {
        if (!ID)
          return Receiver(CASPutWithError(ID.takeError()));
        CASPutResponse grpcResponse;
        grpcResponse.mutable_cas_id()->set_id(std::move(*ID));
        return Receiver(grpcResponse);
      });
}

/// A gRPC server implementation for the remote cache service protocol. The
/// actual work of storing and retrieving data is done via the provided
/// \p RemoteCacheProvider.
///
/// The server is implemented using asynchronous facilities so execution of a
/// request is not blocking receiving additional requests.
class RemoteCacheServer::Implementation final {
public:
  Implementation(StringRef SocketPath,
                 std::unique_ptr<RemoteCacheProvider> CacheProvider)
      : SocketPath(SocketPath), CacheProvider(std::move(CacheProvider)) {}

  ~Implementation() { Shutdown(); }

  void Start() {
    std::string Address("unix:");
    Address += SocketPath;

    ServerBuilder Builder;
    // Listen on the given address without any authentication mechanism.
    Builder.AddListeningPort(Address, grpc::InsecureServerCredentials());
    Builder.RegisterService(&KVService);
    Builder.RegisterService(&CASService);
    CQ = Builder.AddCompletionQueue();
    // Finally assemble the server.
    Server = Builder.BuildAndStart();
  }

  void Listen() {
    // Proceed to the server's main loop.
    HandleRpcs();
  }

  void Shutdown() {
    Server->Shutdown();
    CQ->Shutdown();
  }

private:
  class CallData {
  public:
    virtual ~CallData() = default;
    virtual void proceed(bool ok) = 0;
  };

  template <typename T> struct RequestTraits {};

  template <> struct RequestTraits<GetValueRequest> {
    using ResponseT = GetValueResponse;
    using ServiceT = KeyValueDB::AsyncService;

    static constexpr auto ProviderFunc = GetValueAdapter;
    static constexpr auto ServiceRequest = &ServiceT::RequestGetValue;
  };

  template <> struct RequestTraits<PutValueRequest> {
    using ResponseT = PutValueResponse;
    using ServiceT = KeyValueDB::AsyncService;

    static constexpr auto ProviderFunc = PutValueAdapter;
    static constexpr auto ServiceRequest = &ServiceT::RequestPutValue;
  };

  template <> struct RequestTraits<CASLoadRequest> {
    using ResponseT = CASLoadResponse;
    using ServiceT = CASDBService::AsyncService;

    static constexpr auto ProviderFunc = CASLoadAdapter;
    static constexpr auto ServiceRequest = &ServiceT::RequestLoad;
  };

  template <> struct RequestTraits<CASSaveRequest> {
    using ResponseT = CASSaveResponse;
    using ServiceT = CASDBService::AsyncService;

    static constexpr auto ProviderFunc = CASSaveAdapter;
    static constexpr auto ServiceRequest = &ServiceT::RequestSave;
  };

  template <> struct RequestTraits<CASGetRequest> {
    using ResponseT = CASGetResponse;
    using ServiceT = CASDBService::AsyncService;

    static constexpr auto ProviderFunc = CASGetAdapter;
    static constexpr auto ServiceRequest = &ServiceT::RequestGet;
  };

  template <> struct RequestTraits<CASPutRequest> {
    using ResponseT = CASPutResponse;
    using ServiceT = CASDBService::AsyncService;

    static constexpr auto ProviderFunc = CASPutAdapter;
    static constexpr auto ServiceRequest = &ServiceT::RequestPut;
  };

  // Class encompasing the state and logic needed to serve a request.
  template <typename RequestT> class RequestHandler final : public CallData {
  public:
    using ResponseT = typename RequestTraits<RequestT>::ResponseT;
    using ServiceT = typename RequestTraits<RequestT>::ServiceT;

    RequestHandler(ServiceT *service, ServerCompletionQueue *cq,
                   RemoteCacheProvider *provider)
        : Service(service), CQ(cq), Provider(provider), Responder(&Ctx),
          Status(CallStatus::Create) {
      // Invoke the serving logic right away.
      proceed(true);
    }

  private:
    void proceed(bool ok) override {
      if (!ok) {
        Status = CallStatus::Finish;
      }

      switch (Status) {
      case CallStatus::Create:
        Status = CallStatus::Process;

        // As part of the initial \p CallStatus::Create state, we *request* that
        // the system starts processing \p RequestT requests. In this request,
        // "this" is the tag uniquely identifying the request (so that different
        // \p CallData instances can serve different requests concurrently), in
        // this case the memory address of this \p CallData instance.
        (Service->*RequestTraits<RequestT>::ServiceRequest)(
            &Ctx, &Request, &Responder, CQ, CQ, this);
        break;

      case CallStatus::Process:
        // Spawn a new \p RequestHandler instance to serve new clients while we
        // process the one for this request. The instance will deallocate itself
        // as part of its \p CallStatus::Finish state.
        new RequestHandler(Service, CQ, Provider);

        // The actual processing.
        Status = CallStatus::Finish;
        RequestTraits<RequestT>::ProviderFunc(
            Request, *Provider, [this](const ResponseT &Response) {
              Responder.Finish(Response, grpc::Status::OK, this);
            });
        break;

      case CallStatus::Finish:
        // Once in the \p CallStatus::Finish state, deallocate ourselves.
        delete this;
      }
    }

    // The means of communication with the gRPC runtime for an asynchronous
    // server.
    ServiceT *Service;
    // The producer-consumer queue for asynchronous server notifications.
    ServerCompletionQueue *CQ;
    RemoteCacheProvider *Provider;

    ServerContext Ctx;
    // What we get from the client.
    RequestT Request;
    // The means to respond back to the client.
    ServerAsyncResponseWriter<ResponseT> Responder;

    // A state machine for processing a request.
    enum class CallStatus { Create, Process, Finish };
    CallStatus Status; // The current serving state.
  };

  // This can be run in multiple threads if needed.
  void HandleRpcs() {
    // Spawn new \p RequestHandler instances to serve new clients. These will
    // get deallocated when they reach \p \p CallStatus::Finish state.
    new RequestHandler<GetValueRequest>(&KVService, CQ.get(),
                                        CacheProvider.get());
    new RequestHandler<PutValueRequest>(&KVService, CQ.get(),
                                        CacheProvider.get());
    new RequestHandler<CASLoadRequest>(&CASService, CQ.get(),
                                       CacheProvider.get());
    new RequestHandler<CASSaveRequest>(&CASService, CQ.get(),
                                       CacheProvider.get());
    new RequestHandler<CASGetRequest>(&CASService, CQ.get(),
                                      CacheProvider.get());
    new RequestHandler<CASPutRequest>(&CASService, CQ.get(),
                                      CacheProvider.get());

    void *tag; // uniquely identifies a request.
    bool ok;
    while (true) {
      // Block waiting to read the next event from the completion queue. The
      // event is uniquely identified by its tag, which in this case is the
      // memory address of a \p CallData instance.
      bool gotEvent = CQ->Next(&tag, &ok);
      if (!gotEvent)
        break;
      static_cast<CallData *>(tag)->proceed(ok);
    }
  }

  std::string SocketPath;
  std::unique_ptr<RemoteCacheProvider> CacheProvider;
  std::unique_ptr<ServerCompletionQueue> CQ;
  KeyValueDB::AsyncService KVService;
  CASDBService::AsyncService CASService;
  std::unique_ptr<Server> Server;
};

RemoteCacheServer::~RemoteCacheServer() = default;

void RemoteCacheServer::Start() { return Impl->Start(); }
void RemoteCacheServer::Listen() { return Impl->Listen(); }
void RemoteCacheServer::Shutdown() { return Impl->Shutdown(); }

RemoteCacheServer::RemoteCacheServer(
    std::unique_ptr<RemoteCacheServer::Implementation> Impl)
    : Impl(std::move(Impl)) {}

RemoteCacheServer::RemoteCacheServer(
    StringRef SocketPath, std::unique_ptr<RemoteCacheProvider> CacheProvider)
    : RemoteCacheServer(std::make_unique<RemoteCacheServer::Implementation>(
          SocketPath, std::move(CacheProvider))) {}
