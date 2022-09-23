//===-- RemoteCacheServer.cpp - gRPC Server for Remote Caching Protocol ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RemoteCacheServer.h"
#include "RemoteCacheProvider.h"
#include <grpcpp/grpcpp.h>

using namespace llvm;
using namespace remote_cache_test;
using namespace compilation_cache_service::cas::v1;
using namespace compilation_cache_service::keyvalue::v1;

using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerCompletionQueue;
using grpc::ServerContext;
using grpc::Status;

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

  void Run() {
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

    static constexpr auto ProviderFunc = &RemoteCacheProvider::GetValueAsync;
    static constexpr auto ServiceRequest = &ServiceT::RequestGetValue;
  };

  template <> struct RequestTraits<PutValueRequest> {
    using ResponseT = PutValueResponse;
    using ServiceT = KeyValueDB::AsyncService;

    static constexpr auto ProviderFunc = &RemoteCacheProvider::PutValueAsync;
    static constexpr auto ServiceRequest = &ServiceT::RequestPutValue;
  };

  template <> struct RequestTraits<CASLoadRequest> {
    using ResponseT = CASLoadResponse;
    using ServiceT = CASDBService::AsyncService;

    static constexpr auto ProviderFunc = &RemoteCacheProvider::CASLoadAsync;
    static constexpr auto ServiceRequest = &ServiceT::RequestLoad;
  };

  template <> struct RequestTraits<CASSaveRequest> {
    using ResponseT = CASSaveResponse;
    using ServiceT = CASDBService::AsyncService;

    static constexpr auto ProviderFunc = &RemoteCacheProvider::CASSaveAsync;
    static constexpr auto ServiceRequest = &ServiceT::RequestSave;
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
        (Provider->*RequestTraits<RequestT>::ProviderFunc)(
            Request, [this](const ResponseT &Response) {
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

void RemoteCacheServer::Run() { return Impl->Run(); }

void RemoteCacheServer::Shutdown() { return Impl->Shutdown(); }

RemoteCacheServer::RemoteCacheServer(
    std::unique_ptr<RemoteCacheServer::Implementation> Impl)
    : Impl(std::move(Impl)) {}

RemoteCacheServer remote_cache_test::createServer(
    StringRef SocketPath, std::unique_ptr<RemoteCacheProvider> CacheProvider) {
  return RemoteCacheServer(std::make_unique<RemoteCacheServer::Implementation>(
      SocketPath, std::move(CacheProvider)));
}
