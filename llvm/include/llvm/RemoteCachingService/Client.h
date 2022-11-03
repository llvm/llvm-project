//===-- llvm/RemoteCachingService/Client.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// gRPC client for the remote cache service protocol. It provides asynchronous
// APIs to allow initiating multiple concurrent requests.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_REMOTECACHINGSERVICE_CLIENT_H
#define LLVM_REMOTECACHINGSERVICE_CLIENT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include <atomic>
#include <memory>
#include <string>

namespace llvm::cas::remote {

/// Used to optionally associate additional context with a particular request.
class AsyncCallerContext {
  virtual void anchor();

public:
  virtual ~AsyncCallerContext() = default;
};

class AsyncQueueBase {
  virtual void anchor();

public:
  virtual ~AsyncQueueBase() = default;

  bool hasPending() const { return NumPending != 0; }

protected:
  std::atomic<unsigned> NumPending = 0;
};

/// An asynchronous gRPC client for the key-value service of the remote cache
/// protocol.
///
/// Example usage:
/// \code
///   // Initiate one (or more) `GetValue` request.
///   KVClient->getValueQueue().getValueAsync(ResultCacheKey.getHash());
///   // Wait for one response.
///   auto Response = KVClient->getValueQueue().receiveNext();
/// \endcode
class KeyValueDBClient {
  virtual void anchor();

public:
  virtual ~KeyValueDBClient() = default;

  using ValueTy = StringMap<std::string>;

  Expected<Optional<ValueTy>> getValueSync(std::string Key) {
    return getValueSyncImpl(std::move(Key));
  }
  Expected<Optional<ValueTy>> getValueSync(ArrayRef<uint8_t> Key) {
    return getValueSync(toStringRef(Key).str());
  }
  Error putValueSync(std::string Key, const ValueTy &Value) {
    return putValueSyncImpl(std::move(Key), Value);
  }
  Error putValueSync(ArrayRef<uint8_t> Key, const ValueTy &Value) {
    return putValueSync(toStringRef(Key).str(), Value);
  }

protected:
  virtual Expected<Optional<ValueTy>> getValueSyncImpl(std::string Key) = 0;
  virtual Error putValueSyncImpl(std::string Key, const ValueTy &Value) = 0;

public:
  class GetValueAsyncQueue : public AsyncQueueBase {
    virtual void anchor() override;

  public:
    virtual ~GetValueAsyncQueue() = default;

    void getValueAsync(std::string Key,
                       std::shared_ptr<AsyncCallerContext> CallCtx = nullptr) {
      getValueAsyncImpl(std::move(Key), std::move(CallCtx));
      ++NumPending;
    }
    void getValueAsync(ArrayRef<uint8_t> Key,
                       std::shared_ptr<AsyncCallerContext> CallCtx = nullptr) {
      getValueAsync(toStringRef(Key).str(), std::move(CallCtx));
    }

    struct Response {
      std::shared_ptr<AsyncCallerContext> CallCtx;
      // If this is \p None it means the key was not found.
      Optional<ValueTy> Value;
    };
    Expected<Response> receiveNext() {
      assert(NumPending);
      --NumPending;
      return receiveNextImpl();
    }

  protected:
    virtual void
    getValueAsyncImpl(std::string Key,
                      std::shared_ptr<AsyncCallerContext> CallCtx) = 0;
    virtual Expected<Response> receiveNextImpl() = 0;
  };

  class PutValueAsyncQueue : public AsyncQueueBase {
    virtual void anchor() override;

  public:
    virtual ~PutValueAsyncQueue() = default;

    void putValueAsync(std::string Key, const ValueTy &Value,
                       std::shared_ptr<AsyncCallerContext> CallCtx = nullptr) {
      putValueAsyncImpl(std::move(Key), Value, std::move(CallCtx));
      ++NumPending;
    }
    void putValueAsync(ArrayRef<uint8_t> Key, const ValueTy &Value,
                       std::shared_ptr<AsyncCallerContext> CallCtx = nullptr) {
      putValueAsync(toStringRef(Key).str(), Value, std::move(CallCtx));
    }

    struct Response {
      std::shared_ptr<AsyncCallerContext> CallCtx;
    };
    Expected<Response> receiveNext() {
      assert(NumPending);
      --NumPending;
      return receiveNextImpl();
    }

  protected:
    virtual void
    putValueAsyncImpl(std::string Key, const ValueTy &Value,
                      std::shared_ptr<AsyncCallerContext> CallCtx) = 0;
    virtual Expected<Response> receiveNextImpl() = 0;
  };

  GetValueAsyncQueue &getValueQueue() const { return *GetValueQueue; }
  PutValueAsyncQueue &putValueQueue() const { return *PutValueQueue; }

protected:
  std::unique_ptr<GetValueAsyncQueue> GetValueQueue;
  std::unique_ptr<PutValueAsyncQueue> PutValueQueue;
};

/// An asynchronous gRPC client for the CAS service of the remote cache
/// protocol.
///
/// Example usage:
/// \code
///   auto &LoadQueue = CASClient->loadQueue();
///   // Initiate one or more `CASLoad` requests.
///   LoadQueue.loadAsync(CASID, Path, std::make_shared<CallCtx>(OutputName));
///   // Wait for responses.
///   while (LoadQueue.hasPending()) {
///     auto Response = LoadQueue.receiveNext();
///     ...
///   }
/// \endcode
class CASDBClient {
  virtual void anchor();

public:
  virtual ~CASDBClient() = default;

  struct LoadResponse {
    bool KeyNotFound = false;
    Optional<std::string> BlobData;
  };
  struct GetResponse {
    bool KeyNotFound = false;
    Optional<std::string> BlobData;
    std::vector<std::string> Refs;
  };
  Expected<LoadResponse> loadSync(std::string CASID,
                                  Optional<std::string> OutFilePath = None) {
    return loadSyncImpl(std::move(CASID), std::move(OutFilePath));
  }
  Expected<std::string> saveDataSync(std::string BlobData) {
    return saveDataSyncImpl(std::move(BlobData));
  }
  Expected<std::string> saveFileSync(std::string FilePath) {
    return saveFileSyncImpl(std::move(FilePath));
  }
  Expected<GetResponse> getSync(std::string CASID,
                                Optional<std::string> OutFilePath = None) {
    return getSyncImpl(std::move(CASID), std::move(OutFilePath));
  }
  Expected<std::string> putDataSync(std::string BlobData,
                                    ArrayRef<std::string> Refs) {
    return putDataSyncImpl(std::move(BlobData), Refs);
  }
  Expected<std::string> putFileSync(std::string FilePath,
                                   ArrayRef<std::string> Refs) {
    return putFileSyncImpl(std::move(FilePath), Refs);
  }

protected:
  virtual Expected<LoadResponse>
  loadSyncImpl(std::string CASID, Optional<std::string> OutFilePath) = 0;
  virtual Expected<std::string> saveDataSyncImpl(std::string BlobData) = 0;
  virtual Expected<std::string> saveFileSyncImpl(std::string FilePath) = 0;
  virtual Expected<GetResponse>
  getSyncImpl(std::string CASID, Optional<std::string> OutFilePath) = 0;
  virtual Expected<std::string> putDataSyncImpl(std::string BlobData,
                                                ArrayRef<std::string> Refs) = 0;
  virtual Expected<std::string> putFileSyncImpl(std::string FilePath,
                                                ArrayRef<std::string> Refs) = 0;

public:
  class LoadAsyncQueue : public AsyncQueueBase {
    virtual void anchor() override;

  public:
    virtual ~LoadAsyncQueue() = default;

    void loadAsync(std::string CASID,
                   Optional<std::string> OutFilePath = None,
                   std::shared_ptr<AsyncCallerContext> CallCtx = nullptr) {
      loadAsyncImpl(std::move(CASID), std::move(OutFilePath),
                    std::move(CallCtx));
      ++NumPending;
    }

    struct Response {
      std::shared_ptr<AsyncCallerContext> CallCtx;
      bool KeyNotFound = false;
      Optional<std::string> BlobData;
    };
    Expected<Response> receiveNext() {
      assert(NumPending);
      --NumPending;
      return receiveNextImpl();
    }

  protected:
    virtual void loadAsyncImpl(std::string CASID,
                               Optional<std::string> OutFilePath,
                               std::shared_ptr<AsyncCallerContext> CallCtx) = 0;
    virtual Expected<Response> receiveNextImpl() = 0;
  };

  class SaveAsyncQueue : public AsyncQueueBase {
    virtual void anchor() override;

  public:
    virtual ~SaveAsyncQueue() = default;

    void saveDataAsync(std::string BlobData,
                       std::shared_ptr<AsyncCallerContext> CallCtx = nullptr) {
      saveDataAsyncImpl(std::move(BlobData), std::move(CallCtx));
      ++NumPending;
    }

    void saveFileAsync(std::string FilePath,
                       std::shared_ptr<AsyncCallerContext> CallCtx = nullptr) {
      saveFileAsyncImpl(std::move(FilePath), std::move(CallCtx));
      ++NumPending;
    }

    struct Response {
      std::shared_ptr<AsyncCallerContext> CallCtx;
      std::string CASID;
    };
    Expected<Response> receiveNext() {
      assert(NumPending);
      --NumPending;
      return receiveNextImpl();
    }

  protected:
    virtual void
    saveDataAsyncImpl(std::string BlobData,
                      std::shared_ptr<AsyncCallerContext> CallCtx) = 0;
    virtual void
    saveFileAsyncImpl(std::string FilePath,
                      std::shared_ptr<AsyncCallerContext> CallCtx) = 0;
    virtual Expected<Response> receiveNextImpl() = 0;
  };

  class GetAsyncQueue : public AsyncQueueBase {
    virtual void anchor() override;

  public:
    virtual ~GetAsyncQueue() = default;

    void getAsync(std::string CASID,
                  Optional<std::string> OutFilePath = None,
                  std::shared_ptr<AsyncCallerContext> CallCtx = nullptr) {
      getAsyncImpl(std::move(CASID), std::move(OutFilePath),
                   std::move(CallCtx));
      ++NumPending;
    }

    struct Response {
      std::shared_ptr<AsyncCallerContext> CallCtx;
      bool KeyNotFound = false;
      Optional<std::string> BlobData;
      std::vector<std::string> Refs;
    };
    Expected<Response> receiveNext() {
      assert(NumPending);
      --NumPending;
      return receiveNextImpl();
    }

  protected:
    virtual void getAsyncImpl(std::string CASID,
                              Optional<std::string> OutFilePath,
                              std::shared_ptr<AsyncCallerContext> CallCtx) = 0;
    virtual Expected<Response> receiveNextImpl() = 0;
  };

  class PutAsyncQueue : public AsyncQueueBase {
    virtual void anchor() override;

  public:
    virtual ~PutAsyncQueue() = default;

    void putDataAsync(std::string BlobData, ArrayRef<std::string> Refs,
                      std::shared_ptr<AsyncCallerContext> CallCtx = nullptr) {
      putDataAsyncImpl(std::move(BlobData), Refs, std::move(CallCtx));
      ++NumPending;
    }

    void putFileAsync(std::string FilePath, ArrayRef<std::string> Refs,
                      std::shared_ptr<AsyncCallerContext> CallCtx = nullptr) {
      putFileAsyncImpl(std::move(FilePath), Refs, std::move(CallCtx));
      ++NumPending;
    }

    struct Response {
      std::shared_ptr<AsyncCallerContext> CallCtx;
      std::string CASID;
    };
    Expected<Response> receiveNext() {
      assert(NumPending);
      --NumPending;
      return receiveNextImpl();
    }

  protected:
    virtual void
    putDataAsyncImpl(std::string BlobData, ArrayRef<std::string> Refs,
                     std::shared_ptr<AsyncCallerContext> CallCtx) = 0;
    virtual void
    putFileAsyncImpl(std::string FilePath, ArrayRef<std::string> Refs,
                     std::shared_ptr<AsyncCallerContext> CallCtx) = 0;
    virtual Expected<Response> receiveNextImpl() = 0;
  };

  LoadAsyncQueue &loadQueue() const { return *LoadQueue; }
  SaveAsyncQueue &saveQueue() const { return *SaveQueue; }
  GetAsyncQueue &getQueue() const { return *GetQueue; }
  PutAsyncQueue &putQueue() const { return *PutQueue; }

protected:
  std::unique_ptr<LoadAsyncQueue> LoadQueue;
  std::unique_ptr<SaveAsyncQueue> SaveQueue;
  std::unique_ptr<GetAsyncQueue> GetQueue;
  std::unique_ptr<PutAsyncQueue> PutQueue;
};

struct ClientServices {
  std::unique_ptr<KeyValueDBClient> KVDB;
  std::unique_ptr<CASDBClient> CASDB;
};

Expected<std::unique_ptr<CASDBClient>>
createRemoteCASDBClient(StringRef SocketPath);

Expected<std::unique_ptr<KeyValueDBClient>>
createRemoteKeyValueClient(StringRef SocketPath);

Expected<ClientServices>
createCompilationCachingRemoteClient(StringRef SocketPath);

} // namespace llvm::cas::remote

#endif
