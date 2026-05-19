//===- QueueManager.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QueueManager.h"
#include "PluginManager.h"
#include "Private.h"
#include "Shared/Debug.h"

namespace llvm::acc::target {
QueueManagerTy *QueueManager = nullptr;
} // namespace llvm::acc::target

using namespace llvm::acc::target;

static void synchronizeQueueOrDie(QueueIdTy Queue, DeviceTy &Device,
                                  AsyncInfoTy *AsyncInfo) {

  ODBG() << "Synchronizing stream " << AsyncInfo << " for device "
         << Device.DeviceID << " " << &Device << " with queue ID " << Queue;
  auto Res = AsyncInfo->synchronize();
  if (Res != OFFLOAD_SUCCESS)
    FATAL_MESSAGE(Device.DeviceID,
                  "Failed to synchronize queue %" PRIi32 " on device %d", Queue,
                  Device.DeviceID);
}

static QueueManagerTy::StatusTy
queryQueueOrDie(QueueIdTy Queue, DeviceTy &Device, AsyncInfoTy *AsyncInfo) {

  ODBG() << "Querying stream " << AsyncInfo << " for device " << Device.DeviceID
         << " " << &Device << " with queue ID " << Queue;
  auto Res = AsyncInfo->query();
  if (Res == OFFLOAD_FAIL)
    FATAL_MESSAGE(Device.DeviceID,
                  "Failed to query queue %" PRIi32 " on device %d", Queue,
                  Device.DeviceID);
  return static_cast<QueueManagerTy::StatusTy>(Res);
}

QueueManagerTy::QueueManagerTy() {}

QueueManagerTy::~QueueManagerTy() {}

void QueueManagerTy::synchronize() {
  for (auto &[Key, Q] : QueueMap) {
    auto &[D, Id] = Key;
    synchronizeQueueOrDie(Id, *D, Q.get());
  }
}

void QueueManagerTy::synchronize(DeviceTy &Device) {
  for (auto &[Key, Q] : QueueMap) {
    auto &[D, Id] = Key;
    if (&Device != D)
      continue;
    synchronizeQueueOrDie(Id, Device, Q.get());
  }
}

void QueueManagerTy::synchronize(DeviceTy &Device, QueueIdTy Queue) {
  AsyncInfoTy *AsyncInfo = QueueManager->get(Device, Queue);
  synchronizeQueueOrDie(Queue, Device, AsyncInfo);
}

QueueManagerTy::StatusTy QueueManagerTy::query() {
  for (auto &[Key, Q] : QueueMap) {
    auto &[D, Id] = Key;
    if (queryQueueOrDie(Id, *D, Q.get()) == StatusTy::NOT_READY)
      return StatusTy::NOT_READY;
  }
  return StatusTy::READY;
}

QueueManagerTy::StatusTy QueueManagerTy::query(DeviceTy &Device) {
  for (auto &[Key, Q] : QueueMap) {
    auto &[D, Id] = Key;
    if (&Device != D)
      continue;
    if (queryQueueOrDie(Id, *D, Q.get()) == StatusTy::NOT_READY)
      return StatusTy::NOT_READY;
  }
  return StatusTy::READY;
}

QueueManagerTy::StatusTy QueueManagerTy::query(DeviceTy &Device,
                                               QueueIdTy Queue) {
  AsyncInfoTy *AsyncInfo = QueueManager->get(Device, Queue);
  return queryQueueOrDie(Queue, Device, AsyncInfo);
}

AsyncInfoTy *QueueManagerTy::get(DeviceTy &Device, QueueIdTy QueueId) {
  static std::mutex Mutex;
  std::lock_guard<std::mutex> G(Mutex);

  auto Insertion = QueueMap.insert({std::make_pair(&Device, QueueId), nullptr});
  if (Insertion.second) {
    Insertion.first->second = std::make_unique<AsyncInfoTy>(
        Device, AsyncInfoTy::SyncTy::STATIC_NON_BLOCKING);
    ODBG() << "Initialized new stream for device " << &Device << " id "
           << QueueId << " -> " << Insertion.first->second.get();
  }
  return Insertion.first->second.get();
}

namespace llvm::acc::target {
void accAsyncWait(ident_t *Loc, int64_t DeviceId, int64_t WaitArg) {
  int64_t WaitArgs[] = {WaitArg};
  accAsyncWait(Loc, DeviceId, 1, WaitArgs);
}

void accAsyncWait(ident_t *Loc, int64_t DeviceId, uint32_t WaitNum,
                  int64_t *WaitList) {
  ODBG() << "Synchronizing streams for device " << DeviceId;

  auto DeviceOrErr = PM->getDevice(DeviceId);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceId, "%s", toString(DeviceOrErr.takeError()).c_str());

  if (WaitNum == 0) {
    QueueManager->synchronize(*DeviceOrErr);
  } else {
    for (unsigned I = 0; I < WaitNum; I++)
      QueueManager->synchronize(*DeviceOrErr, WaitList[I]);
  }
}

void accAsyncWaitAll(ident_t *Loc, int64_t DeviceId) {
  ODBG() << "Synchronizing all streams for device " << DeviceId;

  auto DeviceOrErr = PM->getDevice(DeviceId);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceId, "%s", toString(DeviceOrErr.takeError()).c_str());
  QueueManager->synchronize(*DeviceOrErr);
}

void accAsyncWaitAll(ident_t *Loc) {
  ODBG() << "Synchronizing all streams";
  QueueManager->synchronize();
}

int accAsyncTest(ident_t *Loc, int64_t DeviceId, int64_t TestArg) {
  int64_t TestList[] = {TestArg};
  return accAsyncTest(Loc, DeviceId, 1, TestList);
}

int accAsyncTest(ident_t *Loc, int64_t DeviceId, uint32_t TestNum,
                 int64_t *TestList) {
  ODBG() << "Querying streams for device " << DeviceId;

  auto DeviceOrErr = PM->getDevice(DeviceId);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceId, "%s", toString(DeviceOrErr.takeError()).c_str());

  for (unsigned I = 0; I < TestNum; I++)
    if (QueueManager->query(*DeviceOrErr, TestList[I]) ==
        QueueManagerTy::StatusTy::NOT_READY)
      return static_cast<int>(QueueManagerTy::StatusTy::NOT_READY);
  return static_cast<int>(QueueManagerTy::StatusTy::READY);
}

int accAsyncTestAll(ident_t *Loc, int64_t DeviceId) {
  ODBG() << "Querying all streams for device " << DeviceId;

  auto DeviceOrErr = PM->getDevice(DeviceId);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceId, "%s", toString(DeviceOrErr.takeError()).c_str());
  return static_cast<int>(QueueManager->query(*DeviceOrErr));
}

int accAsyncTestAll(ident_t *Loc) {
  ODBG() << "Querying all streams";
  return static_cast<int>(QueueManager->query());
}
} // namespace llvm::acc::target
