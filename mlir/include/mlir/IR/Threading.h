//===- Threading.h - MLIR Threading Utilities -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines various utilies for multithreaded processing within MLIR.
// These utilities automatically handle many of the necessary threading
// conditions, such as properly ordering diagnostics, observing if threading is
// disabled, etc. These utilities should be used over other threading utilities
// whenever feasible.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_THREADING_H
#define MLIR_IR_THREADING_H

#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/ThreadPool.h"
#include <atomic>

namespace mlir {

/// Invoke the given function on the elements between [begin, end)
/// asynchronously. If the given function returns a failure when processing any
/// of the elements, execution is stopped and a failure is returned from this
/// function. This means that in the case of failure, not all elements of the
/// range will be processed. Diagnostics emitted during processing are ordered
/// relative to the element's position within [begin, end). If the provided
/// context does not have multi-threading enabled, this function always
/// processes elements sequentially.
template <typename IteratorT, typename FuncT>
LogicalResult failableParallelForEach(MLIRContext *context, IteratorT begin,
                                      IteratorT end, FuncT &&func) {
  unsigned numElements = static_cast<unsigned>(std::distance(begin, end));
  if (numElements == 0)
    return success();

  // If multithreading is disabled or there is a small number of elements,
  // process the elements directly on this thread.
  // FIXME: ThreadPool should allow work stealing to avoid deadlocks when
  // scheduling work within a worker thread.
  if (!context->isMultithreadingEnabled() || numElements <= 1 ||
      context->getThreadPool().isWorkerThread()) {
    for (; begin != end; ++begin)
      if (failed(func(*begin)))
        return failure();
    return success();
  }

  // Build a wrapper processing function that properly initializes a parallel
  // diagnostic handler.
  ParallelDiagnosticHandler handler(context);
  std::atomic<unsigned> curIndex(0);
  std::atomic<bool> processingFailed(false);
  auto processFn = [&] {
    while (!processingFailed) {
      unsigned index = curIndex++;
      if (index >= numElements)
        break;
      handler.setOrderIDForThread(index);
      if (failed(func(*std::next(begin, index))))
        processingFailed = true;
      handler.eraseOrderIDForThread();
    }
  };

  // Otherwise, process the elements in parallel.
  llvm::ThreadPool &threadPool = context->getThreadPool();
  size_t numActions = std::min(numElements, threadPool.getThreadCount());
  SmallVector<std::shared_future<void>> threadFutures;
  threadFutures.reserve(numActions - 1);
  for (unsigned i = 1; i < numActions; ++i)
    threadFutures.emplace_back(threadPool.async(processFn));
  processFn();

  // Wait for all of the threads to finish.
  for (std::shared_future<void> &future : threadFutures)
    future.wait();
  return failure(processingFailed);
}

/// Invoke the given function on the elements in the provided range
/// asynchronously. If the given function returns a failure when processing any
/// of the elements, execution is stopped and a failure is returned from this
/// function. This means that in the case of failure, not all elements of the
/// range will be processed. Diagnostics emitted during processing are ordered
/// relative to the element's position within the range. If the provided context
/// does not have multi-threading enabled, this function always processes
/// elements sequentially.
template <typename RangeT, typename FuncT>
LogicalResult failableParallelForEach(MLIRContext *context, RangeT &&range,
                                      FuncT &&func) {
  return failableParallelForEach(context, std::begin(range), std::end(range),
                                 std::forward<FuncT>(func));
}

/// Invoke the given function on the elements between [begin, end)
/// asynchronously. If the given function returns a failure when processing any
/// of the elements, execution is stopped and a failure is returned from this
/// function. This means that in the case of failure, not all elements of the
/// range will be processed. Diagnostics emitted during processing are ordered
/// relative to the element's position within [begin, end). If the provided
/// context does not have multi-threading enabled, this function always
/// processes elements sequentially.
template <typename FuncT>
LogicalResult failableParallelForEachN(MLIRContext *context, size_t begin,
                                       size_t end, FuncT &&func) {
  return failableParallelForEach(context, llvm::seq(begin, end),
                                 std::forward<FuncT>(func));
}

/// Invoke the given function on the elements between [begin, end)
/// asynchronously. Diagnostics emitted during processing are ordered relative
/// to the element's position within [begin, end). If the provided context does
/// not have multi-threading enabled, this function always processes elements
/// sequentially.
template <typename IteratorT, typename FuncT>
void parallelForEach(MLIRContext *context, IteratorT begin, IteratorT end,
                     FuncT &&func) {
  (void)failableParallelForEach(context, begin, end, [&](auto &&value) {
    return func(std::forward<decltype(value)>(value)), success();
  });
}

/// Invoke the given function on the elements in the provided range
/// asynchronously. Diagnostics emitted during processing are ordered relative
/// to the element's position within the range. If the provided context does not
/// have multi-threading enabled, this function always processes elements
/// sequentially.
template <typename RangeT, typename FuncT>
void parallelForEach(MLIRContext *context, RangeT &&range, FuncT &&func) {
  parallelForEach(context, std::begin(range), std::end(range),
                  std::forward<FuncT>(func));
}

/// Invoke the given function on the elements between [begin, end)
/// asynchronously. Diagnostics emitted during processing are ordered relative
/// to the element's position within [begin, end). If the provided context does
/// not have multi-threading enabled, this function always processes elements
/// sequentially.
template <typename FuncT>
void parallelForEachN(MLIRContext *context, size_t begin, size_t end,
                      FuncT &&func) {
  parallelForEach(context, llvm::seq(begin, end), std::forward<FuncT>(func));
}

} // namespace mlir

#endif // MLIR_IR_THREADING_H
