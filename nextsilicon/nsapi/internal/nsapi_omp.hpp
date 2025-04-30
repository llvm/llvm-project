#pragma once

#include <nsapi/nsapi.hpp>

struct NSAPICommandOMPSetNumThreads : public NSAPICommand {
public:
  uint32_t NumThreads;

  explicit NSAPICommandOMPSetNumThreads(uint32_t num_threads)
      : NSAPICommand(NSAPI_OMP_SET_NUM_THREADS), NumThreads(num_threads) {}
};

struct NSAPICommandOMPSetDynamic : public NSAPICommand {
public:
  uint32_t IsDynamic;

  explicit NSAPICommandOMPSetDynamic(uint32_t is_dynamic)
      : NSAPICommand(NSAPI_OMP_SET_DYNAMIC), IsDynamic(is_dynamic) {}
};
