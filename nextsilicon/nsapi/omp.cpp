#include "omp.h"
#include "internal/nsapi_omp.hpp"

int nsapi_omp_set_num_threads(uint32_t num_threads) {
  auto cmd = NSAPICommandOMPSetNumThreads(num_threads);
  return NSAPIHandler::Current().Execute(cmd);
}

int nsapi_omp_set_dynamic(uint32_t is_dynamic) {
  auto cmd = NSAPICommandOMPSetDynamic(is_dynamic);
  return NSAPIHandler::Current().Execute(cmd);
}
