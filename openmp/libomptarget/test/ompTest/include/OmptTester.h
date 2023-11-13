#ifndef OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTTESTER_H
#define OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTTESTER_H

#include "AssertMacros.h"
#include "OmptAliases.h"
#include "OmptAssertEvent.h"
#include "OmptAsserter.h"
#include "OmptCallbackHandler.h"

#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <omp-tools.h>

#ifdef __cplusplus
extern "C" {
#endif
ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                          const char *runtime_version);
int start_trace();
int flush_trace();
int stop_trace();
void libomptest_global_eventreporter_set_active(bool State);
#ifdef __cplusplus
}
#endif

#ifdef LIBOMPTARGET_LIBOMPTEST_USE_GOOGLETEST
#include "OmptTesterGoogleTest.h"
#else
#include "OmptTesterStandalone.h"
#endif

#endif // include guard
