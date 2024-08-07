#ifndef OFFLOAD_TEST_OMPTEST_INCLUDE_OMPTTESTER_H
#define OFFLOAD_TEST_OMPTEST_INCLUDE_OMPTTESTER_H

#include "AssertMacros.h"
#include "Logging.h"
#include "OmptAliases.h"
#include "OmptAssertEvent.h"
#include "OmptAsserter.h"
#include "OmptCallbackHandler.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#ifdef LIBOFFLOAD_LIBOMPTEST_USE_GOOGLETEST
#include "OmptTesterGoogleTest.h"
#else
#include "OmptTesterStandalone.h"
#endif

#endif
