// -*- C++ -*-
//===-- parallel_backend.h ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __PSTL_parallel_backend_H
#define __PSTL_parallel_backend_H

#if __PSTL_PAR_BACKEND_TBB
#include "parallel_backend_tbb.h"
#else
__PSTL_PRAGMA_MESSAGE("Parallel backend was not specified");
#endif

#endif /* __PSTL_parallel_backend_H */
