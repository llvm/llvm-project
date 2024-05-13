// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _PSTL_PARALLEL_BACKEND_H
#define _PSTL_PARALLEL_BACKEND_H

#include "pstl_config.h"

#if defined(_PSTL_PAR_BACKEND_SERIAL)
#    include "parallel_backend_serial.h"
namespace __pstl
{
namespace __par_backend = __serial_backend;
}
#elif defined(_PSTL_PAR_BACKEND_TBB)
#    include "parallel_backend_tbb.h"
namespace __pstl
{
namespace __par_backend = __tbb_backend;
}
#elif defined(_PSTL_PAR_BACKEND_OPENMP)
#    include "parallel_backend_omp.h"
namespace __pstl
{
namespace __par_backend = __omp_backend;
}
#else
_PSTL_PRAGMA_MESSAGE("Parallel backend was not specified");
#endif

#endif /* _PSTL_PARALLEL_BACKEND_H */
