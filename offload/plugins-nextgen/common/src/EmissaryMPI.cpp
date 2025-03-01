//===---- offload/plugins-nextgen/common/src/EmissaryFortrt.cpp  ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host support for Fortran runtime Emissary API
//
//===----------------------------------------------------------------------===//
#include "PluginInterface.h"
#include "RPC.h"
#include "Shared/Debug.h"
#include "Shared/RPCOpcodes.h"
#include "shared/rpc.h"
#include "shared/rpc_opcodes.h"
#include "../../../DeviceRTL/include/EmissaryIds.h"
#include "Emissary.h"
#include <assert.h>
#include <cstring>
#include <ctype.h>
#include <list>
#include <mpi.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tuple>
#include <vector>

extern "C" {
extern int V_MPI_Send(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  void *v0 = va_arg(args, void *);
  int v1 = va_arg(args, int);
  MPI_Datatype v2 = va_arg(args, MPI_Datatype);
  int v3 = va_arg(args, int);
  int v4 = va_arg(args, int);
  MPI_Comm v5 = va_arg(args, MPI_Comm);
  va_end(args);
  int rval = MPI_Send(v0, v1, v2, v3, v4, v5);
  return rval;
}
extern int V_MPI_Recv(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  void *v0 = va_arg(args, void *);
  int v1 = va_arg(args, int);
  MPI_Datatype v2 = va_arg(args, MPI_Datatype);
  int v3 = va_arg(args, int);
  int v4 = va_arg(args, int);
  MPI_Comm v5 = va_arg(args, MPI_Comm);
  MPI_Status *v6 = va_arg(args, MPI_Status *);
  va_end(args);
  int rval = MPI_Recv(v0, v1, v2, v3, v4, v5, v6);
  return rval;
}

emis_return_t EmissaryMPI(char *data, emisArgBuf_t *ab) {
  uint64_t *a[MAXVARGS];
  if (EmissaryBuildVargs(ab->NumArgs, ab->keyptr, ab->argptr, ab->strptr,
                         &ab->data_not_used, &a[0]) != _RC_SUCCESS)
    return (emis_return_t)0;

  switch (ab->emisfnid) {
  case _MPI_Send_idx: {
    void *fnptr = (void *)V_MPI_Send;
    int return_value_int =
        V_MPI_Send(fnptr, a[0], a[1], a[2], a[3], a[4], a[5]);
    return (emis_return_t)return_value_int;
  }
  case _MPI_Recv_idx: {
    void *fnptr = (void *)V_MPI_Recv;
    int return_value_int =
        V_MPI_Recv(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
    return (emis_return_t)return_value_int;
  }
  }
  return (emis_return_t)0;
}

} // end extern "C"
