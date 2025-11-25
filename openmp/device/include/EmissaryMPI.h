//===--------------- offload/DeviceRTL/include/EmissaryMPI.h --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// EmissaryMPI.h This include must be included by MPI application
//
//===----------------------------------------------------------------------===//
#include "EmissaryIds.h"
#include <stdarg.h>

typedef enum {
  _MPI_INVALID,
  _MPI_Send_idx,
  _MPI_Recv_idx,
  _MPI_Allreduce_idx,
  _MPI_Reduce_idx,
} offload_emis_mpi_t;

///  Device stubs that call _emissary_exec using identical host API interface
#if defined(__NVPTX__) || defined(__AMDGCN__)
extern "C" int MPI_Send(const void *buf, int count, MPI_Datatype datatype,
                        int dest, int tag, MPI_Comm comm) {
  return (int)_emissary_exec(_PACK_EMIS_IDS(EMIS_ID_MPI, _MPI_Send_idx), 
		         buf, count, datatype, dest, tag, comm);
}
extern "C" int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source,
                        int tag, MPI_Comm comm, MPI_Status *st) {
  return (int)_emissary_exec(_PACK_EMIS_IDS(EMIS_ID_MPI, _MPI_Recv_idx), buf,
                             count, datatype, source, tag, comm, st);
}
extern "C" int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  return (int)_emissary_exec(_PACK_EMIS_IDS(EMIS_ID_MPI, _MPI_Allreduce_idx), 
		  sendbuf, recvbuf, count, datatype, op, comm);
}
extern "C" int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                              MPI_Op op, int root, MPI_Comm comm) {
  return (int)_emissary_exec(_PACK_EMIS_IDS(EMIS_ID_MPI, _MPI_Reduce_idx), 
                   sendbuf, recvbuf, count, datatype, op, root, comm);
}
#endif

/// Host variadic wrapper functions.
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
extern int V_MPI_Allreduce(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  void *buf             = va_arg(args, void *);
  void *recvbuf         = va_arg(args, void *);
  int count             = va_arg(args, int);
  MPI_Datatype datatype = va_arg(args, MPI_Datatype);
  MPI_Op op             = va_arg(args, MPI_Op);
  MPI_Comm comm         = va_arg(args, MPI_Comm);
  va_end(args);
  int rval = MPI_Allreduce(
    buf, recvbuf, count, datatype, op, comm);
  return rval;
}
extern int V_MPI_Reduce(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  void *buf             = va_arg(args, void *);
  void *recvbuf         = va_arg(args, void *);
  int count             = va_arg(args, int);
  MPI_Datatype datatype = va_arg(args, MPI_Datatype);
  MPI_Op op             = va_arg(args, MPI_Op);
  int root              = va_arg(args, int);
  MPI_Comm comm         = va_arg(args, MPI_Comm);
  va_end(args);
  int rval = MPI_Reduce(
    buf, recvbuf, count, datatype, op, root, comm);
  return rval;
}

/// EmissaryMPI function selector
emis_return_t EmissaryMPI(char *data, emisArgBuf_t *ab, emis_argptr_t *a[]) {

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
  case _MPI_Allreduce_idx: {
    void *fnptr = (void *)V_MPI_Allreduce;
    int return_value_int =
        V_MPI_Allreduce(fnptr, a[0], a[1], a[2], a[3], a[4], a[5]);
    return (emis_return_t) return_value_int;
  }
  case _MPI_Reduce_idx: {
    void *fnptr = (void *)V_MPI_Reduce;
    int return_value_int =
        V_MPI_Reduce(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
    return (emis_return_t) return_value_int;
  }
  }
  return (emis_return_t)0;
}

} // end extern "C"
