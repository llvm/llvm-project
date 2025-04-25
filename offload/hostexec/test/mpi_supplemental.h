//
// mpi_supplemental.h: Supplemental header to build device variants for specific
//                     openmpi host function declarations using hostexec:
//
// int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
//              int tag, MPI_Comm comm);
// int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source,
//              int tag, MPI_Comm comm, MPI_Status *status);
//
// These functions can now be called from OpenMP target regions without
// source code modifications.
//
#include <hostexec.h>
#include <mpi.h>
#include <stdarg.h>

//  There are 4 parts to a supplemental header.

// 1. Create variadic proxy functions
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

#pragma omp begin declare target

// 2. Create global variables to store pointers to variadic proxy functions.
//    These must be inside a declare target.
hostexec_int_t *V_MPI_Send_var;
hostexec_int_t *V_MPI_Recv_var;

// 3. Create the device variants that call hostexec.
#pragma omp begin declare variant match(                                       \
        device = {arch(amdgcn, nvptx, nvptx64)},                               \
            implementation = {extension(match_any)})
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm) {
  return hostexec_int((void *)V_MPI_Send_var, buf, count, datatype, dest, tag,
                      comm);
}
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status *st) {
  return hostexec_int((void *)V_MPI_Recv_var, buf, count, datatype, source, tag,
                      comm, st);
}
#pragma omp end declare variant
#pragma omp end declare target

// 4.  Initialize pointers to host-only library functions on the device.
//     These are host pointers stored as device globals which are passed
//     to hostexec in the device variants above.
void _mpilib_set_device_globals() {
  V_MPI_Send_var = V_MPI_Send;
  V_MPI_Recv_var = V_MPI_Recv;
#pragma omp target update to(V_MPI_Send_var, V_MPI_Recv_var)
}
