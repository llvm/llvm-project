/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *     of MPICH source repository.
 */

typedef int MPI_Comm;
#define MPI_COMM_WORLD ((MPI_Comm)0x44000000)

typedef int MPI_Datatype;
#define MPI_FLOAT ((MPI_Datatype)0x4c00040a)
#define MPI_DOUBLE ((MPI_Datatype)0x4c00080b)
#define MPI_INT8_T ((MPI_Datatype)0x4c000137)
#define MPI_INT16_T ((MPI_Datatype)0x4c000238)
#define MPI_INT32_T ((MPI_Datatype)0x4c000439)
#define MPI_INT64_T ((MPI_Datatype)0x4c00083a)
#define MPI_UINT8_T ((MPI_Datatype)0x4c00013b)
#define MPI_UINT16_T ((MPI_Datatype)0x4c00023c)
#define MPI_UINT32_T ((MPI_Datatype)0x4c00043d)
#define MPI_UINT64_T ((MPI_Datatype)0x4c00083e)

typedef struct MPI_Status;
#define MPI_STATUS_IGNORE (MPI_Status *)1

#define _MPI_FALLBACK_DEFS 1
