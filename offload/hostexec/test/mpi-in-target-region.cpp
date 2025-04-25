#include "mpi_supplemental.h"
#include "omp.h"
#include <mpi.h>
#include <stdio.h>
// #define VSIZE 256 * 8
#define VSIZE 256

int main(int argc, char *argv[]) {
  _mpilib_set_device_globals();
  int numranks, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("Number of Ranks= %d My rank= %d\n", numranks, rank);
  MPI_Comm _mpi_comm = MPI_COMM_WORLD;
  MPI_Datatype _mpi_int = MPI_INT;
  int *send_recv_buffer = (int *)malloc(VSIZE * sizeof(int));
  printf("buffer address is %p\n", send_recv_buffer);

#pragma omp target teams distribute parallel for map(                          \
        to : send_recv_buffer[0 : VSIZE])
  for (int i = 0; i < VSIZE; i++) {
    int rc = 0;
    int TH = omp_get_thread_num();
    int NTH = omp_get_num_threads();
    int TM = omp_get_team_num();
    int NTM = omp_get_num_teams();
    int L = TH % 64;
    if (rank == 0) {
      send_recv_buffer[i] = -i;
      if (L == 0 || L == 63) // Only print 1st and last lane
        printf("P:%d TAG:%d sending %d with dev addr %p team:%d of %d  "
               "thread:%d of %d  LANE:%d  WARP:%d \n",
               rank, i, send_recv_buffer[i], &send_recv_buffer[i], TM, NTM, TH,
               NTH, L, TH / 64);
      MPI_Send(&send_recv_buffer[i], 1, _mpi_int, 1, i, _mpi_comm);
    } else {
      MPI_Recv(&send_recv_buffer[i], 1, _mpi_int, 0, i, _mpi_comm,
               MPI_STATUS_IGNORE);
      if (send_recv_buffer[i] != -i)
        rc = 1;
      if (rc != 0 || L == 0 || L == 63) // Only print 1st and last lane
        printf("P:%d TAG:%d received %d with dev addr %p team:%d of %d  "
               "thread:%d of %d  LANE:%d  WARP:%d rc:%d\n",
               rank, i, send_recv_buffer[i], &send_recv_buffer[i], TM, NTM, TH,
               NTH, L, TH / 64, rc);
    }
  } // end for loop and target region

  MPI_Finalize();
  //  fprintf(stderr,"rc = %d\n",rc);
  return 0;
}
