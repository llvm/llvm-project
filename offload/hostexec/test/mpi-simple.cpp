#include "mpi_supplemental.h"
#include "omp.h"
#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  _mpilib_set_device_globals();
  int numranks, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("Number of Ranks= %d My rank= %d\n", numranks, rank);
  MPI_Comm _mpi_comm = MPI_COMM_WORLD;
  MPI_Datatype _mpi_int = MPI_INT;
  int send_recv_buffer[2];
  printf("buffer address is %p\n", send_recv_buffer);

#pragma omp target
  {
    if (rank == 0) {
      send_recv_buffer[0] = -123;
      printf("\nProcess %d sending  %d from dev address %p to process 1\n",
             rank, send_recv_buffer[0], send_recv_buffer);
      MPI_Send(&send_recv_buffer[0], 1, _mpi_int, 1, 0, _mpi_comm);
    }
    if (rank == 1) {
      MPI_Recv(&send_recv_buffer[0], 1, _mpi_int, 0, 0, _mpi_comm,
               MPI_STATUS_IGNORE);
      printf("\nProcess %d received %d from dev address %p from process 0\n\n",
             rank, send_recv_buffer[0], send_recv_buffer);
    }
  }

  MPI_Finalize();

  return 0;
}
