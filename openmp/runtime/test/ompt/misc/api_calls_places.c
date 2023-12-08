// RUN: %libomp-compile && env OMP_PLACES=cores %libomp-run | FileCheck %s
// REQUIRES: ompt, linux
#include "callback.h"
#include <omp.h>
#define __USE_GNU
#include <sched.h>
#undef __USE_GNU

void print_list(char *function_name, int size, int list[]) {
  printf("%" PRIu64 ": %s(0)=(%d", ompt_get_thread_data()->value, function_name,
         list[0]);
  int i;
  for (i = 1; i < size; i++) {
    printf(",%d", list[i]);
  }
  printf(")\n");
}

int main() {
#pragma omp parallel num_threads(1)
  {
    printf("%" PRIu64 ": omp_get_num_places()=%d\n",
           ompt_get_thread_data()->value, omp_get_num_places());
    printf("%" PRIu64 ": ompt_get_num_places()=%d\n",
           ompt_get_thread_data()->value, ompt_get_num_places());

    int omp_ids_size = omp_get_place_num_procs(0);
    int omp_ids[omp_ids_size];
    omp_get_place_proc_ids(0, omp_ids);
    print_list("omp_get_place_proc_ids", omp_ids_size, omp_ids);
    int ompt_ids_size = ompt_get_place_proc_ids(0, 0, NULL);
    int ompt_ids[ompt_ids_size];
    ompt_get_place_proc_ids(0, ompt_ids_size, ompt_ids);
    print_list("ompt_get_place_proc_ids", ompt_ids_size, ompt_ids);

    printf("%" PRIu64 ": omp_get_place_num()=%d\n",
           ompt_get_thread_data()->value, omp_get_place_num());
    printf("%" PRIu64 ": ompt_get_place_num()=%d\n",
           ompt_get_thread_data()->value, ompt_get_place_num());

    int omp_nums_size = omp_get_partition_num_places();
    int omp_nums[omp_nums_size];
    omp_get_partition_place_nums(omp_nums);
    print_list("omp_get_partition_place_nums", omp_nums_size, omp_nums);
    int ompt_nums_size = ompt_get_partition_place_nums(0, omp_nums);
    int ompt_nums[ompt_nums_size];
    ompt_get_partition_place_nums(ompt_nums_size, ompt_nums);
    print_list("ompt_get_partition_place_nums", ompt_nums_size, ompt_nums);

    printf("%" PRIu64 ": sched_getcpu()=%d\n", ompt_get_thread_data()->value,
           sched_getcpu());
    printf("%" PRIu64 ": ompt_get_proc_id()=%d\n",
           ompt_get_thread_data()->value, ompt_get_proc_id());

    printf("%" PRIu64 ": omp_get_num_procs()=%d\n",
           ompt_get_thread_data()->value, omp_get_num_procs());
    printf("%" PRIu64 ": ompt_get_num_procs()=%d\n",
           ompt_get_thread_data()->value, ompt_get_num_procs());
  }

  // Check if libomp supports the callbacks for this test.

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: omp_get_num_places
  // CHECK-SAME: ()=[[NUM_PLACES:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_num_places()=[[NUM_PLACES]]

  // CHECK: {{^}}[[MASTER_ID]]: omp_get_place_proc_ids
  // CHECK-SAME: (0)=([[PROC_IDS:[0-9\,]+]])
  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_place_proc_ids(0)=([[PROC_IDS]])

  // CHECK: {{^}}[[MASTER_ID]]: omp_get_place_num()=[[PLACE_NUM:[-]?[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_place_num()=[[PLACE_NUM]]

  // CHECK: {{^}}[[MASTER_ID]]: omp_get_partition_place_nums
  // CHECK-SAME: (0)=([[PARTITION_PLACE_NUMS:[0-9\,]+]])
  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_partition_place_nums
  // CHECK-SAME: (0)=([[PARTITION_PLACE_NUMS]])

  // CHECK: {{^}}[[MASTER_ID]]: sched_getcpu()=[[CPU_ID:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_proc_id()=[[CPU_ID]]

  // CHECK: {{^}}[[MASTER_ID]]: omp_get_num_procs()=[[NUM_PROCS:[-]?[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_num_procs()=[[NUM_PROCS]]

  return 0;
}
