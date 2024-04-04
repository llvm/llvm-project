; RUN: llc --mtriple=amdgcn-amd-amdhsa %s -o -

define internal void @_omp_reduction_list_to_global_reduce_func() {
  ret void
}
