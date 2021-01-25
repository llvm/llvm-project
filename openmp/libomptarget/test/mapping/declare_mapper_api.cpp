// RUN: %libomptarget-compilexx-run-and-check-aarch64-unknown-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-x86_64-pc-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-nvptx64-nvidia-cuda

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cinttypes>

// Data structure definitions copied from OpenMP RTL.
struct MapComponentInfoTy {
  void *Base;
  void *Begin;
  int64_t Size;
  int64_t Type;
  void *Name;
  MapComponentInfoTy() = default;
  MapComponentInfoTy(void *Base, void *Begin, int64_t Size, int64_t Type, void *Name)
      : Base(Base), Begin(Begin), Size(Size), Type(Type), Name(Name) {}
};

struct MapperComponentsTy {
  std::vector<MapComponentInfoTy> Components;
};

// OpenMP RTL interfaces
#ifdef __cplusplus
extern "C" {
#endif
int64_t __tgt_mapper_num_components(void *rt_mapper_handle);
void __tgt_push_mapper_component(void *rt_mapper_handle, void *base,
                                 void *begin, int64_t size, int64_t type, 
                                 void *name);
#ifdef __cplusplus
}
#endif

int main(int argc, char *argv[]) {
  MapperComponentsTy MC;
  void *base, *begin;
  int64_t size, type;
  // Push 2 elements into MC.
  __tgt_push_mapper_component((void *)&MC, base, begin, size, type, nullptr);
  __tgt_push_mapper_component((void *)&MC, base, begin, size, type, nullptr);
  int64_t num = __tgt_mapper_num_components((void *)&MC);
  // CHECK: num=2
  printf("num=%" PRId64 "\n", num);
  return 0;
}
