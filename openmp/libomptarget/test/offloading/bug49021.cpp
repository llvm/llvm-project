// RUN: %libomptarget-compilexx-aarch64-unknown-linux-gnu -O3 && %libomptarget-run-aarch64-unknown-linux-gnu
// RUN: %libomptarget-compilexx-powerpc64-ibm-linux-gnu -O3 && %libomptarget-run-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-compilexx-powerpc64le-ibm-linux-gnu -O3 && %libomptarget-run-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-compilexx-x86_64-pc-linux-gnu -O3 && %libomptarget-run-x86_64-pc-linux-gnu
// RUN: %libomptarget-compilexx-nvptx64-nvidia-cuda -O3 && %libomptarget-run-nvptx64-nvidia-cuda

#include <iostream>

template <typename T> int test_map() {
  std::cout << "map(complex<>)" << std::endl;
  T a(0.2), a_check;
#pragma omp target map(from : a_check)
  { a_check = a; }

  if (a_check != a) {
    std::cout << " wrong results";
    return 1;
  }

  return 0;
}

template <typename T> int test_reduction() {
  std::cout << "flat parallelism" << std::endl;
  T sum(0), sum_host(0);
  const int size = 100;
  T array[size];
  for (int i = 0; i < size; i++) {
    array[i] = i;
    sum_host += array[i];
  }

#pragma omp target teams distribute parallel for map(to: array[:size])         \
                                                 reduction(+ : sum)
  for (int i = 0; i < size; i++)
    sum += array[i];

  if (sum != sum_host)
    std::cout << " wrong results " << sum << " host " << sum_host << std::endl;

  std::cout << "hierarchical parallelism" << std::endl;
  const int nblock(10), block_size(10);
  T block_sum[nblock];
#pragma omp target teams distribute map(to                                     \
                                        : array[:size])                        \
    map(from                                                                   \
        : block_sum[:nblock])
  for (int ib = 0; ib < nblock; ib++) {
    T partial_sum = 0;
    const int istart = ib * block_size;
    const int iend = (ib + 1) * block_size;
#pragma omp parallel for reduction(+ : partial_sum)
    for (int i = istart; i < iend; i++)
      partial_sum += array[i];
    block_sum[ib] = partial_sum;
  }

  sum = 0;
  for (int ib = 0; ib < nblock; ib++) {
    sum += block_sum[ib];
  }

  if (sum != sum_host) {
    std::cout << " wrong results " << sum << " host " << sum_host << std::endl;
    return 1;
  }

  return 0;
}

template <typename T> int test_complex() {
  int ret = 0;
  ret |= test_map<T>();
  ret |= test_reduction<T>();
  return ret;
}

int main() {
  int ret = 0;
  std::cout << "Testing float" << std::endl;
  ret |= test_complex<float>();
  std::cout << "Testing double" << std::endl;
  ret |= test_complex<double>();
  return ret;
}
