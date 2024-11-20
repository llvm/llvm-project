// RUN: %libomptarget-compilexx-generic -O3 && %libomptarget-run-generic
// RUN: %libomptarget-compilexx-generic -O3 -ffast-math && %libomptarget-run-generic

#include <complex>
#include <iostream>

bool failed = false;

template <typename T> void test_map() {
  std::cout << "map(complex<T>)" << std::endl;
  std::complex<T> a(0.2, 1), a_check;
#pragma omp target map(from : a_check)
  { a_check = a; }

  if (std::abs(a - a_check) > 1e-6) {
    std::cout << "wrong map value check" << a_check << " correct value " << a
              << std::endl;
    failed = true;
  }
}

#if !defined(__NO_UDR)
#pragma omp declare reduction(+ : std::complex <float> : omp_out += omp_in)
#pragma omp declare reduction(+ : std::complex <double> : omp_out += omp_in)
#endif

template <typename T> class initiator {
public:
  static T value(int i) { return T(i); }
};

template <typename T> class initiator<std::complex<T>> {
public:
  static std::complex<T> value(int i) { return {T(i), T(-i)}; }
};

template <typename T> void test_reduction() {
  T sum(0), sum_host(0);
  const int size = 100;
  T array[size];
  for (int i = 0; i < size; i++) {
    array[i] = initiator<T>::value(i);
    sum_host += array[i];
  }

#pragma omp target teams distribute parallel for map(to : array[:size]) reduction(+ : sum)
  for (int i = 0; i < size; i++)
    sum += array[i];

  if (std::abs(sum - sum_host) > 1e-6) {
    std::cout << "wrong reduction value check" << sum << " correct value "
              << sum_host << std::endl;
    failed = true;
  }

  const int nblock(10), block_size(10);
  T block_sum[nblock];
#pragma omp target teams distribute map(to                                     \
                                        : array[:size])                        \
    map(from                                                                   \
        : block_sum[:nblock])
  for (int ib = 0; ib < nblock; ib++) {
    T partial_sum(0);
    const int istart = ib * block_size;
    const int iend = (ib + 1) * block_size;
#pragma omp parallel for reduction(+ : partial_sum)
    for (int i = istart; i < iend; i++)
      partial_sum += array[i];
    block_sum[ib] = partial_sum;
  }

  sum = 0;
  for (int ib = 0; ib < nblock; ib++)
    sum += block_sum[ib];
  if (std::abs(sum - sum_host) > 1e-6) {
    std::cout << "hierarchical parallelism wrong reduction value check" << sum
              << " correct value " << sum_host << std::endl;
    failed = true;
  }
}

template <typename T> void test_complex() {
  test_map<T>();
  test_reduction<std::complex<T>>();
}

int main() {
  std::cout << "Testing complex" << std::endl;
  std::cout << "Testing float" << std::endl;
  test_complex<float>();
  std::cout << "Testing double" << std::endl;
  test_complex<double>();
  return failed;
}
