// RUN: %libomp-cxx-compile -fopenmp-version=60  && %libomp-run
#include <stdio.h>
#include <omp.h>
#include "omp_testsuite.h"

#define N 10
class Sum {
  int val;

public:
  Sum(int v = 0) : val(v) {}
  Sum operator+(const Sum &rhs) const { return Sum(val + rhs.val); }
  Sum &operator+=(const Sum &rhs) {
    val += rhs.val;
    return *this;
  }
  int getValue() const { return val; }
};

// Declare OpenMP reduction
#pragma omp declare reduction(sum_reduction:Sum : omp_out += omp_in)           \
    initializer(omp_priv = Sum(0))

int checkUserDefinedReduction() {
  Sum final_result_udr(0);
  Sum array_sum[N];
  int error_flag = 0;
  int expected_value = 0;
  for (int i = 0; i < N; ++i) {
    array_sum[i] = Sum(i);
    expected_value += i; // Calculate expected sum: 0 + 1 + ... + (N-1)
  }
#pragma omp parallel num_threads(4) private(final_result_udr)
  {
#pragma omp for reduction(sum_reduction : final_result_udr)
    for (int i = 0; i < N; ++i) {
      final_result_udr += array_sum[i];
    }

    if (final_result_udr.getValue() != expected_value)
      error_flag += 1;
  }
  return error_flag;
}

void performReductions(int n_elements, const int *input_values,
                       int &sum_val_out, int &prod_val_out,
                       float &float_sum_val_out) {
  // private variables for this thread's reduction.
  sum_val_out = 0;
  prod_val_out = 1;
  float_sum_val_out = 0.0f;

  const float kPiValue = 3.14f;
#pragma omp for reduction(original(private), + : sum_val_out)                  \
    reduction(original(private), * : prod_val_out)                             \
    reduction(original(private), + : float_sum_val_out)
  for (int i = 0; i < n_elements; ++i) {
    sum_val_out += input_values[i];
    prod_val_out *= (i + 1);
    float_sum_val_out += kPiValue;
  }
}
int main(void) {
  int input_array[N];
  int total_errors = 0;
  const float kPiVal = 3.14f;
  const int kExpectedSum = 45; // Sum of 0..9
  const int kExpectedProd = 3628800; // 10!
  const float kExpectedFsum = kPiVal * N; // 3.14f * 10

  for (int i = 0; i < N; i++)
    input_array[i] = i;
#pragma omp parallel num_threads(4)
  {

    int t_sum_v;
    int t_prod_v;
    float t_fsum_v;
    performReductions(N, input_array, t_sum_v, t_prod_v, t_fsum_v);
    if (t_sum_v != kExpectedSum)
      total_errors++;
    if (t_prod_v != kExpectedProd)
      total_errors++;
    if (t_fsum_v != kExpectedFsum)
      total_errors++;
  }
  total_errors += checkUserDefinedReduction();
  if (total_errors != 0)
    fprintf(stderr, "ERROR: reduction on private variable  %d\n", total_errors);

  return total_errors;
}
