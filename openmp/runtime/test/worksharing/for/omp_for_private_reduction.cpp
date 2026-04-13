// RUN: %libomp-cxx-compile -fopenmp-version=60  && %libomp-run
#include <stdio.h>
#include <omp.h>
#include <limits.h>
#include <complex.h>
#include <math.h>
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

#pragma omp declare reduction(sum_pctor_reduction:Sum : omp_out += omp_in)     \
    initializer(omp_priv = Sum(1)) // non-default ctor

int checkUserDefinedReduction() {
  Sum final_result_udr(0);
  Sum final_result_udr_pctor(1);
  Sum array_sum[N];
  int error_flag = 0;
  int expected_value = 0;
  int expected_value_pctor = 0;
  for (int i = 0; i < N; ++i) {
    array_sum[i] = Sum(i);
    expected_value += i; // Calculate expected sum: 0 + 1 + ... + (N-1)
    expected_value_pctor += i;
  }
  int num_threads_for_pctor_calc = 4; //  num_threads(4)
  int priv_initializer_val_pctor = 1; //  initializer(omp_priv = Sum(1))
  expected_value_pctor +=
      num_threads_for_pctor_calc + priv_initializer_val_pctor;
#pragma omp parallel num_threads(4) private(final_result_udr) private(         \
        final_result_udr_pctor)
  {
#pragma omp for reduction(sum_reduction : final_result_udr)                    \
    reduction(sum_pctor_reduction : final_result_udr_pctor)
    for (int i = 0; i < N; ++i) {
      final_result_udr += array_sum[i];
      final_result_udr_pctor += array_sum[i];
    }

    if (final_result_udr.getValue() != expected_value ||
        final_result_udr_pctor.getValue() != expected_value_pctor)
      error_flag += 1;
  }
  return error_flag;
}
void performMinMaxRed(int &min_val, int &max_val) {
  int input_data[] = {7, 3, 12, 5, 8};
  int n_size = sizeof(input_data) / sizeof(input_data[0]);
  min_val = INT_MAX;
  max_val = INT_MIN;
#pragma omp for reduction(original(private), min : min_val)                    \
    reduction(original(private), max : max_val)
  for (int i = 0; i < n_size; ++i) {
    if (input_data[i] < min_val)
      min_val = input_data[i];
    if (input_data[i] > max_val)
      max_val = input_data[i];
  }
}
std::complex<double> doComplexReduction(std::complex<double> *arr) {
  std::complex<double> result(1, 0);

#pragma omp declare reduction(* : std::complex<double> : omp_out *= omp_in)    \
    initializer(omp_priv = std::complex<double>(1, 0))

#pragma omp for reduction(original(private), * : result)
  for (int i = 0; i < N; ++i)
    result *= arr[i];

  return result;
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
  const float kExpectedFsum = 31.400000f; // 3.14f * 10
  const float kTolerance = 1e-4f;
  const int kExpectedMin = 3;
  const int kExpectedMax = 12;
  std::complex<double> arr[N];
  std::complex<double> kExpectedComplex(1, 0);
  // Initialize the array
  for (int i = 1; i <= N; ++i) {
    arr[i - 1] = std::complex<double>(
        1.0 + 0.1 * i, 0.5 * i); // Avoid zero to prevent multiplication by zero
    kExpectedComplex *= arr[i - 1];
  }

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
    if (std::abs(t_fsum_v - kExpectedFsum) > kTolerance)
      total_errors++;
  }
#pragma omp parallel num_threads(4)
  {
    int t_min_v;
    int t_max_v;
    performMinMaxRed(t_min_v, t_max_v);
    if (t_min_v != kExpectedMin)
      total_errors++;
    if (t_max_v != kExpectedMax)
      total_errors++;
  }
  total_errors += checkUserDefinedReduction();
#pragma omp parallel num_threads(4)
  {
    std::complex<double> result(1, 0);
    result = doComplexReduction(arr);
    if (std::abs(result.real() - kExpectedComplex.real()) > 1e-6 ||
        std::abs(result.imag() - kExpectedComplex.imag()) > 1e-6) {
      total_errors++;
    }
  }
  if (total_errors != 0)
    fprintf(stderr, "ERROR: reduction on private variable  %d\n", total_errors);

  return total_errors;
}
