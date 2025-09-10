int global_var = 0;

int side_effect_func(int x) {
  global_var++;               // 副作用
  return x * 2;
}

void function_dep(int *A, int n) {
  for (int i = 0; i < n; ++i) {
    A[i] = side_effect_func(A[i]);  // 函數有副作用，產生相依
  }
}
