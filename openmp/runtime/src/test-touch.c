// test-touch.c //


//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#ifdef __cplusplus
extern "C" {
#endif
extern double omp_get_wtime();
extern int    omp_get_num_threads();
extern int    omp_get_max_threads();
#ifdef __cplusplus
}
#endif

int main() {
    omp_get_wtime();
    omp_get_num_threads();
    omp_get_max_threads();
    return 0;
}

// end of file //
