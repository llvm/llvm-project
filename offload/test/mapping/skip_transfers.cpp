// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: env LIBOMPTARGET_INFO=128 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic

// REQUIRES: gpu
// clang-format on

int main() {
    float DataStack = 0;

    // CHECK-NOT: omptarget device 0 info: from(unknown)[4] is not used and will not be copied
    #pragma omp target map(from: DataStack)
    {
        DataStack = 1;
    }

    // CHECK-NOT: omptarget device 0 info: to(unknown)[4] is not used and will not be copied
    #pragma omp target map(always to: DataStack)
    ;

    // CHECK: omptarget device 0 info: tofrom(unknown)[4] is not used and will not be copied
    #pragma omp target map(tofrom: DataStack)
    ;

    int Size = 16;
    double *Data = new double[Size];

    // CHECK-NOT: omptarget device 0 info: tofrom(unknown)[8] is not used and will not be copied
    #pragma omp target map(tofrom: Data[0:1])
    {
        Data[0] = 1;
    }

    // CHECK-NOT: omptarget device 0 info: tofrom(unknown)[16] is not used and will not be copied
    #pragma omp target map(always tofrom: Data[0:2])
    ;

    // CHECK: omptarget device 0 info: from(unknown)[24] is not used and will not be copied
    #pragma omp target map(from: Data[0:3])
    ;

    // CHECK: omptarget device 0 info: to(unknown)[24] is not used and will not be copied
    #pragma omp target map(to: Data[0:3])
    ;

    // CHECK: omptarget device 0 info: tofrom(unknown)[32] is not used and will not be copied
    #pragma omp target map(tofrom: Data[0:4])
    ;

    // CHECK-NOT: omptarget device 0 info: to(unknown)[40] is not used and will not be copied
    #pragma omp target map(to: Data[0:5])
    {
        #pragma omp teams
        Data[0] = 1;
    }

    struct {
      double *Data;
    } Wrapper { .Data = Data };

    // CHECK-NOT: omptarget device 0 info: tofrom(unknown)[48] is not used and will not be copied
    #pragma omp target map(tofrom: Wrapper.Data[0:6])
    {
        Wrapper.Data[0] = 1;
    }

    // CHECK: omptarget device 0 info: unknown(unknown)[8] is not used and will not be copied
    // CHECK: omptarget device 0 info: tofrom(unknown)[56] is not used and will not be copied
    #pragma omp target map(tofrom: Wrapper.Data[0:7])
    ;
}
