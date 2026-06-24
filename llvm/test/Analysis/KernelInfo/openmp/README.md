The tests in this directory check that basic KernelInfoPrinter functionality
behaves reasonably for LLVM IR produced by Clang OpenMP codegen.

So that these tests are straightforward to maintain and faithfully represent
Clang OpenMP codegen, do not tweak or reduce the LLVM IR in them.  Other tests
more exhaustively check KernelInfoPrinter features using reduced LLVM IR.

The LLVM IR in each test file `$TEST` can be regenerated as follows in the case
that Clang OpenMP codegen changes or it becomes desirable to adjust the source
OpenMP program below.  First, remove the existing LLVM IR from `$TEST`.  Then,
where `$TARGET` (e.g., `nvptx64-nvidia-cuda-sm_70` or `amdgcn-amd-amdhsa-gfx906`)
depends on `$TEST`:

```
$ cd /tmp
$ cat test.c
#pragma omp declare target
void f();
void g() {
  int i;
  int a[2];
  f();
  g();
}
#pragma omp end declare target

void h(int i) {
  #pragma omp target map(tofrom:i)
  {
    int i;
    int a[2];
    f();
    g();
  }
}

$ clang -g -fopenmp --offload-arch=native -save-temps -c test.c
$ llvm-dis test-openmp-$TARGET.bc
$ cat test-openmp-$TARGET.ll >> $TEST
```
