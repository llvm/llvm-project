int main(int argc, char** argv) {
  #pragma omp TGT1_DIRECTIVE
  {
#ifdef LOOP_DIRECTIVE
    #pragma omp LOOP_DIRECTIVE
    for (int i = 0; i < argc; ++i)
#endif
    {
#ifdef BODY_DIRECTIVE
      #pragma omp BODY_DIRECTIVE
      {
      }
#endif
    }
  }

#ifdef TGT2_DIRECTIVE
#pragma omp TGT2_DIRECTIVE
  {
#ifdef LOOP_DIRECTIVE
    #pragma omp LOOP_DIRECTIVE
    for (int i = 0; i < argc; ++i)
#endif
    {
#ifdef BODY_DIRECTIVE
      #pragma omp BODY_DIRECTIVE
      {
      }
#endif
    }
  }
#endif
}

// Check for an empty kernel (IR level)
// FIRST:       define weak_odr {{.*}} void @__omp_offloading_{{.*}}_main{{.*}}
// FIRST-NEXT:    ret void

// Check for two empty kernels (IR level)
// SECOND:      define weak_odr {{.*}} void @__omp_offloading_{{.*}}_main{{.*}}
// SECOND-NEXT:   ret void
