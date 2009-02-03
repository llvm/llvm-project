// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

/* GCC is not outputting the static array to the LLVM backend, so bad things
 * happen.  Note that if this is defined static, everything seems fine.
 */
double test(unsigned X) {
  double student_t[30]={0.0 , 12.706 , 4.303 , 3.182 , 2.776 , 2.571 ,
                               2.447 , 2.365 , 2.306 , 2.262 , 2.228 ,
                               2.201 , 2.179 , 2.160 , 2.145 , 2.131 ,
                               2.120 , 2.110 , 2.101 , 2.093 , 2.086 ,
                               2.080 , 2.074 , 2.069 , 2.064 , 2.060 ,
                               2.056 , 2.052 , 2.048 , 2.045 };
  return student_t[X];
}
