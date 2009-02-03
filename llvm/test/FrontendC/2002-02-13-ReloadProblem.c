// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

/* This triggered a problem in reload, fixed by disabling most of the 
 * steps of compilation in GCC.  Before this change, the code went through
 * the entire backend of GCC, even though it was unnecessary for LLVM output
 * now it is skipped entirely, and since reload doesn't run, it can't cause
 * a problem.
 */

extern int tolower(int);

const char *rangematch(const char *pattern, int test, int c) {

  if ((c <= test) | (tolower(c) <= tolower((unsigned char)test)))
    return 0;

  return pattern;
}
