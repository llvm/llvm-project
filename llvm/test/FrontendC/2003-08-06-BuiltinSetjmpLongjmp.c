/* RUN: %llvmgcc -xc %s -c -o - | llvm-dis | not grep __builtin_
 *
 * __builtin_longjmp/setjmp should get transformed into llvm.setjmp/longjmp 
 * just like explicit setjmp/longjmp calls are.
 */

void jumpaway(int *ptr) {
  __builtin_longjmp(ptr,1);
}
    
int main(void) {
  __builtin_setjmp(0);
  jumpaway(0);
}
