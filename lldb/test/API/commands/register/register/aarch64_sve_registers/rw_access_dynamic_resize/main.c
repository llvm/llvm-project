#include <pthread.h>
#include <stdbool.h>
#include <sys/prctl.h>

// If USE_SSVE is defined, this program will use streaming mode SVE registers
// instead of non-streaming mode SVE registers.

#ifndef PR_SME_SET_VL
#define PR_SME_SET_VL 63
#endif

#define SMSTART() asm volatile("msr  s0_3_c4_c7_3, xzr" /*smstart*/)

static inline void write_sve_registers() {
  asm volatile("setffr\n\t");
  asm volatile("ptrue p0.b\n\t");
  asm volatile("ptrue p1.h\n\t");
  asm volatile("ptrue p2.s\n\t");
  asm volatile("ptrue p3.d\n\t");
  asm volatile("pfalse p4.b\n\t");
  asm volatile("ptrue p5.b\n\t");
  asm volatile("ptrue p6.h\n\t");
  asm volatile("ptrue p7.s\n\t");
  asm volatile("ptrue p8.d\n\t");
  asm volatile("pfalse p9.b\n\t");
  asm volatile("ptrue p10.b\n\t");
  asm volatile("ptrue p11.h\n\t");
  asm volatile("ptrue p12.s\n\t");
  asm volatile("ptrue p13.d\n\t");
  asm volatile("pfalse p14.b\n\t");
  asm volatile("ptrue p15.b\n\t");

  asm volatile("cpy  z0.b, p0/z, #1\n\t");
  asm volatile("cpy  z1.b, p5/z, #2\n\t");
  asm volatile("cpy  z2.b, p10/z, #3\n\t");
  asm volatile("cpy  z3.b, p15/z, #4\n\t");
  asm volatile("cpy  z4.b, p0/z, #5\n\t");
  asm volatile("cpy  z5.b, p5/z, #6\n\t");
  asm volatile("cpy  z6.b, p10/z, #7\n\t");
  asm volatile("cpy  z7.b, p15/z, #8\n\t");
  asm volatile("cpy  z8.b, p0/z, #9\n\t");
  asm volatile("cpy  z9.b, p5/z, #10\n\t");
  asm volatile("cpy  z10.b, p10/z, #11\n\t");
  asm volatile("cpy  z11.b, p15/z, #12\n\t");
  asm volatile("cpy  z12.b, p0/z, #13\n\t");
  asm volatile("cpy  z13.b, p5/z, #14\n\t");
  asm volatile("cpy  z14.b, p10/z, #15\n\t");
  asm volatile("cpy  z15.b, p15/z, #16\n\t");
  asm volatile("cpy  z16.b, p0/z, #17\n\t");
  asm volatile("cpy  z17.b, p5/z, #18\n\t");
  asm volatile("cpy  z18.b, p10/z, #19\n\t");
  asm volatile("cpy  z19.b, p15/z, #20\n\t");
  asm volatile("cpy  z20.b, p0/z, #21\n\t");
  asm volatile("cpy  z21.b, p5/z, #22\n\t");
  asm volatile("cpy  z22.b, p10/z, #23\n\t");
  asm volatile("cpy  z23.b, p15/z, #24\n\t");
  asm volatile("cpy  z24.b, p0/z, #25\n\t");
  asm volatile("cpy  z25.b, p5/z, #26\n\t");
  asm volatile("cpy  z26.b, p10/z, #27\n\t");
  asm volatile("cpy  z27.b, p15/z, #28\n\t");
  asm volatile("cpy  z28.b, p0/z, #29\n\t");
  asm volatile("cpy  z29.b, p5/z, #30\n\t");
  asm volatile("cpy  z30.b, p10/z, #31\n\t");
  asm volatile("cpy  z31.b, p15/z, #32\n\t");
}

int SET_VL_OPT = PR_SVE_SET_VL;

// These ensure that when lldb stops in one of threadX / threadY, the other has
// at least been created. That means we can continue the other onto the expected
// breakpoint. Otherwise we could get to the breakpoint in one thread before the
// other has started.
volatile bool threadX_ready = false;
volatile bool threadY_ready = false;

void *threadX_func(void *x_arg) {
  threadX_ready = true;
  while (!threadY_ready) {
  }

  prctl(SET_VL_OPT, 8 * 4);
#ifdef USE_SSVE
  SMSTART();
#endif
  write_sve_registers();
  write_sve_registers(); // Thread X breakpoint 1
  return NULL;           // Thread X breakpoint 2
}

void *threadY_func(void *y_arg) {
  threadY_ready = true;
  while (!threadX_ready) {
  }

  prctl(SET_VL_OPT, 8 * 2);
#ifdef USE_SSVE
  SMSTART();
#endif
  write_sve_registers();
  write_sve_registers(); // Thread Y breakpoint 1
  return NULL;           // Thread Y breakpoint 2
}

int main() {
#ifdef USE_SSVE
  SET_VL_OPT = PR_SME_SET_VL;
#endif

  /* this variable is our reference to the second thread */
  pthread_t x_thread, y_thread;

  /* Set vector length to 8 and write SVE registers values */
  prctl(SET_VL_OPT, 8 * 8);
#ifdef USE_SSVE
  SMSTART();
#endif
  write_sve_registers();

  /* create a second thread which executes with argument x */
  if (pthread_create(&x_thread, NULL, threadX_func, 0)) // Break in main thread
    return 1;

  /* create a second thread which executes with argument y */
  if (pthread_create(&y_thread, NULL, threadY_func, 0))
    return 1;

  /* wait for the first thread to finish */
  if (pthread_join(x_thread, NULL))
    return 2;

  /* wait for the second thread to finish */
  if (pthread_join(y_thread, NULL))
    return 2;

  return 0;
}
