// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>

int
main(int argc, char** argv)
{
  char     c1;
  short    s1, ssf1, ssd1;
  unsigned char  ubs0;
  signed char   bs0;
  unsigned char ubc0, uc2;
  unsigned short us2, usf1, usd1;
  int ic3, is3, sif1, sid1;
  unsigned int     uic4, uis4, uif1, uid1;
  long     slf1, sld1;
  unsigned long    ulf1, uld1;
  float    f1;
  double   d1;
  
  /* Test integer to integer conversions */
  
  c1 = (char)  (argc >= 2)? atoi(argv[1]) : 0xff64; /* 100 = 'd' */
  s1 = (short) (argc >= 3)? atoi(argv[2]) : -769;   /* 0xf7ff = -769 */
  
  ubc0 = (unsigned char) c1;                      /* 100 = 'd' */
  ubs0 = (unsigned char) s1;                            /* 0xff = 255 */
  bs0  = (signed char) s1;                             /* 0xff = -1 */
  
  uc2 = (unsigned char) c1;                       /* 100 = 'd' */
  us2 = (unsigned short) s1;                      /* 0xf7ff = 64767 */
  
  ic3 = (int) c1;                                 /* 100 = 'd' */
  is3 = (int) s1;                                 /* 0xfffff7ff = -769 */
  
  uic4 = (unsigned int) c1;                       /*  100 = 'd' */
  uis4 = (unsigned int) s1;                       /* 0xfffff7ff = 4294966527 */
  
  printf("ubc0 = '%c'\n", ubc0);
  printf("ubs0 = %u\n",   ubs0);
  printf("bs0  = %d\n",   bs0);
  printf("c1   = '%c'\n", c1);
  printf("s1   = %d\n",   s1);
  printf("uc2  = '%c'\n", uc2);
  printf("us2  = %u\n",   us2);
  printf("ic3  = '%c'\n", ic3);
  printf("is3  = %d\n",   is3);
  printf("uic4 = '%c'\n", uic4);
  printf("uis4 = %u\n",   uis4);
  
  /* Test floating-point to integer conversions */
  f1 = (float)  (argc >= 4)? atof(argv[3]) : 1.0;
  d1 =          (argc >= 5)? atof(argv[4]) : 2.0;
  
  usf1 = (unsigned short) f1;
  usd1 = (unsigned short) d1;
  uif1 = (unsigned int) f1;
  uid1 = (unsigned int) d1;
  ulf1 = (unsigned long) f1;
  uld1 = (unsigned long) d1;
  
  ssf1 = (short) f1;
  ssd1 = (short) d1;
  sif1 = (int) f1;
  sid1 = (int) d1;
  slf1 = (long) f1;
  sld1 = (long) d1;
  
  printf("usf1 = %u\n", usf1);
  printf("usd1 = %u\n", usd1);
  printf("uif1 = %u\n", uif1);
  printf("uid1 = %u\n", uid1);
  printf("ulf1 = %u\n", ulf1);
  printf("uld1 = %u\n", uld1);
  
  printf("ssf1 = %d\n", ssf1);
  printf("ssd1 = %d\n", ssd1);
  printf("sif1 = %d\n", sif1);
  printf("sid1 = %d\n", sid1);
  printf("slf1 = %d\n", slf1);
  printf("sld1 = %d\n", sld1);
  
  return 0;
}
