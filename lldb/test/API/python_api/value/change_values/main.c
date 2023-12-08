#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

struct foo
{
  uint8_t   first_val;
  uint32_t  second_val;
  uint64_t  third_val;
};
  
int main ()
{
  int val = 100;
  struct foo mine = {55, 5555, 55555555};
  struct foo *ptr = (struct foo *) malloc (sizeof (struct foo));
  ptr->first_val = 66;
  ptr->second_val = 6666;
  ptr->third_val = 66666666;

  // Stop here and set values
  printf ("Val - %d Mine - %d, %d, %llu. Ptr - %d, %d, %llu\n", val, 
          mine.first_val, mine.second_val, mine.third_val,
          ptr->first_val, ptr->second_val, ptr->third_val); 

  // Stop here and check values
  printf ("This is just another call which we won't make it over %d.", val);
  return 0; // Set a breakpoint here at the end
}
