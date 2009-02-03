// RUN: %llvmgxx -S %s -o - | llvm-as -f -o /dev/null


#ifdef PACKED
// This is an example where size of Packed struct is smaller then 
// the size of bit field type.
#define P __attribute__((packed))
#else
#define P
#endif

struct P M_Packed { 
  unsigned long long X:50;
  unsigned Y:2;
}; 

struct M_Packed sM_Packed; 

int testM_Packed (void) { 
  struct M_Packed x; 
  return (0 != x.Y);
}
      
int testM_Packed2 (void) { 
  struct M_Packed x; 
  return (0 != x.X);
}
