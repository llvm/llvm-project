// RUN: %llvmgxx %s -emit-llvm -S -o -

#pragma reverse_bitfields on
typedef unsigned long UINT32;

extern void abort(void);

typedef struct TestStruct
{
  long	first: 15,
    second: 17;	
} TestStruct;

int main (int argc, char * const argv[]) {

  TestStruct testStruct = {1, 0};
  
  UINT32 dw = *(UINT32 *)(&testStruct);
  
  if(!(dw & 0xFFFF))
    abort ();

  return 0;
}
