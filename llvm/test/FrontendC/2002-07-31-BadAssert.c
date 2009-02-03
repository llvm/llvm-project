// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

typedef struct
{
        unsigned char type;        /* Indicates, NORMAL, SUBNORMAL, etc. */
} InternalFPF;


static void SetInternalFPFZero(InternalFPF *dest) {
  dest->type=0;
}

void denormalize(InternalFPF *ptr) {
   SetInternalFPFZero(ptr);
}

