// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

typedef struct { unsigned long pgprot; } pgprot_t;

void split_large_page(unsigned long addr, pgprot_t prot)
{
  (addr ? prot : ((pgprot_t) { 0x001 } )).pgprot;
}

