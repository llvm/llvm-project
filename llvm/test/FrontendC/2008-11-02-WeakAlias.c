// RUN: %llvmgcc -S -emit-llvm -o - %s | grep weak
// PR2691

void init_IRQ(void) __attribute__((weak, alias("native_init_IRQ")));
void native_init_IRQ(void) {}