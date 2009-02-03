// RUN: %llvmgcc -x objective-c -S %s -fobjc-gc -o - | grep objc_assign_strongCast
// rdar://5541393

typedef struct {
    void (^ivarBlock)(void);
} StructWithBlock_t;

int main(char *argc, char *argv[]) {
   StructWithBlock_t *swbp = (StructWithBlock_t *)malloc(sizeof(StructWithBlock_t*));
   __block   int i = 10;
   // assigning a Block into an struct slot should elicit a write-barrier under GC
   swbp->ivarBlock = ^ { ++i; }; 
   return 0;
}
