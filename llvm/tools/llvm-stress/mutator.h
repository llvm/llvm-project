#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif
void createISelMutator();
size_t LLVMFuzzerCustomMutator(uint8_t *Data, size_t Size, size_t MaxSize,
                               unsigned int Seed);

#ifdef __cplusplus
}
#endif