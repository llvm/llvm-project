// RUN: %llvmgcc -xc++ %s -S -o /dev/null |& not grep warning
// rdar://5683899
void** f(void **Buckets, unsigned NumBuckets) {
  return Buckets + NumBuckets;
}

