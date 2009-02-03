// RUN: %llvmgcc -x objective-c -c %s -o /dev/null -fobjc-gc
// rdar://5541393

typedef unsigned int NSUInteger;
__attribute__((objc_gc(strong))) float *_scores;

void foo(int i, float f) {
  _scores[i] = f; 
}
