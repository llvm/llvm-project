// Verify that the %c modifier works and strips off any prefixes from 
// immediates.
// RUN: %llvmgcc -S %s -o - | llvm-as | llc | grep {pickANumber: 789514}

void foo() {
  __asm__         volatile("/* " "pickANumber" ": %c0 */"::"i"(0xC0C0A));
  
  // Check that non-c modifiers work also (not greped for above).
   __asm__         volatile("/* " "pickANumber2 " ": %0 */"::"i"(123));
}
