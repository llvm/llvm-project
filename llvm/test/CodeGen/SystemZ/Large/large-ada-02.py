# Test code generation for retrieving function descriptors
# from the ADA when the ADA is extremely large and forces the
# generation of a different instruction sequence
# RUN: %python %s | llc -mtriple=s390x-ibm-zos -O2 | FileCheck %s

# CHECK: algfi	8,{{[0-9]+}}
# CHECK: la	8,0(8)

from __future__ import print_function

num_calls = 35000

print("define hidden signext i32 @main() {")
print("entry:")

for i in range(num_calls):
    print("  call void @foo%d()" % i)

# This is added to force the use of register r8 to generate
# la 8, 0(8) which generates the algfi instruction
print('%0 = call ptr asm " LGR $0,$1\0A", "=r,{r8}"(ptr nonnull @foo)')
print("  call void @bar(ptr noundef %0)")
print("ret i32 0")
print("}")

for i in range(num_calls):
    print("declare void @foo%d(...)" % i)

print("declare void @bar(ptr noundef)")
print("define internal void @foo() {")
print("entry:")
print("  ret void")
print("  }")
