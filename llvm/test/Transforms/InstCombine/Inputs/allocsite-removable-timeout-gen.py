#!/usr/bin/env python3
"""Generate IR with a large non-removable alloca to stress-test
isAllocSiteRemovable compile time."""

import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12000

print("declare void @escape(ptr)")
print()
print("define i32 @stress() {")
print("entry:")
print(f"  %a = alloca [{N} x i8], align 1")
print("  %sum0 = add i32 0, 0")
for i in range(N):
    print(f"  %p{i} = getelementptr inbounds [{N} x i8], ptr %a, i64 0, i64 {i}")
    print(f"  %v{i} = load i8, ptr %p{i}, align 1")
    print(f"  %z{i} = zext i8 %v{i} to i32")
    print(f"  %sum{i+1} = add i32 %sum{i}, %z{i}")
print("  call void @escape(ptr %a)")
print(f"  ret i32 %sum{N}")
print("}")
