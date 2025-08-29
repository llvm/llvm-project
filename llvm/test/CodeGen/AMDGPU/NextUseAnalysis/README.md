# AMDGPU NextUseAnalysis Tests

This directory contains comprehensive tests for the AMDGPU NextUseAnalysis V2 implementation.

## Running Tests

### Individual Test
```bash
cd build/Debug
./bin/llc -mtriple=amdgcn -mcpu=gfx900 -run-pass=amdgpu-next-use -debug-only=amdgpu-next-use \
    ../../llvm/test/CodeGen/AMDGPU/NextUseAnalysis/basic-distances.mir -o /dev/null 2>&1 | \
    ./bin/FileCheck ../../llvm/test/CodeGen/AMDGPU/NextUseAnalysis/basic-distances.mir
```

### All Tests  
```bash
cd build/Debug
for test in ../../llvm/test/CodeGen/AMDGPU/NextUseAnalysis/*.mir; do
    echo "Testing: $test"
    ./bin/llc -mtriple=amdgcn -mcpu=gfx900 -run-pass=amdgpu-next-use -debug-only=amdgpu-next-use \
        "$test" -o /dev/null 2>&1 | ./bin/FileCheck "$test" && echo "PASS" || echo "FAIL"
done
```

## Test Categories

1. **basic-distances.mir** - Fundamental distance calculations
2. **subreg-distances.mir** - Sub-register handling  
3. **multiblock-distances.mir** - Control flow analysis
4. **dead-registers.mir** - Dead register detection
5. **subreg-interference.mir** - Advanced sub-register interference

All tests validate the V2 implementation's sub-register aware analysis capabilities.
