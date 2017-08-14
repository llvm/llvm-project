// Coverage/profile data recycled from the showLineExecutionCounts.cpp test.
//
// RUN: llvm-profdata merge %S/Inputs/lineExecutionCounts.proftext -o %t.profdata
// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -j 1 -o %t1.dir -instr-profile %t.profdata -path-equivalence=/tmp,%S %S/showLineExecutionCounts.cpp
// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -num-threads 2 -o %t2.dir -instr-profile %t.profdata -path-equivalence=/tmp,%S %S/showLineExecutionCounts.cpp
// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -o %t3.dir -instr-profile %t.profdata -path-equivalence=/tmp,%S %S/showLineExecutionCounts.cpp
//
// RUN: diff %t1.dir/index.txt %t2.dir/index.txt
// RUN: diff %t1.dir/coverage/tmp/showLineExecutionCounts.cpp.txt %t2.dir/coverage/tmp/showLineExecutionCounts.cpp.txt
// RUN: diff %t1.dir/index.txt %t3.dir/index.txt
// RUN: diff %t1.dir/coverage/tmp/showLineExecutionCounts.cpp.txt %t3.dir/coverage/tmp/showLineExecutionCounts.cpp.txt
