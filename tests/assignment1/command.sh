clang -O0 -S -emit-llvm 1.c -o 1.ll
opt -load ${LLVM_HOME}/build/lib/LLVMHello.so -enable-new-pm=0 -myhello  1.ll -S -o 2.ll
