clang -O0 -S -emit-llvm cfcss.c -o cfcss.ll
opt -load ${LLVM_HOME}/build/lib/LLVMCfcss.so -enable-new-pm=0 -cfcss  cfcss.ll -S -o out_cfcss.ll
