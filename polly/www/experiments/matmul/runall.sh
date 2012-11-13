#!/bin/sh -a

echo "--> 1. Create LLVM-IR from C"
clang -S -emit-llvm matmul.c -o matmul.s

echo "--> 2. Load Polly automatically when calling the 'opt' tool"
export PATH_TO_POLLY_LIB="~/polly/build/lib/"
alias opt="opt -load ${PATH_TO_POLLY_LIB}/LLVMPolly.so"

echo "--> 3. Prepare the LLVM-IR for Polly"
opt -S -mem2reg -loop-simplify -polly-indvars matmul.s > matmul.preopt.ll

echo "--> 4. Show the SCoPs detected by Polly"
opt -basicaa -polly-cloog -analyze -q matmul.preopt.ll

echo "--> 5.1 Highlight the detected SCoPs in the CFGs of the program"
# We only create .dot files, as directly -view-scops directly calls graphviz
# which would require user interaction to continue the script.
# opt -basicaa -view-scops -disable-output matmul.preopt.ll
opt -basicaa -dot-scops -disable-output matmul.preopt.ll

echo "--> 5.2 Highlight the detected SCoPs in the CFGs of the program (print \
no instructions)"
# We only create .dot files, as directly -view-scops-only directly calls
# graphviz which would require user interaction to continue the script.
# opt -basicaa -view-scops-only -disable-output matmul.preopt.ll
opt -basicaa -dot-scops-only -disable-output matmul.preopt.ll

echo "--> 5.3 Create .png files from the .dot files"
for i in `ls *.dot`; do dot -Tpng $i > $i.png; done

echo "--> 6. View the polyhedral representation of the SCoPs"
opt -basicaa -polly-scops -analyze matmul.preopt.ll

echo "--> 7. Show the dependences for the SCoPs"
opt -basicaa -polly-dependences -analyze matmul.preopt.ll

echo "--> 8. Export jscop files"
opt -basicaa -polly-export-jscop matmul.preopt.ll

echo "--> 9. Import the updated jscop files and print the new SCoPs. (optional)"
opt -basicaa -polly-import-jscop -polly-cloog -analyze matmul.preopt.ll
opt -basicaa -polly-import-jscop -polly-cloog -analyze matmul.preopt.ll \
    -polly-import-jscop-postfix=interchanged
opt -basicaa -polly-import-jscop -polly-cloog -analyze matmul.preopt.ll \
    -polly-import-jscop-postfix=interchanged+tiled
opt -basicaa -polly-import-jscop -polly-cloog -analyze matmul.preopt.ll \
    -polly-import-jscop-postfix=interchanged+tiled+vector

echo "--> 10. Codegenerate the SCoPs"
opt -basicaa -polly-import-jscop -polly-import-jscop-postfix=interchanged \
    -polly-codegen \
    matmul.preopt.ll | opt -O3 > matmul.polly.interchanged.ll
opt -basicaa -polly-import-jscop \
    -polly-import-jscop-postfix=interchanged+tiled -polly-codegen \
    matmul.preopt.ll | opt -O3 > matmul.polly.interchanged+tiled.ll
opt -basicaa -polly-import-jscop \
    -polly-import-jscop-postfix=interchanged+tiled+vector -polly-codegen \
    matmul.preopt.ll -polly-vectorizer=polly\
    | opt -O3 > matmul.polly.interchanged+tiled+vector.ll
opt -basicaa -polly-import-jscop \
    -polly-import-jscop-postfix=interchanged+tiled+vector -polly-codegen \
    matmul.preopt.ll -polly-vectorizer=polly -enable-polly-openmp\
    | opt -O3 > matmul.polly.interchanged+tiled+vector+openmp.ll
opt matmul.preopt.ll | opt -O3 > matmul.normalopt.ll

echo "--> 11. Create the executables"
llc matmul.polly.interchanged.ll -o matmul.polly.interchanged.s && gcc matmul.polly.interchanged.s \
    -o matmul.polly.interchanged.exe
llc matmul.polly.interchanged+tiled.ll -o matmul.polly.interchanged+tiled.s && gcc matmul.polly.interchanged+tiled.s \
    -o matmul.polly.interchanged+tiled.exe
llc matmul.polly.interchanged+tiled+vector.ll \
    -o matmul.polly.interchanged+tiled+vector.s \
    && gcc matmul.polly.interchanged+tiled+vector.s \
    -o matmul.polly.interchanged+tiled+vector.exe
llc matmul.polly.interchanged+tiled+vector+openmp.ll \
    -o matmul.polly.interchanged+tiled+vector+openmp.s \
    && gcc -lgomp matmul.polly.interchanged+tiled+vector+openmp.s \
    -o matmul.polly.interchanged+tiled+vector+openmp.exe
llc matmul.normalopt.ll -o matmul.normalopt.s && gcc matmul.normalopt.s \
    -o matmul.normalopt.exe

echo "--> 12. Compare the runtime of the executables"

echo "time ./matmul.normalopt.exe"
time -f "%E real, %U user, %S sys" ./matmul.normalopt.exe
echo "time ./matmul.polly.interchanged.exe"
time -f "%E real, %U user, %S sys" ./matmul.polly.interchanged.exe
echo "time ./matmul.polly.interchanged+tiled.exe"
time -f "%E real, %U user, %S sys" ./matmul.polly.interchanged+tiled.exe
echo "time ./matmul.polly.interchanged+tiled+vector.exe"
time -f "%E real, %U user, %S sys" ./matmul.polly.interchanged+tiled+vector.exe
echo "time ./matmul.polly.interchanged+tiled+vector+openmp.exe"
time -f "%E real, %U user, %S sys" ./matmul.polly.interchanged+tiled+vector+openmp.exe
