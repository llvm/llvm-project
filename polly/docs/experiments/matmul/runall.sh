#!/bin/sh -a

echo "--> 1. Create LLVM-IR from C"
clang -S -emit-llvm matmul.c -Xclang -disable-O0-optnone -o matmul.ll

echo "--> 2. Prepare the LLVM-IR for Polly"
opt -S -polly-canonicalize matmul.ll -o matmul.preopt.ll

echo "--> 3. Show the SCoPs detected by Polly"
opt -basicaa -polly-ast -analyze matmul.preopt.ll \
    -polly-process-unprofitable -polly-use-llvm-names

echo "--> 4.1 Highlight the detected SCoPs in the CFGs of the program"
# We only create .dot files, as directly -view-scops directly calls graphviz
# which would require user interaction to continue the script.
# opt -basicaa -view-scops -disable-output matmul.preopt.ll
opt -basicaa -dot-scops -disable-output matmul.preopt.ll -polly-use-llvm-names

echo "--> 4.2 Highlight the detected SCoPs in the CFGs of the program (print \
no instructions)"
# We only create .dot files, as directly -view-scops-only directly calls
# graphviz which would require user interaction to continue the script.
# opt -basicaa -view-scops-only -disable-output matmul.preopt.ll
opt -basicaa -dot-scops-only -disable-output matmul.preopt.ll -polly-use-llvm-names

echo "--> 4.3 Create .png files from the .dot files"
for i in `ls *.dot`; do dot -Tpng $i > $i.png; done

echo "--> 5. View the polyhedral representation of the SCoPs"
opt -basicaa -polly-scops -analyze matmul.preopt.ll \
    -polly-process-unprofitable -polly-use-llvm-names

echo "--> 6. Show the dependences for the SCoPs"
opt -basicaa -polly-dependences -analyze matmul.preopt.ll \
    -polly-process-unprofitable -polly-use-llvm-names

echo "--> 7. Export jscop files"
opt -basicaa -polly-export-jscop matmul.preopt.ll \
    -polly-process-unprofitable -disable-output -polly-use-llvm-names

echo "--> 8. Import the updated jscop files and print the new SCoPs. (optional)"
opt -basicaa -polly-import-jscop -polly-ast -analyze matmul.preopt.ll \
    -polly-process-unprofitable -polly-use-llvm-names
opt -basicaa -polly-import-jscop -polly-ast -analyze matmul.preopt.ll \
    -polly-import-jscop-postfix=interchanged -polly-process-unprofitable -polly-use-llvm-names
opt -basicaa -polly-import-jscop -polly-ast -analyze matmul.preopt.ll \
    -polly-import-jscop-postfix=interchanged+tiled -polly-process-unprofitable -polly-use-llvm-names
opt -basicaa -polly-import-jscop -polly-ast -analyze matmul.preopt.ll \
    -polly-import-jscop-postfix=interchanged+tiled+vector \
    -polly-process-unprofitable -polly-use-llvm-names

echo "--> 9. Codegenerate the SCoPs"
opt -S -basicaa -polly-import-jscop -polly-import-jscop-postfix=interchanged \
    -polly-codegen -polly-process-unprofitable -polly-use-llvm-names \
    matmul.preopt.ll | opt -O3 -S -o matmul.polly.interchanged.ll
opt -S -basicaa -polly-import-jscop \
    -polly-import-jscop-postfix=interchanged+tiled -polly-codegen \
    matmul.preopt.ll -polly-process-unprofitable -polly-use-llvm-names \
    | opt -O3 -S -o matmul.polly.interchanged+tiled.ll
opt -S -basicaa -polly-import-jscop -polly-process-unprofitable\
    -polly-import-jscop-postfix=interchanged+tiled+vector -polly-codegen \
    matmul.preopt.ll -polly-vectorizer=polly -polly-use-llvm-names \
    | opt -O3 -S -o matmul.polly.interchanged+tiled+vector.ll
opt -S -basicaa -polly-import-jscop -polly-process-unprofitable\
    -polly-import-jscop-postfix=interchanged+tiled+vector -polly-codegen \
    matmul.preopt.ll -polly-vectorizer=polly -polly-parallel -polly-use-llvm-names \
    | opt -O3 -S -o matmul.polly.interchanged+tiled+vector+openmp.ll
opt -S matmul.preopt.ll | opt -O3 -S -o matmul.normalopt.ll

echo "--> 10. Create the executables"
llc matmul.polly.interchanged.ll -o matmul.polly.interchanged.s -relocation-model=pic
gcc matmul.polly.interchanged.s -o matmul.polly.interchanged.exe
llc matmul.polly.interchanged+tiled.ll -o matmul.polly.interchanged+tiled.s -relocation-model=pic
gcc matmul.polly.interchanged+tiled.s -o matmul.polly.interchanged+tiled.exe
llc matmul.polly.interchanged+tiled+vector.ll -o matmul.polly.interchanged+tiled+vector.s -relocation-model=pic
gcc matmul.polly.interchanged+tiled+vector.s  -o matmul.polly.interchanged+tiled+vector.exe
llc matmul.polly.interchanged+tiled+vector+openmp.ll -o matmul.polly.interchanged+tiled+vector+openmp.s -relocation-model=pic
gcc matmul.polly.interchanged+tiled+vector+openmp.s -lgomp -o matmul.polly.interchanged+tiled+vector+openmp.exe
llc matmul.normalopt.ll -o matmul.normalopt.s -relocation-model=pic
gcc matmul.normalopt.s -lgomp -o matmul.normalopt.exe

echo "--> 11. Compare the runtime of the executables"

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
