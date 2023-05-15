clang++ -emit-llvm -S Logger.cpp

sleep 1

python3 AddIntrinsic.py
