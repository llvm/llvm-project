clang++ -emit-llvm -S Logger.cpp -std=c++20 -g

sleep 1

python3 AddIntrinsic.py
