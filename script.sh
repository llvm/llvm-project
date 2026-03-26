#!/bin/bash
for std in c++03 c++11 c++14 c++17 c++20 c++23 c++26; do python3 ./build/generic-cxx26/bin/llvm-lit --param std=$std libcxx/test/libcxx/transitive_includes.gen.py; done
